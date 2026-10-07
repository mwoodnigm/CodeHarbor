# 可靠性设计与故障演练

这份文档对应四项生产化改造：GitHub API 限流与容错、外部写操作幂等、Agent 断点续跑、证据门禁。每一项都回答一个问题：失败了怎么办、重复执行了怎么办、中断了怎么办、结论凭什么可信。

> 文中 `【】` 标记的数字和截图需要用你自己的环境跑出来再填，不要估算。`scripts/drills/reliability_report.py` 会从真实对话记录里汇总这些数字。

## 1. GitHub API 限流与容错

代码：`backend/app/services/github/client.py`、`backend/app/services/github/resilience.py`

所有 GitHub 请求都经过 `GitHubClient._request`，策略如下。

| 情况 | 读请求（GET） | 写请求（POST/PATCH） | 计入熔断 |
|---|---|---|---|
| 连接失败 / 连接超时 | 退避重试 | 退避重试（请求没发出去） | 是 |
| 读超时 / 其他网络错误 | 退避重试 | **不重试**（可能已执行） | 是 |
| 5xx | 退避重试 | **不重试**（可能已执行） | 是 |
| 429 / 二级限流 403 | 按 Retry-After 或配额重置时间等待后重试 | 同左（服务端明确拒绝） | 否 |
| 其他 4xx | 直接失败 | 直接失败 | 否 |

关键取舍：

- **写请求不盲目重试。** 请求发出后超时或收到 5xx 时，GitHub 可能已经写入成功。这时重放会产生重复评论，所以客户端只抛错，由第 2 节的幂等记录负责对账后安全重放。
- **主动限速。** 每次响应都读取 `x-ratelimit-remaining` / `x-ratelimit-reset`。剩余配额低于阈值时，下一个请求先等到重置时间；如果需要等待的时间超过上限，直接快速失败，不把请求线程挂住。
- **熔断状态按 host 在进程内共享。** 项目里每次调用都会新建 `GitHubClient`，状态挂在实例上就等于没有熔断。连续 N 次失败后熔断打开，冷却后进入半开状态，只放行一个探测请求。
- **退避采用 full jitter**：`sleep = random(0, min(cap, base * 2^attempt))`，避免多个任务同时重试形成尖峰。

配置项见 `.env.example` 中以 `GITHUB_` 开头的变量。

### 故障演练

```bash
cd backend
python ../scripts/drills/drill_github_rate_limit.py
```

这个脚本用 `httpx.MockTransport` 注入 429、502/503、连续 500，依次展示：按 Retry-After 等待、读请求退避成功、写请求不重放、熔断打开后快速失败、半开探测后恢复。

【截图：演练输出】

## 2. 外部写操作幂等

代码：`backend/app/services/action_execution.py`、`ActionExecution` 模型

原来的确认流程是：先检查 `status == pending_confirmation`，然后调用 GitHub，最后才改状态。如果两个确认请求同时到达，两个都能通过检查，PR 下就会出现两条一样的评论。

现在的流程分四步：

1. **原子抢占。** 执行 `UPDATE action_drafts SET status='executing' WHERE id=? AND status='pending_confirmation'`，只有影响行数为 1 的请求能继续。抢占和插入执行记录在同一个事务里提交，不会出现"草稿已 executing 但没有执行记录"的中间态。
2. **先落库再执行。** `ActionExecution` 以 `idempotency_key` 建唯一索引，key 由操作类型、仓库、目标对象和草稿 ID 计算 SHA-256 得到。重复提交时直接返回上次的结果，并在审计日志里记一条 `action_draft.confirm.replayed`。
3. **远端标记。** GitHub REST API 没有 Idempotency-Key 请求头，所以评论和新建 Issue 的正文末尾会嵌入一个隐藏标记 `<!-- codeharbor:idempotency-key=... -->`。
4. **结果未知时先对账。** 写请求超时或收到 5xx 后，执行记录标为 `unknown`。下次确认时，先按标记去 GitHub 查找：找到就直接认领结果，找不到才重放。关闭 Issue、加标签这类操作本身是幂等的，直接重放。进程在执行中途被杀的情况也能处理：执行记录的心跳超过 `ACTION_EXECUTION_STALE_SECONDS` 后，同样走对账流程。

错误分为三类：GitHub 明确拒绝（4xx）记为终态 failed；请求确定没有发出去（熔断、限流、连接失败）记为 aborted，草稿退回待确认，之后可以再点一次；请求可能已执行（写超时、5xx）记为 unknown，走对账流程。

测试 `test_action_idempotency.py` 覆盖了以下场景：并发确认只写一次、超时后对账不重发、5xx 后对账确认未写入再重放、进程被杀后的恢复、未超时的执行记录不会被抢走。这些场景在 SQLite 和 PostgreSQL 16 上都跑通过。

## 3. Agent 断点续跑

代码：`backend/app/services/agents/checkpointing.py`、`chat_agent.py` 中的 `_build_agent_graph` / `_finish_graph_run`

- 后端由 `AGENT_CHECKPOINT_BACKEND` 选择，可选 `memory`、`postgres` 或 `none`。`postgres` 默认复用项目的 `DATABASE_URL`。
- 前端在流式响应的第一个事件 `run_started` 里拿到 `run_id`。连接中断或服务重启后，带上 `resume_run_id` 重新请求 `/api/chat/stream`，Agent 会从最后一个 checkpoint 的下一个节点继续，已经完成的工具调用不会重跑。
- 使用 `durability="sync"`：每一步的 checkpoint 都写完后才进入下一步。
  - 演练中发现，LangGraph 默认的 async 模式下，如果进程在工具执行期间被杀，前几步的 checkpoint 还没来得及写入，恢复时会直接退回起点。
  - 改成 sync 后，恢复点稳定落在被中断的那一步。
- 运行结束后删除该 run 的 checkpoint。checkpoint 只用于中断恢复，不做长期存档。

**状态裁剪。** LangGraph 实际持久化的状态 `GraphState` 只包含循环进度：步数、工具调用记录、观察摘要、结论、计数器。以下内容都不进 checkpoint：

- 工具函数对象：不可序列化。
- 服务端上下文、证据原文、历史消息：恢复时由请求从数据库重建，存进去只是重复数据。
- 流式 UI 事件：改为通过运行时缓冲区推给前端。

每次 run 会在 `agent_events` 里记录一条 `checkpoint_stats`，包含裁剪前后的 JSON 字节数。

单任务状态大小：完整运行时 【X】 KB，checkpoint 【Y】 KB（取真实对话的中位数，由 `reliability_report.py` 统计）。

### 故障演练

```bash
docker compose up -d postgres
cd backend
python ../scripts/drills/drill_checkpoint_resume.py crash    # 第 2 个工具执行时 os._exit(137)
python ../scripts/drills/drill_checkpoint_resume.py resume   # 新进程按同一个 run_id 继续
```

预期结果：恢复点为 `resumed_from_step=2, next_nodes=["tool_action"]`，执行日志里 `search_code` 只出现一次。演练脚本里的模型和工具都是脚本化的，所以每次结果一致。

真实 LLM 下的演练步骤：发起一个需要多步工具调用的问题，在前端看到第二个工具开始执行后，用 Ctrl+C 停掉 uvicorn。重启服务后，带上 `resume_run_id` 重发同一个问题。

【截图：crash / resume 两次输出】

## 4. 证据门禁

代码：`backend/app/services/agents/evidence_gate.py`

- 每次调用模型时，都会在提示词里附上本轮"可引用证据编号"：
  - 预检索证据记为 `[E1]`…
  - 成功的工具调用记为 `[T:tool_use_id]`。
  - 代码位置要求写成 `路径:行号`。
- 最终回答由**代码**逐句检查：
  - 判断句没有有效引用的，标记为 `〔推测·待人工确认〕`。
  - 引用了不存在的编号（即模型编造的），也算无依据，并单独记入 `invalid_citations`。
  - 代码位置的路径必须在本轮工具参数、证据元数据或观察结果里出现过。
  - 问句、标题、代码块，以及自己声明为推测、建议或下一步的句子，不计入判断句。
- `EVIDENCE_GATE_MODE`：
  - `annotate`：标注并统计（默认）。
  - `report`：只统计，不改动回答。
  - `off`：关闭。
- 门禁结果写进最终事件的 `evidence_gate` 字段和 `agent_events`。

在固定评测集上：判断句中有证据的比例 【X%】，无依据的比例 【Y%】；被拦下的编造引用 【N】 处。

## 测试

```bash
cd backend
pytest app/tests/test_github_resilience.py app/tests/test_action_idempotency.py \
       app/tests/test_chat_checkpoint_resume.py app/tests/test_evidence_gate.py -q
```

这四项改造共新增 39 个测试；原有测试全部通过。
