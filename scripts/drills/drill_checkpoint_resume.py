"""故障演练：Agent 跑到一半进程被杀，重启后从断点继续。

需要 PostgreSQL（docker compose up -d postgres）。在 backend 目录下运行：

    set PYTHONPATH=.            (PowerShell: $env:PYTHONPATH=".")
    python ../scripts/drills/drill_checkpoint_resume.py crash    # 第 2 个工具执行时 os._exit(137)
    python ../scripts/drills/drill_checkpoint_resume.py resume   # 新进程按同一个 run_id 继续

模型和工具是脚本化的假实现，保证每次演练结果一致；真实 LLM 下的演练见 docs/RELIABILITY.md。
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage

from app.core.config import settings
from app.services.agents.chat_agent import ChatAgent
from app.services.agents.checkpointing import close_checkpointer

RUN_ID = os.environ.get("DRILL_RUN_ID", "drill-checkpoint-resume-001")
LOG = Path(os.environ.get("DRILL_LOG", ".data/drill_tool_calls.log"))
PLAN = [
    ("workspace__search_code", {"query": "refresh_token"}),
    ("workspace__read_file", {"path": "auth/refresh.py"}),
    ("workspace__read_file", {"path": "auth/session.py"}),
]


class ScriptedModel:
    """按已完成的工具数决定下一步，和真实模型一样只依赖输入状态。"""

    def bind_tools(self, tools: list[Any]) -> "ScriptedModel":
        return self

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        prompt = "\n".join(str(message.content) for message in messages)
        done = prompt.count('"type": "tool_result"')
        if done < len(PLAN):
            name, args = PLAN[done]
            return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{done + 1}"}])
        return AIMessage(content="排查完成：refresh 重试路径没有加锁 [T:call_2]。")


def make_tools(phase: str) -> dict[str, Any]:
    def log(name: str, args: dict[str, Any]) -> None:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"phase": phase, "tool": name, "args": args}, ensure_ascii=False) + "\n")

    async def search_code(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
        log("search_code", args)
        return {"route": "workspace_search_code", "answer": "auth/refresh.py:42 refresh_token()"}

    async def read_file(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
        log("read_file", args)
        if phase == "crash" and args.get("path") == "auth/refresh.py":
            print(">>> 模拟进程被杀：os._exit(137)", flush=True)
            os._exit(137)
        return {"route": "workspace_read_file", "answer": f"{args.get('path')}:40-44 retry() without lock"}

    return {"workspace.search_code": search_code, "workspace.read_file": read_file}


async def main(phase: str) -> None:
    settings.agent_checkpoint_backend = "postgres"
    if phase == "crash" and LOG.exists():
        LOG.unlink()
    agent = ChatAgent(model=ScriptedModel())  # type: ignore[arg-type]
    final: dict[str, Any] = {}
    try:
        async for event in agent.run_stream(
            {"message": "Investigate the token refresh failure in the auth module.", "checkpoint_thread_id": RUN_ID},
            {"tools": make_tools(phase)},
        ):
            if event["event"] == "checkpoint_resumed":
                print(">>> 从断点恢复：", json.dumps(event["data"], ensure_ascii=False))
            if event["event"] == "final":
                final = event["data"]
    finally:
        await close_checkpointer()
    print(">>> 最终回答：", final.get("answer"))
    print(">>> 本次 run 全部工具调用：", [call["tool_name"] for call in final.get("tool_calls", [])])
    stats = next((e for e in final.get("agent_events", []) if e.get("type") == "checkpoint_stats"), None)
    if stats:
        print(f">>> 状态大小：完整运行时 {stats['full_state_bytes']} B，checkpoint {stats['checkpoint_state_bytes']} B")
    print(">>> 实际执行日志（search_code 只应出现一次）：")
    print(LOG.read_text(encoding="utf-8") if LOG.exists() else "(empty)")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"crash", "resume"}:
        raise SystemExit("usage: drill_checkpoint_resume.py crash|resume")
    if sys.platform == "win32":
        # Psycopg async connections require SelectorEventLoop on Windows.
        # Keep this policy local to the drill; the app also runs MCP subprocesses.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main(sys.argv[1]))
