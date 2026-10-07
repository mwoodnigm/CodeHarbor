from typing import Any

import pytest
from langchain_core.messages import AIMessage

from app.core.config import settings
from app.services.agents.chat_agent import ChatAgent
from app.services.agents.evidence_gate import (
    UNSUPPORTED_TAG,
    EvidenceRegistry,
    apply_evidence_gate,
    build_evidence_registry,
)


def _registry() -> EvidenceRegistry:
    return build_evidence_registry(
        {
            "context": {"evidence": [{"title": "Retry policy", "source_type": "project_doc"}]},
            "tool_calls": [
                {"tool_name": "workspace.read_file", "tool_use_id": "call_read", "status": "success", "arguments": {"path": "backend/auth/refresh.py"}},
                {"tool_name": "workspace.search_code", "tool_use_id": "call_fail", "status": "error", "arguments": {}},
            ],
            "observations": [],
        }
    )


def test_registry_contains_prefetched_evidence_successful_tools_and_paths() -> None:
    registry = _registry()
    assert {"E1", "T:call_read"} <= registry.ids
    assert "T:call_fail" not in registry.ids
    assert registry.has_location("backend/auth/refresh.py")
    assert registry.has_location("auth/refresh.py")
    assert not registry.has_location("payments/charge.py")


def test_cited_claims_pass_and_uncited_claim_is_flagged() -> None:
    answer = (
        "刷新逻辑在重试时没有加锁 [T:call_read]。\n"
        "项目约定重试必须带幂等键 [E1]。\n"
        "并发刷新会写入两个 token，见 `backend/auth/refresh.py:42`。\n"
        "这个问题已经影响了所有生产用户。"
    )
    gated, report = apply_evidence_gate(answer, _registry())

    assert report["supported"] == 3
    assert report["unsupported"] == 1
    assert report["unsupported_ratio"] == 0.25
    assert report["status"] == "needs_review"
    assert gated.splitlines()[-1].endswith(UNSUPPORTED_TAG)
    assert UNSUPPORTED_TAG not in gated.splitlines()[0]


def test_fabricated_citation_id_does_not_count_as_evidence() -> None:
    gated, report = apply_evidence_gate("缓存层会在 30 秒后失效 [E9]。", _registry())

    assert report["unsupported"] == 1
    assert report["invalid_citations"] == ["E9"]
    assert UNSUPPORTED_TAG in gated


def test_code_location_must_have_been_observed() -> None:
    _gated, report = apply_evidence_gate("扣费逻辑重复执行，见 `payments/charge.py:10`。", _registry())
    assert report["unsupported"] == 1


def test_hedges_questions_headings_code_and_next_steps_are_not_claims() -> None:
    answer = "\n".join(
        [
            "## 结论",
            "可能是 Redis 连接池耗尽导致的。",
            "你能提供一下完整日志吗？",
            "```python",
            "def refresh(): pass",
            "```",
            "## 下一步",
            "- 在 refresh 外层加分布式锁并补一个并发测试。",
        ]
    )
    gated, report = apply_evidence_gate(answer, _registry())

    assert report["unsupported"] == 0
    assert report["hedged"] == 2
    assert report["status"] == "passed"
    assert gated == answer


def test_table_rows_are_checked_but_header_and_separator_are_not() -> None:
    answer = "| 风险 | 级别 |\n|---|---|\n| 重试无锁导致重复 token [T:call_read] | 高 |\n| 日志缺少 trace_id | 中 |"
    gated, report = apply_evidence_gate(answer, _registry())

    assert report["supported"] == 1
    assert report["unsupported"] == 1
    assert gated.splitlines()[-1].endswith(f"{UNSUPPORTED_TAG} |")


def test_report_mode_measures_without_changing_text() -> None:
    answer = "这个问题已经影响了所有生产用户。"
    gated, report = apply_evidence_gate(answer, _registry(), annotate=False)
    assert gated == answer
    assert report["unsupported"] == 1


class OneToolThenAnswer:
    def __init__(self, final: str) -> None:
        self.calls = 0
        self.final = final
        self.prompts: list[str] = []

    def bind_tools(self, tools: list[Any]) -> "OneToolThenAnswer":
        return self

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        self.calls += 1
        self.prompts.append("\n".join(str(message.content) for message in messages))
        if self.calls == 1:
            return AIMessage(content="", tool_calls=[{"name": "workspace__read_file", "args": {"path": "auth/refresh.py"}, "id": "call_read"}])
        return AIMessage(content=self.final)


async def _read_file(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    return {"route": "workspace_read_file", "answer": "auth/refresh.py:40-44\n40: retry()", "citations": []}


@pytest.mark.asyncio
async def test_agent_lists_citable_ids_in_prompt_and_gates_final_answer(monkeypatch) -> None:
    monkeypatch.setattr(settings, "evidence_gate_mode", "annotate")
    model = OneToolThenAnswer("重试路径没有加锁 [T:call_read]。\n所有用户都受到了影响。")

    final = await ChatAgent(model=model).run(  # type: ignore[arg-type]
        {"message": "Investigate the token refresh failure in the auth module."},
        {"tools": {"workspace.read_file": _read_file}},
    )

    assert "[T:call_read] workspace.read_file" in model.prompts[-1]
    assert final["evidence_gate"]["supported"] == 1
    assert final["evidence_gate"]["unsupported"] == 1
    assert final["answer"].endswith(UNSUPPORTED_TAG)
    assert any(event["type"] == "evidence_gate" for event in final["agent_events"])


@pytest.mark.asyncio
async def test_gate_off_leaves_answer_untouched(monkeypatch) -> None:
    monkeypatch.setattr(settings, "evidence_gate_mode", "off")
    model = OneToolThenAnswer("所有用户都受到了影响。")

    final = await ChatAgent(model=model).run(  # type: ignore[arg-type]
        {"message": "Investigate the token refresh failure in the auth module."},
        {"tools": {"workspace.read_file": _read_file}},
    )

    assert final["answer"] == "所有用户都受到了影响。"
    assert final["evidence_gate"] == {}
    assert "可引用证据编号" not in model.prompts[-1]
