from typing import Any

import pytest
from langchain_core.messages import AIMessage

from app.core.config import settings
from app.services.agents.chat_agent import ChatAgent
from app.services.agents.checkpointing import checkpoint_view, get_checkpointer, reset_checkpointer_for_tests


class SimulatedCrash(BaseException):
    """模拟进程被杀：不是 Exception，工具层的重试和兜底都拦不住。"""


class SharedScriptModel:
    """两次运行共享同一份脚本，模拟重启后模型继续按原计划回答。"""

    def __init__(self, script: list[AIMessage]) -> None:
        self.script = script
        self.calls = 0

    def bind_tools(self, tools: list[Any]) -> "SharedScriptModel":
        return self

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        self.calls += 1
        return self.script.pop(0)


def _tool_call(name: str, call_id: str, args: dict[str, Any]) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


@pytest.fixture(autouse=True)
def _memory_checkpointer(monkeypatch):
    monkeypatch.setattr(settings, "agent_checkpoint_backend", "memory")
    monkeypatch.setattr(settings, "evidence_gate_mode", "off")
    reset_checkpointer_for_tests()
    yield
    reset_checkpointer_for_tests()


def _tools(invocations: list[str], *, crash_on_read: bool) -> dict[str, Any]:
    async def search_code(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        invocations.append("search")
        return {"route": "workspace_search_code", "answer": "auth/refresh.py:42 refresh_token()", "citations": []}

    async def read_file(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        invocations.append("read")
        if crash_on_read:
            raise SimulatedCrash()
        return {"route": "workspace_read_file", "answer": "auth/refresh.py:40-44 retries without lock", "citations": []}

    return {"workspace.search_code": search_code, "workspace.read_file": read_file}


@pytest.mark.asyncio
async def test_killed_run_resumes_from_checkpoint_without_rerunning_finished_tools() -> None:
    script = [
        _tool_call("workspace__search_code", "call_search", {"query": "refresh_token"}),
        _tool_call("workspace__read_file", "call_read", {"path": "auth/refresh.py"}),
        AIMessage(content="根因是 refresh 重试没有加锁。"),
    ]
    model = SharedScriptModel(script)
    invocations: list[str] = []
    message = {"message": "Investigate the token refresh failure in the auth module.", "checkpoint_thread_id": "run-1"}

    with pytest.raises(SimulatedCrash):
        async for _event in ChatAgent(model=model).run_stream(  # type: ignore[arg-type]
            message, {"tools": _tools(invocations, crash_on_read=True)}
        ):
            pass
    assert invocations == ["search", "read"]

    # "重启"：全新的 Agent 实例和工具，只靠 checkpoint 恢复
    events = [
        event
        async for event in ChatAgent(model=model).run_stream(  # type: ignore[arg-type]
            message, {"tools": _tools(invocations, crash_on_read=False)}
        )
    ]
    final = [event for event in events if event["event"] == "final"][-1]["data"]

    assert invocations == ["search", "read", "read"]  # search 没有重跑
    assert model.calls == 3  # 两次运行合计正好消耗完脚本，没有多余的模型调用
    assert final["answer"] == "根因是 refresh 重试没有加锁。"
    assert [call["tool_name"] for call in final["tool_calls"]] == ["workspace.search_code", "workspace.read_file"]
    resumed = next(event for event in final["agent_events"] if event["type"] == "checkpoint_resumed")
    assert resumed["completed_tool_calls"] == 1
    assert resumed["next_nodes"] == ["tool_action"]
    assert any(event["event"] == "checkpoint_resumed" for event in events)


@pytest.mark.asyncio
async def test_completed_run_records_state_size_and_cleans_up_checkpoint() -> None:
    model = SharedScriptModel(
        [
            _tool_call("workspace__search_code", "call_search", {"query": "refresh_token"}),
            AIMessage(content="done"),
        ]
    )
    invocations: list[str] = []
    evidence = [{"title": f"doc {i}", "snippet": "x" * 2000, "source_type": "project_doc"} for i in range(5)]
    events = [
        event
        async for event in ChatAgent(model=model).run_stream(  # type: ignore[arg-type]
            {"message": "Investigate the token refresh failure in the auth module.", "checkpoint_thread_id": "run-2"},
            {"tools": _tools(invocations, crash_on_read=False), "evidence": evidence, "server_context": "ctx " * 2000},
        )
    ]
    final = [event for event in events if event["event"] == "final"][-1]["data"]
    stats = next(event for event in final["agent_events"] if event["type"] == "checkpoint_stats")

    assert stats["checkpoint_state_bytes"] < stats["full_state_bytes"]
    saver = await get_checkpointer()
    assert await saver.aget_tuple({"configurable": {"thread_id": "run-2"}}) is None


@pytest.mark.asyncio
async def test_without_thread_id_no_checkpoint_is_used() -> None:
    model = SharedScriptModel([AIMessage(content="plain answer")])
    events = [
        event
        async for event in ChatAgent(model=model).run_stream(  # type: ignore[arg-type]
            {"message": "Investigate the token refresh failure in the auth module."},
            {"tools": _tools([], crash_on_read=False)},
        )
    ]
    final = [event for event in events if event["event"] == "final"][-1]["data"]
    assert final["answer"] == "plain answer"
    assert not any(event["type"] == "checkpoint_stats" for event in final["agent_events"])


def test_checkpoint_view_drops_runtime_and_transient_keys() -> None:
    view = checkpoint_view({"message": "m", "tools": {"x": object()}, "context": {"evidence": []}, "tool_name_map": {}, "stream_events": [1], "step": 2})
    assert view == {"message": "m", "step": 2}
