import asyncio
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import ActionDraft, ActionExecution, AuditLog, Base, Repository, User
from app.services.action_execution import (
    build_idempotency_key,
    classify_write_error,
    confirm_draft_idempotently,
    idempotency_marker,
)
from app.services.github.resilience import GitHubCircuitOpenError


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://api.github.test/repos/acme/api/issues/7/comments")


class FakeGitHub:
    """记录真实写入次数；可以注入失败或延迟，模拟网络抖动和并发。"""

    def __init__(self) -> None:
        self.comments: list[dict] = []
        self.write_calls = 0
        self.fail_with: list[Exception] = []
        self.persist_before_failing = False
        self.gate: asyncio.Event | None = None

    async def create_issue_comment(self, owner, repo, number, body):
        self.write_calls += 1
        if self.gate is not None:
            await self.gate.wait()
        comment = {"id": 1000 + len(self.comments), "body": body, "html_url": f"https://github.test/{owner}/{repo}/issues/{number}#c"}
        if self.fail_with:
            exc = self.fail_with.pop(0)
            if self.persist_before_failing:
                # 模拟：GitHub 已经写入成功，但响应在回程丢了
                self.comments.append(comment)
            raise exc
        self.comments.append(comment)
        return comment

    async def list_issue_comments(self, owner, repo, number, per_page=100):
        return list(reversed(self.comments))

    async def close_issue(self, owner, repo, number):
        self.write_calls += 1
        return {"number": number, "state": "closed"}


@pytest.fixture()
def env():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False)
    db = Session()
    user = User(name="reviewer", role="owner")
    repo = Repository(owner="acme", name="api", full_name="acme/api")
    db.add_all([user, repo])
    db.flush()
    draft = ActionDraft(
        repo_id=repo.id,
        draft_type="issue_comment",
        target_type="issue",
        title="分析结论",
        content="复现步骤已确认，根因在 token 刷新。",
        meta={"issue_number": 7},
    )
    db.add(draft)
    db.commit()
    github = FakeGitHub()
    return {"Session": Session, "db": db, "user": user, "repo": repo, "draft": draft, "github": github}


def _confirm(env, session=None):
    db = session or env["db"]
    user = db.get(User, env["user"].id)
    return confirm_draft_idempotently(db, env["draft"].id, user, client_factory=lambda _repo: env["github"])


@pytest.mark.asyncio
async def test_repeat_confirm_returns_previous_result_without_second_comment(env) -> None:
    draft, first_audit = await _confirm(env)
    assert draft.status == "executed"

    draft_again, replay_audit = await _confirm(env)

    assert env["github"].write_calls == 1
    assert len(env["github"].comments) == 1
    assert draft_again.execution_result["id"] == draft.execution_result["id"]
    assert replay_audit.action == "action_draft.confirm.replayed"
    assert replay_audit.id != first_audit.id


@pytest.mark.asyncio
async def test_comment_body_carries_hidden_idempotency_marker(env) -> None:
    await _confirm(env)
    key = env["db"].query(ActionExecution).one().idempotency_key

    assert idempotency_marker(key) in env["github"].comments[0]["body"]
    assert key == build_idempotency_key(env["draft"], env["repo"], 7)


@pytest.mark.asyncio
async def test_concurrent_confirms_only_write_once(env) -> None:
    env["github"].gate = asyncio.Event()
    session_a = env["Session"]()
    session_b = env["Session"]()

    task_a = asyncio.create_task(_confirm(env, session_a))
    await asyncio.sleep(0)  # A 已抢占并卡在 GitHub 调用上
    with pytest.raises(HTTPException) as exc_info:
        await _confirm(env, session_b)
    env["github"].gate.set()
    draft, _audit = await task_a

    assert exc_info.value.status_code == 409
    assert draft.status == "executed"
    assert env["github"].write_calls == 1


@pytest.mark.asyncio
async def test_not_executed_error_returns_draft_to_pending_and_retry_reuses_key(env) -> None:
    env["github"].fail_with = [GitHubCircuitOpenError("api.github.test", 30)]

    draft, audit = await _confirm(env)
    assert draft.status == "pending_confirmation"
    assert audit.status == "aborted"
    key = env["db"].query(ActionExecution).one().idempotency_key

    draft, _ = await _confirm(env)
    execution = env["db"].query(ActionExecution).one()
    assert draft.status == "executed"
    assert execution.idempotency_key == key
    assert execution.attempts == 2
    assert len(env["github"].comments) == 1


@pytest.mark.asyncio
async def test_ambiguous_timeout_then_reconcile_finds_remote_comment_and_does_not_repost(env) -> None:
    env["github"].fail_with = [httpx.ReadTimeout("response lost", request=_request())]
    env["github"].persist_before_failing = True

    draft, audit = await _confirm(env)
    assert draft.status == "executing"
    assert audit.status == "unknown"
    assert len(env["github"].comments) == 1

    draft, audit = await _confirm(env)

    assert draft.status == "executed"
    assert audit.action == "action_draft.reconcile"
    assert audit.result_json["reconciled"] == "found_remote"
    assert env["github"].write_calls == 1
    assert len(env["github"].comments) == 1


@pytest.mark.asyncio
async def test_ambiguous_5xx_then_reconcile_replays_when_remote_has_nothing(env) -> None:
    response = httpx.Response(502, request=_request())
    env["github"].fail_with = [httpx.HTTPStatusError("bad gateway", request=_request(), response=response)]

    draft, _ = await _confirm(env)
    assert draft.status == "executing"
    assert env["github"].comments == []

    draft, audit = await _confirm(env)

    assert draft.status == "executed"
    assert audit.result_json["replay_reason"] == "reconciled_not_found"
    assert len(env["github"].comments) == 1


@pytest.mark.asyncio
async def test_process_killed_mid_execution_is_recovered_after_stale_timeout(env) -> None:
    db = env["db"]
    draft = env["draft"]
    # 模拟：抢占和落库已经完成，进程在调用 GitHub 前后被杀
    draft.status = "executing"
    db.add(
        ActionExecution(
            idempotency_key=build_idempotency_key(draft, env["repo"], 7),
            draft_id=draft.id,
            repo_id=env["repo"].id,
            action_type="issue_comment",
            status="in_progress",
            heartbeat_at=datetime.now(timezone.utc) - timedelta(minutes=10),
        )
    )
    db.commit()

    draft, _ = await _confirm(env)

    assert draft.status == "executed"
    assert len(env["github"].comments) == 1


@pytest.mark.asyncio
async def test_fresh_in_progress_execution_is_not_hijacked(env) -> None:
    db = env["db"]
    draft = env["draft"]
    draft.status = "executing"
    db.add(
        ActionExecution(
            idempotency_key=build_idempotency_key(draft, env["repo"], 7),
            draft_id=draft.id,
            repo_id=env["repo"].id,
            action_type="issue_comment",
            status="in_progress",
            heartbeat_at=datetime.now(timezone.utc),
        )
    )
    db.commit()

    with pytest.raises(HTTPException) as exc_info:
        await _confirm(env)
    assert exc_info.value.status_code == 409
    assert env["github"].write_calls == 0


@pytest.mark.asyncio
async def test_github_rejection_is_terminal_failure(env) -> None:
    response = httpx.Response(422, request=_request())
    env["github"].fail_with = [httpx.HTTPStatusError("unprocessable", request=_request(), response=response)]

    draft, audit = await _confirm(env)

    assert draft.status == "failed"
    assert audit.status == "failed"
    assert env["db"].query(AuditLog).count() == 1


def test_error_classification() -> None:
    req = _request()
    assert classify_write_error(httpx.ConnectError("refused")) == "not_executed"
    assert classify_write_error(httpx.ReadTimeout("slow")) == "unknown"
    assert classify_write_error(httpx.HTTPStatusError("x", request=req, response=httpx.Response(503, request=req))) == "unknown"
    assert classify_write_error(httpx.HTTPStatusError("x", request=req, response=httpx.Response(429, request=req))) == "not_executed"
    assert classify_write_error(httpx.HTTPStatusError("x", request=req, response=httpx.Response(404, request=req))) == "failed"
    assert classify_write_error(ValueError("bad draft")) == "failed"
