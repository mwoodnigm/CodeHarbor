"""ActionDraft 的幂等执行：原子抢占 -> 先落库 -> 调 GitHub -> 回写结果。

GitHub REST API 没有 Idempotency-Key 请求头，所以幂等分两层：
1. 本地层：草稿状态原子抢占（pending_confirmation -> executing）加 ActionExecution
   唯一索引，保证同一草稿只有一个请求真正去调 GitHub；重复提交直接返回上次结果。
2. 远端层：评论和新建 Issue 的正文里嵌入隐藏标记 <!-- codeharbor:idempotency-key=... -->。
   如果写请求超时或 5xx（结果未知），下次确认时先按标记去 GitHub 对账：
   找到就直接认领结果，找不到才重放。关闭 Issue、加标签本身幂等，直接重放即可。
"""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from fastapi import HTTPException
from sqlalchemy import or_, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import decrypt_token
from app.db.models import ActionDraft, ActionExecution, AuditLog, Issue, PullRequest, Repository, User
from app.services.github.client import GitHubClient
from app.services.github.resilience import GitHubCircuitOpenError, GitHubRateLimitExceeded
from app.services.permissions import write_audit_log

MARKER_PREFIX = "codeharbor:idempotency-key="
COMMENT_TYPES = {"issue_comment", "pr_comment", "comment"}
LABEL_TYPES = {"label", "add_label", "add_labels"}
# 重放不会产生副作用叠加的操作
NATURALLY_IDEMPOTENT_TYPES = {"close_issue", "send_report", *LABEL_TYPES}

ClientFactory = Callable[[Repository], GitHubClient]


class OutcomeUnknownError(RuntimeError):
    """写请求可能已被 GitHub 处理，但没有拿到确定结果。"""


class NotExecutedError(RuntimeError):
    """请求确定没有到达 GitHub，可以安全地稍后重试。"""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def target_number(db: Session, draft: ActionDraft) -> int | None:
    meta = draft.meta or {}
    raw = meta.get("target_number") or meta.get("issue_number") or meta.get("pr_number")
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None
    if draft.target_id and draft.target_type == "issue":
        issue = db.get(Issue, draft.target_id)
        return issue.number if issue else None
    if draft.target_id and draft.target_type in {"pull_request", "pr"}:
        pr = db.get(PullRequest, draft.target_id)
        return pr.number if pr else None
    return None


def build_idempotency_key(draft: ActionDraft, repo: Repository, number: int | None) -> str:
    """操作类型 + 仓库 + 目标对象 + 草稿 ID。同一草稿永远得到同一个 key。"""
    target = str(number) if number is not None else str(draft.target_id or "-")
    raw = "|".join([draft.draft_type.lower(), repo.full_name.lower(), target, str(draft.id)])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def idempotency_marker(key: str) -> str:
    return f"<!-- {MARKER_PREFIX}{key} -->"


def with_marker(body: str, key: str) -> str:
    marker = idempotency_marker(key)
    return body if marker in (body or "") else f"{body or ''}\n\n{marker}"


def default_client_factory(repo: Repository) -> GitHubClient:
    token = decrypt_token(repo.github_token_encrypted, settings.token_encryption_key) if repo.github_token_encrypted else None
    return GitHubClient(token=token, base_url=repo.api_base_url)


async def perform_github_write(client: GitHubClient, db: Session, draft: ActionDraft, repo: Repository, key: str) -> dict[str, Any]:
    draft_type = draft.draft_type.lower()
    number = target_number(db, draft)
    if draft_type in COMMENT_TYPES:
        if number is None:
            raise ValueError("评论草稿缺少 issue/pr number")
        return await client.create_issue_comment(repo.owner, repo.name, number, with_marker(draft.content, key))
    if draft_type == "create_issue":
        title = draft.title or (draft.content.splitlines()[0][:120] if draft.content else "CodeHarbor Issue")
        return await client.create_issue(repo.owner, repo.name, title, with_marker(draft.content, key))
    if draft_type == "close_issue":
        if number is None:
            raise ValueError("关闭 Issue 草稿缺少 issue number")
        return await client.close_issue(repo.owner, repo.name, number)
    if draft_type in LABEL_TYPES:
        if number is None:
            raise ValueError("Label 草稿缺少 issue/pr number")
        labels = draft.meta.get("labels") if draft.meta else None
        if not isinstance(labels, list) or not labels:
            labels = [draft.title] if draft.title else []
        if not labels:
            raise ValueError("Label 草稿缺少 labels")
        return await client.add_issue_labels(repo.owner, repo.name, number, [str(item) for item in labels])
    if draft_type == "send_report":
        return {"status": "recorded", "message": "报告草稿已标记为已发送，未接入外部消息渠道。"}
    raise ValueError(f"暂不支持执行草稿类型：{draft.draft_type}")


async def find_remote_result(client: GitHubClient, db: Session, draft: ActionDraft, repo: Repository, key: str) -> dict[str, Any] | None:
    """按隐藏标记在 GitHub 上查找上次可能已经成功的写入。"""
    marker = idempotency_marker(key)
    draft_type = draft.draft_type.lower()
    if draft_type in COMMENT_TYPES:
        number = target_number(db, draft)
        if number is None:
            return None
        items = await client.list_issue_comments(repo.owner, repo.name, number)
    elif draft_type == "create_issue":
        items = await client.list_recent_issues(repo.owner, repo.name)
    else:
        return None
    for item in items or []:
        if isinstance(item, dict) and marker in str(item.get("body") or ""):
            return item
    return None


def classify_write_error(exc: Exception) -> str:
    """把异常归为 failed（GitHub 明确拒绝）、not_executed（没发出去）或 unknown（可能已执行）。"""
    if isinstance(exc, (GitHubCircuitOpenError, GitHubRateLimitExceeded, httpx.ConnectError, httpx.ConnectTimeout)):
        return "not_executed"
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status == 429:
            return "not_executed"
        return "unknown" if status >= 500 else "failed"
    if isinstance(exc, httpx.TransportError):
        return "unknown"
    return "failed"


def _audit(db: Session, user: User | None, draft: ActionDraft, action: str, status: str, key: str, result: dict[str, Any]) -> AuditLog:
    return write_audit_log(
        db,
        user=user,
        repo_id=draft.repo_id,
        action=action,
        target_type=draft.target_type,
        target_id=str(draft.target_id or draft.id),
        status=status,
        request_json={"draft_id": str(draft.id), "draft_type": draft.draft_type, "idempotency_key": key},
        result_json=result,
    )


async def confirm_draft_idempotently(
    db: Session,
    draft_id,
    user: User,
    *,
    client_factory: ClientFactory = default_client_factory,
) -> tuple[ActionDraft, AuditLog]:
    draft = db.get(ActionDraft, draft_id)
    if draft is None:
        raise HTTPException(status_code=404, detail="未找到安全草稿")
    repo = db.get(Repository, draft.repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")

    key = build_idempotency_key(draft, repo, target_number(db, draft))

    # 第一步：原子抢占。并发的两个确认请求只有一个能把状态从 pending 改成 executing。
    # 抢占和执行记录在同一个事务里提交：不会出现"草稿已 executing 却没有执行记录"的中间态
    claimed = db.execute(
        update(ActionDraft)
        .where(ActionDraft.id == draft.id, ActionDraft.status == "pending_confirmation")
        .values(status="executing", confirmed_by=user.id)
        .execution_options(synchronize_session=False)
    ).rowcount
    if not claimed:
        db.rollback()
        db.refresh(draft)
        return await _handle_already_claimed(db, draft, repo, user, key, client_factory)

    # 第二步：先落库再执行。唯一索引是并发的最后一道防线
    execution = db.query(ActionExecution).filter(ActionExecution.draft_id == draft.id).one_or_none()
    now = _utcnow()
    if execution is None:
        execution = ActionExecution(
            idempotency_key=key,
            draft_id=draft.id,
            repo_id=repo.id,
            action_type=draft.draft_type.lower(),
            target=f"{repo.full_name}#{target_number(db, draft) or draft.target_id or '-'}",
            status="in_progress",
            attempts=1,
            heartbeat_at=now,
        )
        db.add(execution)
    else:
        # 上次 aborted（确定没发出去），复用同一个 key 重新执行
        execution.status = "in_progress"
        execution.attempts = (execution.attempts or 0) + 1
        execution.error = None
        execution.heartbeat_at = now
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        db.refresh(draft)
        return await _handle_already_claimed(db, draft, repo, user, key, client_factory)
    db.refresh(draft)

    return await _execute(db, draft, repo, user, execution, key, client_factory, replay_reason=None)


async def _execute(
    db: Session,
    draft: ActionDraft,
    repo: Repository,
    user: User,
    execution: ActionExecution,
    key: str,
    client_factory: ClientFactory,
    *,
    replay_reason: str | None,
) -> tuple[ActionDraft, AuditLog]:
    client = client_factory(repo)
    try:
        result = await perform_github_write(client, db, draft, repo, key)
    except Exception as exc:  # noqa: BLE001 - 需要按类型落库
        return _record_failure(db, draft, user, execution, key, exc)

    now = _utcnow()
    execution.status = "succeeded"
    execution.result_json = result if isinstance(result, dict) else {"result": result}
    execution.completed_at = now
    draft.status = "executed"
    draft.executed_at = now
    draft.execution_result = execution.result_json
    draft.error_message = None
    audit_result = dict(execution.result_json)
    if replay_reason:
        audit_result["replay_reason"] = replay_reason
    audit = _audit(db, user, draft, f"action_draft.confirm.{draft.draft_type}", "success", key, audit_result)
    db.commit()
    db.refresh(draft)
    db.refresh(audit)
    return draft, audit


def _record_failure(
    db: Session,
    draft: ActionDraft,
    user: User,
    execution: ActionExecution,
    key: str,
    exc: Exception,
) -> tuple[ActionDraft, AuditLog]:
    kind = classify_write_error(exc)
    now = _utcnow()
    execution.error = str(exc)
    execution.heartbeat_at = now
    if kind == "not_executed":
        # 确定没发出去：草稿退回待确认，用户稍后可以再点一次
        execution.status = "aborted"
        draft.status = "pending_confirmation"
        draft.error_message = f"本次未执行，可稍后重试：{exc}"
    elif kind == "unknown":
        # 可能已执行：保持 executing，下次确认先对账
        execution.status = "unknown"
        draft.status = "executing"
        draft.error_message = f"执行结果未知，再次确认会先到 GitHub 对账：{exc}"
    else:
        execution.status = "failed"
        execution.completed_at = now
        draft.status = "failed"
        draft.executed_at = now
        draft.error_message = str(exc)
    audit = _audit(
        db,
        user,
        draft,
        f"action_draft.confirm.{draft.draft_type}",
        {"not_executed": "aborted", "unknown": "unknown"}.get(kind, "failed"),
        key,
        {"error": str(exc), "error_kind": kind, "attempts": execution.attempts},
    )
    db.commit()
    db.refresh(draft)
    db.refresh(audit)
    return draft, audit


async def _handle_already_claimed(
    db: Session,
    draft: ActionDraft,
    repo: Repository,
    user: User,
    key: str,
    client_factory: ClientFactory,
) -> tuple[ActionDraft, AuditLog]:
    if draft.status in {"executed", "failed"}:
        # 重复提交：不再调用 GitHub，直接返回上次的结果
        audit = _audit(
            db,
            user,
            draft,
            "action_draft.confirm.replayed",
            "success",
            key,
            {"replayed_status": draft.status, "previous_result": draft.execution_result or {}},
        )
        db.commit()
        db.refresh(audit)
        return draft, audit
    if draft.status != "executing":
        raise HTTPException(status_code=400, detail="只有待确认草稿可以执行")

    execution = db.query(ActionExecution).filter(ActionExecution.draft_id == draft.id).one_or_none()
    if execution is None:
        raise HTTPException(status_code=409, detail="草稿正在执行中，请稍后刷新")

    cutoff = _utcnow() - timedelta(seconds=settings.action_execution_stale_seconds)
    # 只有一个请求能拿到对账权：unknown，或 in_progress 但心跳超时（进程中途被杀）
    won = db.execute(
        update(ActionExecution)
        .where(
            ActionExecution.id == execution.id,
            or_(
                ActionExecution.status == "unknown",
                (ActionExecution.status == "in_progress") & (ActionExecution.heartbeat_at < cutoff),
            ),
        )
        .values(status="reconciling", attempts=ActionExecution.attempts + 1, heartbeat_at=_utcnow())
        .execution_options(synchronize_session=False)
    ).rowcount
    db.commit()
    if not won:
        raise HTTPException(status_code=409, detail="草稿正在执行中，请稍后刷新")
    db.refresh(execution)
    return await _reconcile(db, draft, repo, user, execution, key, client_factory)


async def _reconcile(
    db: Session,
    draft: ActionDraft,
    repo: Repository,
    user: User,
    execution: ActionExecution,
    key: str,
    client_factory: ClientFactory,
) -> tuple[ActionDraft, AuditLog]:
    client = client_factory(repo)
    if draft.draft_type.lower() not in NATURALLY_IDEMPOTENT_TYPES:
        try:
            found = await find_remote_result(client, db, draft, repo, key)
        except Exception as exc:  # noqa: BLE001
            # 对账本身失败，不能贸然重放，退回 unknown 等下次
            execution.status = "unknown"
            execution.error = f"对账失败：{exc}"
            draft.error_message = f"对账失败，稍后再次确认：{exc}"
            audit = _audit(db, user, draft, "action_draft.reconcile", "unknown", key, {"error": str(exc)})
            db.commit()
            db.refresh(draft)
            db.refresh(audit)
            return draft, audit
        if found is not None:
            now = _utcnow()
            execution.status = "succeeded"
            execution.result_json = found
            execution.completed_at = now
            draft.status = "executed"
            draft.executed_at = now
            draft.execution_result = found
            draft.error_message = None
            audit = _audit(
                db,
                user,
                draft,
                "action_draft.reconcile",
                "success",
                key,
                {"reconciled": "found_remote", "remote_id": found.get("id"), "html_url": found.get("html_url")},
            )
            db.commit()
            db.refresh(draft)
            db.refresh(audit)
            return draft, audit
    # 远端确认没写成功（或本身幂等）：用同一个 key 重放
    return await _execute(db, draft, repo, user, execution, key, client_factory, replay_reason="reconciled_not_found")
