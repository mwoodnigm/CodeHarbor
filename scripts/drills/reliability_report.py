"""从真实对话记录里汇总可写进 README / 简历的数字（只统计你实际跑过的数据）。

在 backend 目录下运行：python ../scripts/drills/reliability_report.py [--repo owner/name] [--since 2026-10-01]
输出：
- 证据门禁：判断句总数、有引用比例、无依据比例、编造引用次数
- 断点续跑：checkpoint 前后的单任务状态大小（中位数）
- 写操作幂等：执行记录状态分布、被拦下的重复提交次数、对账次数
"""

import argparse
import statistics
from datetime import datetime

from app.db.models import ActionExecution, AuditLog, ChatMessage, Repository
from app.db.session import SessionLocal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo")
    parser.add_argument("--since")
    args = parser.parse_args()
    db = SessionLocal()
    query = db.query(ChatMessage).filter(ChatMessage.role == "assistant")
    if args.repo:
        repo = db.query(Repository).filter(Repository.full_name == args.repo).one()
        query = query.filter(ChatMessage.repo_id == repo.id)
    if args.since:
        query = query.filter(ChatMessage.created_at >= datetime.fromisoformat(args.since))

    gates, sizes = [], []
    for message in query.all():
        for event in (message.meta or {}).get("agent_events") or []:
            if event.get("type") == "evidence_gate":
                gates.append(event)
            elif event.get("type") == "checkpoint_stats":
                sizes.append(event)

    print(f"== 证据门禁（{len(gates)} 条回答）==")
    claims = sum(item.get("claims", 0) for item in gates)
    supported = sum(item.get("supported", 0) for item in gates)
    unsupported = sum(item.get("unsupported", 0) for item in gates)
    invalid = sum(len(item.get("invalid_citations") or []) for item in gates)
    if claims:
        print(f"判断句 {claims}，有证据 {supported}（{supported / claims:.1%}），无依据 {unsupported}（{unsupported / claims:.1%}）")
        print(f"被拦下的编造引用 {invalid} 处；需要人工确认的回答 {sum(1 for g in gates if g.get('status') == 'needs_review')} 条")

    print(f"\n== 断点续跑状态大小（{len(sizes)} 次 run）==")
    if sizes:
        full = statistics.median(item["full_state_bytes"] for item in sizes) / 1024
        kept = statistics.median(item["checkpoint_state_bytes"] for item in sizes) / 1024
        print(f"中位数：完整运行时状态 {full:.1f} KB -> checkpoint {kept:.1f} KB")

    print("\n== 写操作幂等 ==")
    for status in ["succeeded", "failed", "aborted", "unknown", "reconciling", "in_progress"]:
        count = db.query(ActionExecution).filter(ActionExecution.status == status).count()
        if count:
            print(f"执行记录 {status}: {count}")
    print(f"拦下的重复提交：{db.query(AuditLog).filter(AuditLog.action == 'action_draft.confirm.replayed').count()} 次")
    print(f"结果未知后的对账：{db.query(AuditLog).filter(AuditLog.action == 'action_draft.reconcile').count()} 次")


if __name__ == "__main__":
    main()
