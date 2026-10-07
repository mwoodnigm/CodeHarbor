"""ChatAgent 的 LangGraph 持久化 Checkpointer。

- memory：进程内保存，适合开发和单测，进程退出即丢失。
- postgres：复用项目已有的 PostgreSQL，进程被杀后可以按 run_id 继续。
- none：关闭断点续跑。

状态裁剪原则：checkpoint 只保存"可序列化、无法重建"的循环进度
（步数、工具调用记录、观察摘要、结论），不保存可以按请求重建的运行时依赖
（工具函数、服务端上下文、证据原文、历史消息）和一次性的流式 UI 事件。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver

from app.core.config import settings

logger = logging.getLogger(__name__)

# 恢复时由请求重新构建、不进 checkpoint 的运行时字段
RUNTIME_ONLY_KEYS = ("tools", "context", "tool_name_map")
# 一次性的流式事件，只在当次请求推给前端
TRANSIENT_KEYS = ("stream_events",)

_saver: BaseCheckpointSaver | None = None
_pool: Any = None


def _postgres_dsn() -> str:
    url = settings.agent_checkpoint_database_url or settings.database_url
    return url.replace("postgresql+psycopg://", "postgresql://", 1)


async def get_checkpointer() -> BaseCheckpointSaver | None:
    global _saver, _pool
    backend = (settings.agent_checkpoint_backend or "none").strip().lower()
    if backend in {"", "none", "off", "false"}:
        return None
    if _saver is not None:
        return _saver
    if backend == "memory":
        from langgraph.checkpoint.memory import InMemorySaver

        _saver = InMemorySaver()
        return _saver
    if backend == "postgres":
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import AsyncConnectionPool

        pool = AsyncConnectionPool(
            conninfo=_postgres_dsn(),
            max_size=10,
            open=False,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        )
        await pool.open()
        saver = AsyncPostgresSaver(pool)
        await saver.setup()
        _pool, _saver = pool, saver
        logger.info("Agent checkpointer ready: postgres")
        return _saver
    raise ValueError(f"Unknown AGENT_CHECKPOINT_BACKEND: {settings.agent_checkpoint_backend}")


async def close_checkpointer() -> None:
    global _saver, _pool
    if _pool is not None:
        await _pool.close()
    _pool, _saver = None, None


def reset_checkpointer_for_tests() -> None:
    global _saver, _pool
    _saver, _pool = None, None


def checkpoint_view(state: dict[str, Any]) -> dict[str, Any]:
    """只保留需要持久化的字段。"""
    return {key: value for key, value in state.items() if key not in RUNTIME_ONLY_KEYS and key not in TRANSIENT_KEYS}


def json_size_bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))


def state_size_stats(state: dict[str, Any]) -> dict[str, int]:
    """裁剪前后同口径（JSON 字节数）对比，写进 agent_events 方便 README 引用真实数字。"""
    full = {key: value for key, value in state.items() if key != "tools"}
    full["tools"] = sorted((state.get("tools") or {}).keys())
    return {
        "full_state_bytes": json_size_bytes(full),
        "checkpoint_state_bytes": json_size_bytes(checkpoint_view(state)),
    }
