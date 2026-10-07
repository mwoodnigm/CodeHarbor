"""GitHub API 容错：主动限速、指数退避重试、熔断。

设计要点：
- 状态按 API host 维度在进程内共享。GitHubClient 每次调用都会新建实例，
  所以限流和熔断状态不能挂在实例上，否则形同虚设。
- 读请求（GET）对超时、网络错误、5xx、429 和限流 403 做指数退避重试。
- 写请求（POST/PATCH）只在"确定没被服务端处理"时重试：连接失败、429、限流 403。
  请求已发出后超时或 5xx 不重试，因为 GitHub 可能已经执行了，盲目重试会产生重复评论。
  写操作的安全重放由 ActionExecution 幂等记录负责。
- 其他 4xx（401、404、422 等）直接失败，重试没有意义。
"""

from __future__ import annotations

import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx

SleepFn = Callable[[float], Awaitable[None]]


class GitHubCircuitOpenError(RuntimeError):
    """熔断打开时快速失败。文案包含 temporarily unavailable，Agent 工具层会归类为瞬时错误。"""

    def __init__(self, host: str, retry_after: float) -> None:
        self.host = host
        self.retry_after = max(0.0, retry_after)
        super().__init__(
            f"GitHub API temporarily unavailable for {host}: circuit open, retry after {self.retry_after:.0f}s"
        )


class GitHubRateLimitExceeded(RuntimeError):
    """配额耗尽且重置时间超过可等待上限时快速失败，文案包含 rate limit 便于上层归类。"""

    def __init__(self, host: str, reset_in: float) -> None:
        self.host = host
        self.reset_in = max(0.0, reset_in)
        super().__init__(f"GitHub API rate limit exhausted for {host}, resets in {self.reset_in:.0f}s")


@dataclass
class RateLimitState:
    remaining: int | None = None
    limit: int | None = None
    reset_at: float | None = None  # epoch seconds

    def update_from_headers(self, headers: httpx.Headers) -> None:
        remaining = headers.get("x-ratelimit-remaining")
        limit = headers.get("x-ratelimit-limit")
        reset = headers.get("x-ratelimit-reset")
        if remaining is not None and remaining.isdigit():
            self.remaining = int(remaining)
        if limit is not None and limit.isdigit():
            self.limit = int(limit)
        if reset is not None and reset.isdigit():
            self.reset_at = float(reset)

    def seconds_until_reset(self, now: float) -> float:
        if self.reset_at is None:
            return 0.0
        return max(0.0, self.reset_at - now)


@dataclass
class CircuitBreaker:
    """连续失败计数熔断：closed -> open -> half_open -> closed/open。"""

    failure_threshold: int
    reset_seconds: float
    state: str = "closed"
    consecutive_failures: int = 0
    opened_at: float = 0.0
    half_open_in_flight: bool = False

    def before_request(self, host: str, now: float) -> None:
        if self.state == "open":
            elapsed = now - self.opened_at
            if elapsed < self.reset_seconds:
                raise GitHubCircuitOpenError(host, self.reset_seconds - elapsed)
            self.state = "half_open"
            self.half_open_in_flight = False
        if self.state == "half_open":
            if self.half_open_in_flight:
                # 半开状态只放行一个探测请求
                raise GitHubCircuitOpenError(host, 1.0)
            self.half_open_in_flight = True

    def record_success(self) -> None:
        self.state = "closed"
        self.consecutive_failures = 0
        self.half_open_in_flight = False

    def record_failure(self, now: float) -> None:
        self.half_open_in_flight = False
        if self.state == "half_open":
            self._open(now)
            return
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.failure_threshold:
            self._open(now)

    def _open(self, now: float) -> None:
        self.state = "open"
        self.opened_at = now


@dataclass
class HostResilienceState:
    rate_limit: RateLimitState
    breaker: CircuitBreaker


_HOST_STATES: dict[str, HostResilienceState] = {}


def host_state(host: str, *, failure_threshold: int, reset_seconds: float) -> HostResilienceState:
    state = _HOST_STATES.get(host)
    if state is None:
        state = HostResilienceState(
            rate_limit=RateLimitState(),
            breaker=CircuitBreaker(failure_threshold=failure_threshold, reset_seconds=reset_seconds),
        )
        _HOST_STATES[host] = state
    return state


def reset_resilience_state() -> None:
    """测试和故障演练用：清空进程内限流与熔断状态。"""
    _HOST_STATES.clear()


def is_rate_limit_response(response: httpx.Response) -> bool:
    if response.status_code == 429:
        return True
    if response.status_code != 403:
        return False
    if response.headers.get("retry-after"):
        return True
    if response.headers.get("x-ratelimit-remaining") == "0":
        return True
    text = response.text.lower() if response.content else ""
    return "rate limit" in text


def retry_after_seconds(response: httpx.Response, now: float) -> float | None:
    """GitHub 建议：有 Retry-After 用它；配额为 0 时等到 x-ratelimit-reset。"""
    retry_after = response.headers.get("retry-after")
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            return None
    if response.headers.get("x-ratelimit-remaining") == "0":
        reset = response.headers.get("x-ratelimit-reset")
        if reset and reset.isdigit():
            return max(0.0, float(reset) - now)
    return None


def backoff_delay(attempt: int, *, base: float, cap: float, rng: Callable[[], float] = random.random) -> float:
    """Full jitter 指数退避：sleep = random(0, min(cap, base * 2**attempt))。"""
    ceiling = min(cap, base * (2**attempt))
    return ceiling * rng()


def wall_clock() -> float:
    return time.time()


def monotonic_clock() -> float:
    return time.monotonic()
