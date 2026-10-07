import logging
from typing import Any
from urllib.parse import urlparse

import httpx

from app.core.config import settings
from app.services.github.resilience import (
    GitHubRateLimitExceeded,
    HostResilienceState,
    SleepFn,
    backoff_delay,
    host_state,
    is_rate_limit_response,
    monotonic_clock,
    retry_after_seconds,
    wall_clock,
)

logger = logging.getLogger(__name__)

READ_METHODS = {"GET", "HEAD"}


class GitHubClient:
    def __init__(
        self,
        token: str | None = None,
        base_url: str | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: SleepFn | None = None,
    ) -> None:
        self.token = token or settings.github_token
        self.base_url = (base_url or settings.github_api_base_url).rstrip("/")
        self.host = urlparse(self.base_url).netloc or self.base_url
        self._transport = transport
        self._sleep = sleep
        self.last_request_stats: dict[str, Any] = {}

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _state(self) -> HostResilienceState:
        return host_state(
            self.host,
            failure_threshold=settings.github_breaker_failure_threshold,
            reset_seconds=settings.github_breaker_reset_seconds,
        )

    async def _sleep_for(self, seconds: float) -> None:
        if seconds <= 0:
            return
        if self._sleep is not None:
            await self._sleep(seconds)
            return
        import asyncio

        await asyncio.sleep(seconds)

    async def _throttle_if_needed(self, state: HostResilienceState) -> float:
        """主动限速：剩余配额低于阈值时，等到重置；等待时间过长就快速失败。"""
        rate = state.rate_limit
        if rate.remaining is None or rate.remaining > settings.github_rate_limit_min_remaining:
            return 0.0
        wait = rate.seconds_until_reset(wall_clock())
        if wait <= 0:
            return 0.0
        if wait > settings.github_rate_limit_max_wait_seconds:
            raise GitHubRateLimitExceeded(self.host, wait)
        logger.warning("GitHub quota low (remaining=%s), throttling %.1fs until reset", rate.remaining, wait)
        await self._sleep_for(wait)
        rate.remaining = None  # 重置后配额未知，等下一次响应头刷新
        return wait

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        method = method.upper()
        is_read = method in READ_METHODS
        state = self._state()
        max_retries = max(0, settings.github_max_retries)
        if timeout is None:
            timeout = settings.github_read_timeout_seconds if is_read else settings.github_write_timeout_seconds
        stats: dict[str, Any] = {"method": method, "path": path, "attempts": 0, "retries": [], "throttled_seconds": 0.0}
        self.last_request_stats = stats

        attempt = 0
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, transport=self._transport) as client:
            while True:
                attempt += 1
                stats["attempts"] = attempt
                # 先限速再过熔断：限速等待或快速失败都不应占用半开探测名额
                stats["throttled_seconds"] += await self._throttle_if_needed(state)
                state.breaker.before_request(self.host, monotonic_clock())

                try:
                    response = await client.request(
                        method,
                        f"{self.base_url}{path}",
                        headers=self._headers(),
                        params=params,
                        json=json_body,
                    )
                except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                    # 连接都没建立，读写都可以安全重试
                    state.breaker.record_failure(monotonic_clock())
                    if attempt > max_retries:
                        raise
                    await self._backoff(stats, attempt, reason=type(exc).__name__)
                    continue
                except httpx.TransportError as exc:
                    state.breaker.record_failure(monotonic_clock())
                    if not is_read or attempt > max_retries:
                        # 写请求可能已被服务端处理，不能盲目重放
                        raise
                    await self._backoff(stats, attempt, reason=type(exc).__name__)
                    continue

                state.rate_limit.update_from_headers(response.headers)

                if is_rate_limit_response(response):
                    # 限流不是服务不健康，不计入熔断；服务端明确拒绝了请求，读写都可重试
                    state.breaker.record_success()
                    if attempt > max_retries:
                        response.raise_for_status()
                    wait = retry_after_seconds(response, wall_clock())
                    if wait is not None and wait > settings.github_rate_limit_max_wait_seconds:
                        raise GitHubRateLimitExceeded(self.host, wait)
                    await self._backoff(stats, attempt, reason=f"http_{response.status_code}_rate_limited", fixed=wait)
                    continue

                if response.status_code >= 500:
                    state.breaker.record_failure(monotonic_clock())
                    if not is_read or attempt > max_retries:
                        response.raise_for_status()
                    await self._backoff(stats, attempt, reason=f"http_{response.status_code}")
                    continue

                # 2xx/3xx 和普通 4xx 都说明服务健康；4xx 直接抛出不重试
                state.breaker.record_success()
                response.raise_for_status()
                return response

    async def _backoff(self, stats: dict[str, Any], attempt: int, *, reason: str, fixed: float | None = None) -> None:
        delay = fixed if fixed is not None else backoff_delay(
            attempt - 1,
            base=settings.github_backoff_base_seconds,
            cap=settings.github_backoff_max_seconds,
        )
        stats["retries"].append({"attempt": attempt, "reason": reason, "delay_seconds": round(delay, 3)})
        logger.info("GitHub %s %s retry after %.2fs (%s)", stats["method"], stats["path"], delay, reason)
        await self._sleep_for(delay)

    async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        response = await self._request("GET", path, params=params)
        return response.json()

    async def get_bytes(self, path: str, params: dict[str, Any] | None = None) -> bytes:
        response = await self._request("GET", path, params=params, timeout=60)
        return response.content

    async def post(self, path: str, payload: dict[str, Any]) -> Any:
        response = await self._request("POST", path, json_body=payload)
        return response.json()

    async def patch(self, path: str, payload: dict[str, Any]) -> Any:
        response = await self._request("PATCH", path, json_body=payload)
        return response.json()

    def rate_limit_snapshot(self) -> dict[str, Any]:
        state = self._state()
        return {
            "host": self.host,
            "remaining": state.rate_limit.remaining,
            "limit": state.rate_limit.limit,
            "reset_at": state.rate_limit.reset_at,
            "breaker_state": state.breaker.state,
            "consecutive_failures": state.breaker.consecutive_failures,
        }

    async def get_repo(self, owner: str, repo: str) -> dict[str, Any]:
        return await self.get(f"/repos/{owner}/{repo}")

    async def create_issue_comment(self, owner: str, repo: str, issue_number: int, body: str) -> dict[str, Any]:
        return await self.post(f"/repos/{owner}/{repo}/issues/{issue_number}/comments", {"body": body})

    async def list_issue_comments(self, owner: str, repo: str, issue_number: int, per_page: int = 100) -> list[dict[str, Any]]:
        return await self.get(
            f"/repos/{owner}/{repo}/issues/{issue_number}/comments",
            {"per_page": per_page, "sort": "created", "direction": "desc"},
        )

    async def create_issue(self, owner: str, repo: str, title: str, body: str) -> dict[str, Any]:
        return await self.post(f"/repos/{owner}/{repo}/issues", {"title": title, "body": body})

    async def list_recent_issues(self, owner: str, repo: str, per_page: int = 50) -> list[dict[str, Any]]:
        return await self.get(
            f"/repos/{owner}/{repo}/issues",
            {"state": "all", "per_page": per_page, "sort": "created", "direction": "desc"},
        )

    async def close_issue(self, owner: str, repo: str, issue_number: int) -> dict[str, Any]:
        return await self.patch(f"/repos/{owner}/{repo}/issues/{issue_number}", {"state": "closed"})

    async def add_issue_labels(self, owner: str, repo: str, issue_number: int, labels: list[str]) -> dict[str, Any]:
        return await self.post(f"/repos/{owner}/{repo}/issues/{issue_number}/labels", {"labels": labels})
