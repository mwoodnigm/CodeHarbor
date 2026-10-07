import time

import httpx
import pytest

from app.core.config import settings
from app.services.github.client import GitHubClient
from app.services.github.resilience import (
    CircuitBreaker,
    GitHubCircuitOpenError,
    GitHubRateLimitExceeded,
    backoff_delay,
    reset_resilience_state,
)


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    reset_resilience_state()
    monkeypatch.setattr(settings, "github_max_retries", 3)
    monkeypatch.setattr(settings, "github_breaker_failure_threshold", 3)
    monkeypatch.setattr(settings, "github_breaker_reset_seconds", 30.0)
    monkeypatch.setattr(settings, "github_rate_limit_min_remaining", 5)
    monkeypatch.setattr(settings, "github_rate_limit_max_wait_seconds", 60.0)
    yield
    reset_resilience_state()


class Recorder:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeSleep:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def _client(recorder: Recorder, sleep: FakeSleep, base_url: str = "https://api.github.test") -> GitHubClient:
    return GitHubClient("token", base_url, transport=httpx.MockTransport(recorder), sleep=sleep)


@pytest.mark.asyncio
async def test_read_retries_5xx_with_backoff_then_succeeds() -> None:
    recorder = Recorder([httpx.Response(502), httpx.Response(503), httpx.Response(200, json={"ok": True})])
    sleep = FakeSleep()
    client = _client(recorder, sleep)

    assert await client.get("/repos/a/b") == {"ok": True}
    assert len(recorder.requests) == 3
    assert len(sleep.calls) == 2
    assert [item["reason"] for item in client.last_request_stats["retries"]] == ["http_502", "http_503"]


@pytest.mark.asyncio
async def test_read_retries_timeout() -> None:
    recorder = Recorder([httpx.ReadTimeout("slow"), httpx.Response(200, json=[1])])
    sleep = FakeSleep()

    assert await _client(recorder, sleep).get("/repos/a/b/issues") == [1]
    assert len(recorder.requests) == 2


@pytest.mark.asyncio
async def test_plain_4xx_fails_fast_without_retry() -> None:
    recorder = Recorder([httpx.Response(404, json={"message": "Not Found"})])
    sleep = FakeSleep()

    with pytest.raises(httpx.HTTPStatusError):
        await _client(recorder, sleep).get("/repos/a/missing")
    assert len(recorder.requests) == 1
    assert sleep.calls == []


@pytest.mark.asyncio
async def test_429_uses_retry_after_header() -> None:
    recorder = Recorder(
        [
            httpx.Response(429, headers={"retry-after": "7"}),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    sleep = FakeSleep()

    assert await _client(recorder, sleep).get("/repos/a/b") == {"ok": True}
    assert sleep.calls == [7.0]


@pytest.mark.asyncio
async def test_secondary_rate_limit_403_is_retried_but_permission_403_is_not() -> None:
    sleep = FakeSleep()
    limited = Recorder(
        [
            httpx.Response(403, json={"message": "You have exceeded a secondary rate limit"}),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    assert await _client(limited, sleep).get("/repos/a/b") == {"ok": True}

    forbidden = Recorder([httpx.Response(403, json={"message": "Resource not accessible by integration"})])
    with pytest.raises(httpx.HTTPStatusError):
        await _client(forbidden, sleep).get("/repos/a/b")
    assert len(forbidden.requests) == 1


@pytest.mark.asyncio
async def test_rate_limit_reset_too_far_fails_fast() -> None:
    reset = int(time.time()) + 3600
    recorder = Recorder([httpx.Response(403, headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(reset)})])

    with pytest.raises(GitHubRateLimitExceeded):
        await _client(recorder, FakeSleep()).get("/repos/a/b")


@pytest.mark.asyncio
async def test_low_remaining_quota_throttles_next_request_until_reset() -> None:
    reset = int(time.time()) + 20
    recorder = Recorder(
        [
            httpx.Response(200, json={"n": 1}, headers={"x-ratelimit-remaining": "2", "x-ratelimit-reset": str(reset)}),
            httpx.Response(200, json={"n": 2}, headers={"x-ratelimit-remaining": "5000"}),
        ]
    )
    sleep = FakeSleep()
    client = _client(recorder, sleep)

    await client.get("/repos/a/b")
    await client.get("/repos/a/b")

    assert len(sleep.calls) == 1
    assert 15 <= sleep.calls[0] <= 21
    assert client.last_request_stats["throttled_seconds"] > 0


@pytest.mark.asyncio
async def test_write_is_not_replayed_after_ambiguous_timeout() -> None:
    recorder = Recorder([httpx.ReadTimeout("sent but no response"), httpx.Response(201, json={"id": 1})])

    with pytest.raises(httpx.ReadTimeout):
        await _client(recorder, FakeSleep()).create_issue_comment("a", "b", 1, "hello")
    assert len(recorder.requests) == 1


@pytest.mark.asyncio
async def test_write_is_not_replayed_after_5xx() -> None:
    recorder = Recorder([httpx.Response(502), httpx.Response(201, json={"id": 1})])

    with pytest.raises(httpx.HTTPStatusError):
        await _client(recorder, FakeSleep()).create_issue_comment("a", "b", 1, "hello")
    assert len(recorder.requests) == 1


@pytest.mark.asyncio
async def test_write_is_retried_when_request_never_reached_server() -> None:
    recorder = Recorder([httpx.ConnectError("refused"), httpx.Response(201, json={"id": 1})])

    assert await _client(recorder, FakeSleep()).create_issue_comment("a", "b", 1, "hello") == {"id": 1}
    assert len(recorder.requests) == 2


@pytest.mark.asyncio
async def test_write_is_retried_on_429() -> None:
    recorder = Recorder([httpx.Response(429, headers={"retry-after": "1"}), httpx.Response(201, json={"id": 1})])

    assert await _client(recorder, FakeSleep()).create_issue_comment("a", "b", 1, "hello") == {"id": 1}


@pytest.mark.asyncio
async def test_breaker_opens_after_consecutive_failures_and_fails_fast(monkeypatch) -> None:
    monkeypatch.setattr(settings, "github_max_retries", 0)
    recorder = Recorder([httpx.Response(500), httpx.Response(500), httpx.Response(500)])
    sleep = FakeSleep()

    for _ in range(3):
        with pytest.raises(httpx.HTTPStatusError):
            await _client(recorder, sleep).get("/repos/a/b")

    with pytest.raises(GitHubCircuitOpenError):
        await _client(recorder, sleep).get("/repos/a/b")
    assert len(recorder.requests) == 3


@pytest.mark.asyncio
async def test_breaker_state_is_shared_per_host_but_isolated_between_hosts(monkeypatch) -> None:
    monkeypatch.setattr(settings, "github_max_retries", 0)
    failing = Recorder([httpx.Response(500)] * 3)
    for _ in range(3):
        with pytest.raises(httpx.HTTPStatusError):
            await _client(failing, FakeSleep(), "https://ghe.internal/api/v3").get("/x")

    healthy = Recorder([httpx.Response(200, json={"ok": True})])
    assert await _client(healthy, FakeSleep(), "https://api.github.test").get("/x") == {"ok": True}


def test_breaker_half_open_allows_single_probe_then_closes() -> None:
    breaker = CircuitBreaker(failure_threshold=2, reset_seconds=10)
    breaker.record_failure(now=0)
    breaker.record_failure(now=0)
    assert breaker.state == "open"

    with pytest.raises(GitHubCircuitOpenError):
        breaker.before_request("h", now=5)

    breaker.before_request("h", now=11)
    assert breaker.state == "half_open"
    with pytest.raises(GitHubCircuitOpenError):
        breaker.before_request("h", now=11)

    breaker.record_success()
    assert breaker.state == "closed"


def test_breaker_half_open_failure_reopens() -> None:
    breaker = CircuitBreaker(failure_threshold=1, reset_seconds=10)
    breaker.record_failure(now=0)
    breaker.before_request("h", now=10)
    breaker.record_failure(now=10)
    assert breaker.state == "open"
    with pytest.raises(GitHubCircuitOpenError):
        breaker.before_request("h", now=15)


def test_backoff_is_capped_full_jitter() -> None:
    assert backoff_delay(0, base=0.5, cap=8, rng=lambda: 1.0) == 0.5
    assert backoff_delay(3, base=0.5, cap=8, rng=lambda: 1.0) == 4.0
    assert backoff_delay(10, base=0.5, cap=8, rng=lambda: 1.0) == 8.0
    assert backoff_delay(10, base=0.5, cap=8, rng=lambda: 0.0) == 0.0
