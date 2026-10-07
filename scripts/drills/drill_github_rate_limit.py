"""故障演练：GitHub 返回 429 / 5xx / 连续故障时的退避、限速和熔断。

不需要真实 token，用 httpx.MockTransport 注入故障；输出的重试日志可以直接截图放 README。
在 backend 目录下运行：python ../scripts/drills/drill_github_rate_limit.py
"""

import asyncio
import logging
import time

import httpx

from app.core.config import settings
from app.services.github.client import GitHubClient
from app.services.github.resilience import GitHubCircuitOpenError, reset_resilience_state

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def scripted(responses):
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        item = queue.pop(0)
        print(f"    <- GitHub {request.method} {request.url.path} => {item.status_code}")
        return item

    return httpx.MockTransport(handler)


async def scenario(title: str, client: GitHubClient, call):
    print(f"\n=== {title} ===")
    started = time.perf_counter()
    try:
        result = await call(client)
        print(f"    结果：成功 {result}")
    except Exception as exc:  # noqa: BLE001
        print(f"    结果：{type(exc).__name__}: {exc}")
    print(f"    尝试次数 {client.last_request_stats.get('attempts')}，重试 {client.last_request_stats.get('retries')}，耗时 {time.perf_counter() - started:.2f}s")


async def main() -> None:
    settings.github_backoff_base_seconds = 0.2
    settings.github_breaker_failure_threshold = 3
    settings.github_breaker_reset_seconds = 5
    reset_resilience_state()
    base = "https://api.github.drill"

    await scenario(
        "1. 429 + Retry-After：按服务端要求等待后重试成功",
        GitHubClient("t", base, transport=scripted([httpx.Response(429, headers={"retry-after": "1"}), httpx.Response(200, json={"ok": 1})])),
        lambda c: c.get("/repos/acme/api"),
    )
    await scenario(
        "2. 读请求遇到 502/503：指数退避后成功",
        GitHubClient("t", base, transport=scripted([httpx.Response(502), httpx.Response(503), httpx.Response(200, json={"ok": 2})])),
        lambda c: c.get("/repos/acme/api/issues"),
    )
    await scenario(
        "3. 写请求遇到 502：不重放，交给幂等对账",
        GitHubClient("t", base, transport=scripted([httpx.Response(502)])),
        lambda c: c.create_issue_comment("acme", "api", 7, "hello"),
    )
    reset_resilience_state()
    settings.github_max_retries = 0
    for index in range(3):
        await scenario(
            f"4.{index + 1} 连续 500，累计失败 {index + 1} 次",
            GitHubClient("t", base, transport=scripted([httpx.Response(500)])),
            lambda c: c.get("/repos/acme/api"),
        )
    await scenario(
        "5. 熔断打开：不再请求 GitHub，直接快速失败",
        GitHubClient("t", base, transport=scripted([])),
        lambda c: c.get("/repos/acme/api"),
    )
    print(f"\n等待 {settings.github_breaker_reset_seconds}s 进入半开...")
    await asyncio.sleep(settings.github_breaker_reset_seconds)
    await scenario(
        "6. 半开探测成功：熔断关闭",
        GitHubClient("t", base, transport=scripted([httpx.Response(200, json={"ok": 6})])),
        lambda c: c.get("/repos/acme/api"),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except GitHubCircuitOpenError as exc:
        print(exc)
