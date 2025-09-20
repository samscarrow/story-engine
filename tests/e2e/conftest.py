
import os
from typing import AsyncIterator

import pytest

from tests.e2e import _mock_fixtures


@pytest.fixture()
async def lmstudio_endpoint() -> AsyncIterator[str]:
    """Provide an endpoint for e2e tests.

    - If RUN_LIVE_E2E=1, use LM_ENDPOINT or default http://localhost:1234
      and assert reachability (else fail the test early).
    - Otherwise, start a local mock implementing /v1/chat/completions.
    """
    if os.getenv("RUN_LIVE_E2E", "").strip().lower() in {"1", "true", "yes", "on"}:
        base = os.getenv("LM_ENDPOINT", "http://localhost:1234")
        # quick reachability check
        try:
            import aiohttp
            async with aiohttp.ClientSession() as s:
                # Prefer GET /v1/models as a simple health signal
                try:
                    async with s.get(f"{base}/v1/models", timeout=2) as r:
                        if r.status == 200:
                            yield base
                            return
                except Exception:
                    pass

                # Fallback: attempt POST /v1/chat/completions; accept most non-5xx as reachable
                async with s.post(
                    f"{base}/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "ping"}], "model": "auto", "max_tokens": 1},
                    timeout=3,
                ) as resp:
                    if resp.status < 500:
                        # 2xx/3xx/4xx are considered reachable; test will verify behavior
                        yield base
                        return
            raise AssertionError(f"Live LM endpoint not reachable at {base}: non-2xx/4xx responses")
        except Exception as e:
            raise AssertionError(f"Live LM endpoint not reachable at {base}: {e}")

    # Fallback: start mock server
    async with _mock_fixtures.mock_server_ctx() as (host, port):
        yield f"http://{host}:{port}"
