import asyncio
import pytest

from tests.e2e import _mock_fixtures
from story_engine.core.core.orchestration.llm_orchestrator import LLMConfig, LMStudioProvider, ModelProvider


@pytest.mark.asyncio
async def test_streaming_plain_payload(monkeypatch):
    monkeypatch.delenv("LM_MODEL", raising=False)
    async with _mock_fixtures.mock_server_ctx(mode="plain") as (host, port):
        provider = LMStudioProvider(
            LLMConfig(provider=ModelProvider.LMSTUDIO, endpoint=f"http://{host}:{port}", max_tokens=64)
        )
        resp = await provider.generate("Hello there", stream=True)
        assert resp.text.startswith("[mock:auto]")
        assert resp.raw_response is not None
        assert resp.raw_response.get("stream")
        assert resp.raw_response["normalized"]["reasoning"] == ""


@pytest.mark.asyncio
async def test_streaming_reasoning_payload(monkeypatch):
    monkeypatch.delenv("LM_MODEL", raising=False)
    async with _mock_fixtures.mock_server_ctx(mode="reasoning", reasoning_chunks=2) as (host, port):
        provider = LMStudioProvider(
            LLMConfig(provider=ModelProvider.LMSTUDIO, endpoint=f"http://{host}:{port}", max_tokens=64)
        )
        resp = await provider.generate("Solve the puzzle", stream=True)
        assert resp.text.startswith("[reasoned:auto]")
        assert "chunk-0" in resp.raw_response["normalized"]["reasoning"]
        assert resp.raw_response.get("stream")
