import asyncio
import os
import pytest

from story_engine.core.core.orchestration.llm_orchestrator import (
    LLMConfig,
    ModelProvider,
    GenerationError,
    LMStudioProvider,
)


@pytest.mark.asyncio
async def test_lmstudio_circuit_open_fast_fails(monkeypatch):
    cfg = LLMConfig(provider=ModelProvider.LMSTUDIO, endpoint="http://localhost:1234")
    prov = LMStudioProvider(cfg)
    prov._cb_open_until = 1e9  # simulate open circuit
    with pytest.raises(GenerationError) as ei:
        await prov.generate("hello")
    assert "circuit_open" in str(ei.value)


@pytest.mark.asyncio
async def test_lmstudio_circuit_opens_after_failures(monkeypatch):
    class FakeResp:
        def __init__(self, status: int = 503, text: str = "unavailable"):
            self.status = status
            self._text = text
            self.headers = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def text(self):
            return self._text

        async def json(self):
            return {"error": self._text}

    class FakeSess:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def post(self, *a, **k):
            return FakeResp()

    def _fake_client_session(*args, **kwargs):
        return FakeSess()

    monkeypatch.setenv("LLM_CB_THRESHOLD", "2")
    monkeypatch.setenv("LM_RETRY_ATTEMPTS", "1")
    monkeypatch.setattr(
        "story_engine.core.core.orchestration.llm_orchestrator.aiohttp.ClientSession",
        _fake_client_session,
    )

    cfg = LLMConfig(provider=ModelProvider.LMSTUDIO, endpoint="http://fake")
    prov = LMStudioProvider(cfg)

    # Two failing calls should open circuit
    for _ in range(2):
        with pytest.raises(GenerationError):
            await prov.generate("x")

    # Third call fast-fails due to circuit
    with pytest.raises(GenerationError) as ei:
        await prov.generate("x")
    assert "circuit_open" in str(ei.value)

