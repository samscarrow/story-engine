import asyncio
import os
import pytest

from story_engine.core.core.orchestration.llm_orchestrator import (
    KoboldCppProvider,
    LLMConfig,
    ModelProvider,
    GenerationError,
)


@pytest.mark.asyncio
async def test_kobold_circuit_open_fast_fails(monkeypatch):
    cfg = LLMConfig(provider=ModelProvider.KOBOLDCPP, endpoint="http://localhost:5001")
    prov = KoboldCppProvider(cfg)
    # Simulate open circuit
    prov._cb_open_until = 1e9  # far in the future
    with pytest.raises(GenerationError) as ei:
        await prov.generate("hello")
    assert "circuit_open" in str(ei.value)


@pytest.mark.asyncio
async def test_kobold_circuit_opens_after_failures(monkeypatch):
    # Force no retries and a 503 response to trip failures quickly
    class FakeResp:
        def __init__(self):
            self.status = 503

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def text(self):
            return "unavailable"

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
    monkeypatch.setenv("KOBOLD_RETRY_ATTEMPTS", "1")
    monkeypatch.setattr(
        "story_engine.core.core.orchestration.llm_orchestrator.aiohttp.ClientSession",
        _fake_client_session,
    )
    cfg = LLMConfig(provider=ModelProvider.KOBOLDCPP, endpoint="http://fake")
    prov = KoboldCppProvider(cfg)

    # Two failing calls should open circuit
    for _ in range(2):
        with pytest.raises(GenerationError):
            await prov.generate("x")

    # Third call fast-fails due to circuit
    with pytest.raises(GenerationError) as ei:
        await prov.generate("x")
    assert "circuit_open" in str(ei.value)

