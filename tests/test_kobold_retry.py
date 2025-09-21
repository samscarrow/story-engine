import asyncio
import os
import pytest

from story_engine.core.core.orchestration.llm_orchestrator import (
    KoboldCppProvider,
    LLMConfig,
    ModelProvider,
)


class FakeResponse:
    def __init__(self, status: int, body: dict | None = None, text: str | None = None):
        self.status = status
        self._body = body or {}
        self._text = text or ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def text(self):
        return self._text

    async def json(self):
        return self._body


class FakeSession:
    def __init__(self, seq):
        self._seq = list(seq)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def post(self, url, json=None, timeout=None):
        try:
            return self._seq.pop(0)
        except IndexError:
            return FakeResponse(500, text="out of responses")


@pytest.mark.asyncio
async def test_kobold_retry_then_success(monkeypatch):
    # First 500 then 200
    seq = [
        FakeResponse(500, text="oops"),
        FakeResponse(200, body={"results": [{"text": "ok"}]}),
    ]

    def _fake_client_session(*args, **kwargs):
        return FakeSession(seq)

    monkeypatch.setenv("KOBOLD_RETRY_ATTEMPTS", "2")
    monkeypatch.setenv("KOBOLD_RETRY_BASE_DELAY", "0")
    monkeypatch.setattr(
        "story_engine.core.core.orchestration.llm_orchestrator.aiohttp.ClientSession",
        _fake_client_session,
    )
    cfg = LLMConfig(provider=ModelProvider.KOBOLDCPP, endpoint="http://fake")
    prov = KoboldCppProvider(cfg)
    out = await prov.generate("hello")
    assert out.text == "ok"


@pytest.mark.asyncio
async def test_kobold_non200_no_retry(monkeypatch):
    seq = [FakeResponse(503, text="unavailable")]

    def _fake_client_session(*args, **kwargs):
        return FakeSession(seq)

    monkeypatch.setenv("KOBOLD_RETRY_ATTEMPTS", "1")
    monkeypatch.setattr(
        "story_engine.core.core.orchestration.llm_orchestrator.aiohttp.ClientSession",
        _fake_client_session,
    )
    cfg = LLMConfig(provider=ModelProvider.KOBOLDCPP, endpoint="http://fake")
    prov = KoboldCppProvider(cfg)
    with pytest.raises(Exception):
        await prov.generate("hello")

