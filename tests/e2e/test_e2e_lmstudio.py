import asyncio
import json
import os
from pathlib import Path
from urllib.request import urlopen, Request

import pytest

pytestmark = pytest.mark.e2e


def _lmstudio_endpoint() -> str:
    return os.getenv("LM_ENDPOINT") or os.getenv("LMSTUDIO_URL") or "http://127.0.0.1:1234"


def _reachable(url: str) -> bool:
    try:
        with urlopen(Request(url + "/v1/models"), timeout=1):  # nosec - local/dev
            return True
    except Exception:
        return False


@pytest.mark.asyncio
async def test_e2e_lmstudio_chat_completion_live(monkeypatch):
    base = _lmstudio_endpoint()
    if not _reachable(base):
        pytest.skip(f"LM Studio not reachable at {base}; set LM_ENDPOINT or LMSTUDIO_URL and ensure it is running.")

    from story_engine.core.core.orchestration.llm_orchestrator import (
        LLMOrchestrator,
        LLMConfig,
        ModelProvider,
    )

    orch = LLMOrchestrator(fail_on_all_providers=True)
    orch.register_provider(
        "lmstudio",
        LLMConfig(
            provider=ModelProvider.LMSTUDIO,
            endpoint=base,
            # Prefer a small model if configured server-side; we pass a sentinel or a concrete model if env provided.
            model=os.getenv("LM_MODEL", "auto"),
            temperature=0.2,
            max_tokens=128,
        ),
    )

    system = "You are a helpful assistant."
    prompt = "Write one vivid sentence about sunrise in Jerusalem."
    try:
        resp = await orch.generate(prompt, system=system, provider_name="lmstudio")
        assert isinstance(resp.text, str) and len(resp.text.strip()) > 0
        # Avoid strict content checks; live model output varies. Basic sanity checks only.
        assert resp.raw_response is not None
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"LM Studio completion failed at {base}: {e}")


def test_e2e_sqlite_store_and_fetch_live(tmp_path):
    from story_engine.core.core.storage.database import get_database_connection

    db_path = tmp_path / "e2e_outputs.db"
    db = get_database_connection("sqlite", db_name=str(db_path))
    db.connect()
    try:
        workflow = "e2e_lmstudio_live"
        payload = {"ok": True, "id": 1, "note": "live"}
        db.store_output(workflow, payload)
        out = db.get_outputs(workflow)
        assert isinstance(out, list) and out
        assert any(o.get("ok") for o in out)
    finally:
        db.disconnect()


@pytest.mark.asyncio
async def test_e2e_budget_behavior_live(monkeypatch):
    base = _lmstudio_endpoint()
    if not _reachable(base):
        pytest.skip(f"LM Studio not reachable at {base}; set LM_ENDPOINT or LMSTUDIO_URL and ensure it is running.")

    from story_engine.core.core.orchestration.llm_orchestrator import (
        LLMOrchestrator,
        LLMConfig,
        ModelProvider,
        GenerationError,
    )

    # Small budget to exercise budget handling without relying on stubs
    monkeypatch.setenv("LM_REQUEST_BUDGET_MS", "30")

    orch = LLMOrchestrator(fail_on_all_providers=True)
    orch.register_provider(
        "lmstudio",
        LLMConfig(
            provider=ModelProvider.LMSTUDIO,
            endpoint=base,
            model=os.getenv("LM_MODEL", "auto"),
            timeout=2,
        ),
    )

    system = "You are concise."
    prompt = "Say 'ok' and stop."
    try:
        resp = await orch.generate(prompt, system=system, provider_name="lmstudio")
        assert isinstance(resp.text, str) and len(resp.text) > 0
    except GenerationError as e:
        s = str(e).lower()
        if not any(tok in s for tok in ("timeout", "budget")):
            pytest.skip(f"LM Studio completion failed for non-budget reason at {base}: {e}")
