import json
import os
from urllib.request import urlopen, Request

import pytest


pytestmark = [pytest.mark.e2e, pytest.mark.acceptance]


def _lmstudio_endpoint() -> str:
    ep = os.getenv("LM_ENDPOINT") or os.getenv("LMSTUDIO_URL")
    if not ep:
        raise AssertionError("LM_ENDPOINT not set for acceptance run")
    return ep


def _reachable(url: str) -> None:
    # Fail-fast if LM Studio is not reachable
    try:
        with urlopen(Request(url + "/v1/models"), timeout=2) as r:  # nosec - acceptance requires real API
            assert r.status == 200, f"/v1/models returned {r.status} at {url}"
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f"LM Studio not reachable at {url}: {e}")


@pytest.mark.asyncio
async def test_acceptance_user_journey_live():
    base = _lmstudio_endpoint()
    _reachable(base)

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
            model=os.getenv("LM_MODEL", "auto"),
            temperature=0.2,
            max_tokens=128,
        ),
    )

    system = "You are a helpful, concise assistant."
    prompt = "Write one vivid sentence about sunrise in Jerusalem."
    resp = await orch.generate(prompt, system=system, provider_name="lmstudio")
    text = (resp.text or "").strip()
    assert text, "empty response from LM Studio"
    # Authenticity: ensure no placeholders/mocks leaked through
    bad = {"mock", "placeholder", "lorem ipsum"}
    tl = text.lower()
    assert not any(b in tl for b in bad), f"inauthentic content found: {text[:120]}"


def test_acceptance_db_persist_live(tmp_path):
    from story_engine.core.core.common.settings import get_db_settings
    from story_engine.core.core.storage.database import get_database_connection

    s = get_db_settings()
    db_type = s.get("db_type", "sqlite")

    if db_type == "oracle":
        # Require Oracle variables for acceptance runs
        for k in ("user", "password", "dsn", "wallet_location"):
            assert s.get(k), f"missing Oracle setting: {k}"
        db = get_database_connection(
            db_type="oracle",
            user=s["user"],
            password=s["password"],
            dsn=s["dsn"],
            wallet_location=s["wallet_location"],
            wallet_password=s.get("wallet_password"),
            use_pool=True,
            retry_attempts=2,
        )
    elif db_type == "postgresql":
        db = get_database_connection(
            db_type="postgresql",
            db_name=s.get("db_name", "story_db"),
            user=s.get("user", "story"),
            password=s.get("password"),
            host=s.get("host", "localhost"),
            port=int(s.get("port", 5432)),
            sslmode=s.get("sslmode"),
            sslrootcert=s.get("sslrootcert"),
            sslcert=s.get("sslcert"),
            sslkey=s.get("sslkey"),
        )
    else:
        # SQLite acceptance is allowed if Oracle is not configured
        db_path = tmp_path / "acceptance_outputs.db"
        db = get_database_connection("sqlite", db_name=str(db_path))

    db.connect()
    try:
        wf = "acceptance_user_journey"
        payload = {"ok": True, "stage": "persist", "note": "live"}
        db.store_output(wf, payload)
        rows = db.get_outputs(wf)
        assert isinstance(rows, list) and rows, "expected at least one persisted row"
        assert any(r.get("ok") for r in rows)
    finally:
        db.disconnect()

