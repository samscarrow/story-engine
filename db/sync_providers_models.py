from __future__ import annotations

import os
import sys
import json
import argparse
from typing import Any, Dict


# Ensure required DB environment is present even outside direnv
def _ensure_oracle_env() -> None:
    """Load .env/.env.oracle and normalize to ORACLE_* variables only.

    Preference order for DSN/user/password:
    - ORACLE_* (canonical)
    - DB_* (legacy), mapped into ORACLE_* for backward compatibility
    """
    try:
        # lightweight loader: include both ORACLE_* and legacy DB_*
        sys.path.append("src")
        from story_engine.core.common.dotenv_loader import load_dotenv_keys  # type: ignore

        load_dotenv_keys(keys_prefixes=["ORACLE_", "DB_", "STORE_ALL"])  # .env
        load_dotenv_keys(path=".env.oracle", keys_prefixes=["ORACLE_", "DB_"])
    except Exception:
        # Best-effort; continue with whatever env is present
        pass

    # Map DB_* → ORACLE_* if missing
    # Canonicalize DSN
    if not os.environ.get("ORACLE_DSN"):
        if os.environ.get("DB_DSN"):
            os.environ["ORACLE_DSN"] = os.environ["DB_DSN"]
        elif os.environ.get("ORACLE_CONNECT_STRING"):
            os.environ["ORACLE_DSN"] = os.environ["ORACLE_CONNECT_STRING"]

    if not os.environ.get("ORACLE_USER") and os.environ.get("DB_USER"):
        os.environ["ORACLE_USER"] = os.environ["DB_USER"]
    if not os.environ.get("ORACLE_PASSWORD") and os.environ.get("DB_PASSWORD"):
        os.environ["ORACLE_PASSWORD"] = os.environ["DB_PASSWORD"]


def _connect_oracle():
    _ensure_oracle_env()
    try:
        import oracledb  # type: ignore
    except Exception:
        print(
            "oracledb is required. Install with: uv add oracledb (or pip)",
            file=sys.stderr,
        )
        raise

    dsn = os.environ.get("ORACLE_DSN")
    user = os.environ.get("ORACLE_USER")
    password = os.environ.get("ORACLE_PASSWORD")
    if not all([dsn, user, password]):
        raise RuntimeError("Set ORACLE_DSN, ORACLE_USER, ORACLE_PASSWORD in env")
    return oracledb.connect(user=user, password=password, dsn=dsn)


def _load_orchestrator(config_path: str):
    sys.path.append("src")
    from story_engine.core.orchestration.orchestrator_loader import (
        create_orchestrator_from_yaml,
    )

    return create_orchestrator_from_yaml(config_path)


def _upsert_provider(
    cur, name: str, ptype: str, endpoint: str, meta: Dict[str, Any]
) -> int:
    pid_var = cur.var(int)
    cur.execute(
        """
        MERGE INTO providers p
        USING (SELECT :name AS name FROM dual) s
        ON (p.name = s.name)
        WHEN MATCHED THEN
          UPDATE SET p.type = :ptype, p.endpoint = :endpoint, p.meta = :meta
        WHEN NOT MATCHED THEN
          INSERT (name, type, endpoint, meta) VALUES (:name, :ptype, :endpoint, :meta)
        RETURNING p.provider_id INTO :pid
        """,
        name=name,
        ptype=ptype,
        endpoint=endpoint,
        meta=json.dumps(meta) if meta else None,
        pid=pid_var,
    )
    pid_val = pid_var.getvalue()
    if isinstance(pid_val, list):
        pid_val = pid_val[0]
    return int(pid_val)


def _upsert_model(cur, provider_id: int, model: Dict[str, Any]) -> int:
    model_key = (
        model.get("id") or model.get("name") or model.get("model") or model.get("slug")
    )
    if not model_key:
        return -1

    mid_var = cur.var(int)
    cur.execute(
        """
        MERGE INTO models m
        USING (SELECT :provider_id AS provider_id, :model_key AS model_key FROM dual) s
        ON (m.provider_id = s.provider_id AND m.model_key = s.model_key)
        WHEN MATCHED THEN
          UPDATE SET m.display_name = :display_name,
                     m.family = :family,
                     m.size_b = :size_b,
                     m.capabilities = :capabilities,
                     m.modalities = :modalities,
                     m.defaults = :defaults,
                     m.active = 'Y'
        WHEN NOT MATCHED THEN
          INSERT (provider_id, model_key, display_name, family, size_b, capabilities, modalities, defaults, active)
          VALUES (:provider_id, :model_key, :display_name, :family, :size_b, :capabilities, :modalities, :defaults, 'Y')
        RETURNING m.model_id INTO :mid
        """,
        provider_id=provider_id,
        model_key=model_key,
        display_name=model.get("id") or model.get("name") or model.get("model"),
        family=None,
        size_b=None,
        capabilities=(
            json.dumps(model.get("capabilities")) if model.get("capabilities") else None
        ),
        modalities=(
            json.dumps(model.get("modalities")) if model.get("modalities") else None
        ),
        defaults=None,
        mid=mid_var,
    )
    mid_val = mid_var.getvalue()
    if isinstance(mid_val, list):
        mid_val = mid_val[0]
    return int(mid_val)


def main():
    ap = argparse.ArgumentParser(
        description="Sync providers/models from orchestrator into DB"
    )
    ap.add_argument(
        "--config", default="config.yaml", help="Orchestrator YAML config path"
    )
    args = ap.parse_args()

    orch = _load_orchestrator(args.config)

    # Collect health across providers
    import asyncio

    async def _collect():
        health = await orch.health_check_all()
        return health

    health = asyncio.run(_collect())

    conn = _connect_oracle()
    try:
        cur = conn.cursor()
        for name, info in health.items():
            endpoint = info.get("endpoint") or ""
            meta = {k: v for k, v in info.items() if k not in ("models",)}
            pid = _upsert_provider(
                cur, name=name, ptype="custom", endpoint=str(endpoint), meta=meta
            )
            ms = info.get("models") or []
            for m in ms:
                if isinstance(m, dict):
                    _upsert_model(cur, provider_id=pid, model=m)
        conn.commit()
        print("Sync complete.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
