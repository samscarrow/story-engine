from __future__ import annotations

import os
from typing import Any, Dict


def store_workflow_output(workflow_name: str, output: Dict[str, Any]) -> None:
    """Best-effort store of workflow output to configured DB (Oracle or Postgres).

    - Detects DB via `DB_TYPE` env (oracle|postgresql). Defaults to postgresql.
    - Loads DB_* env keys from `.env` and also tries `.env.oracle` if present.
    - If credentials are missing or connection fails, this function is a no-op.
    """
    try:
        from story_engine.core.storage import get_database_connection
        from .dotenv_loader import load_dotenv_keys
        from .settings import get_db_settings
    except Exception:
        return

    try:
        # Load DB_* env vars from .env (Postgres compose) and .env.oracle (Oracle wallet)
        load_dotenv_keys()
        try:
            # Attempt to also load Oracle env without failing if missing
            load_dotenv_keys(path=".env.oracle")
        except Exception:
            pass

        s = get_db_settings()
        if s.get("db_type") == "oracle":
            db = get_database_connection(
                db_type="oracle",
                user=s.get("user"),
                password=s.get("password"),
                dsn=s.get("dsn"),
                wallet_location=s.get("wallet_location"),
                wallet_password=s.get("wallet_password"),
                use_pool=bool(s.get("use_pool", True)),
                pool_min=int(s.get("pool_min", 1)),
                pool_max=int(s.get("pool_max", 4)),
                pool_increment=int(s.get("pool_increment", 1)),
                pool_timeout=int(s.get("pool_timeout", 60)),
                wait_timeout=s.get("wait_timeout"),
                retry_attempts=int(s.get("retry_attempts", 3)),
                retry_backoff_seconds=float(s.get("retry_backoff_seconds", 1.0)),
                ping_on_connect=bool(s.get("ping_on_connect", True)),
            )
        elif s.get("db_type") == "postgresql":
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
            db = get_database_connection("sqlite", db_name=s.get("db_name", "workflow_outputs.db"))
        db.connect()
        db.store_output(workflow_name, output)
    except Exception:
        # Best-effort; do not raise
        pass
    finally:
        try:
            if "db" in locals() and getattr(db, "conn", None):
                db.disconnect()
        except Exception:
            pass
