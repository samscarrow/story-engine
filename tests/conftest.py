"""Test session bootstrap.

Loads DB-related environment variables from `.env.oracle` (and `.env` fallback)
so tests don't rely on an interactive direnv session.
"""
from __future__ import annotations

import os
from pathlib import Path


def pytest_sessionstart(session) -> None:  # type: ignore[no-untyped-def]
    try:
        # Prefer project-local .env.oracle
        env_oracle = Path(".env.oracle")
        if env_oracle.exists():
            from story_engine.core.core.common.dotenv_loader import load_dotenv_keys

            load_dotenv_keys(path=str(env_oracle), keys_prefixes=("DB_", "ORACLE_"))
        # Fallback: generic .env if present
        env_generic = Path(".env")
        if env_generic.exists():
            from story_engine.core.core.common.dotenv_loader import load_dotenv_keys

            load_dotenv_keys(path=str(env_generic), keys_prefixes=("DB_", "ORACLE_"))

        # Normalize defaults for local XE if DSN missing
        if not os.getenv("DB_DSN") and os.getenv("DB_TYPE", "").lower() == "oracle":
            os.environ.setdefault("DB_DSN", "localhost/XEPDB1")
    except Exception:
        # Never fail test startup due to env loading
        pass

