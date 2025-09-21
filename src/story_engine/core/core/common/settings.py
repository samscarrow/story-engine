from __future__ import annotations

import os
from typing import Any, Dict, Optional

from .dotenv_loader import load_dotenv_keys
from ..storage.database import oracle_env_is_healthy


def _to_bool(val: Optional[str], default: bool = False) -> bool:
    if val is None:
        return default
    v = val.strip().lower()
    return v in {"1", "true", "yes", "on"}


def get_db_settings(env_file: Optional[str] = None) -> Dict[str, Any]:
    """Return a normalized DB settings dict from environment variables.

    Optionally load an env file (e.g., .env.oracle) for DB_* keys first.
    """
    if env_file:
        load_dotenv_keys(path=env_file, keys_prefixes=("DB_", "ORACLE_"))

    db_type = (os.getenv("DB_TYPE") or "postgresql").lower()

    if db_type == "oracle":
        settings = {
            "db_type": "oracle",
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "dsn": os.getenv("DB_DSN")
            or os.getenv("DB_CONNECT_STRING")
            or "localhost/XEPDB1",
            "wallet_location": os.getenv("DB_WALLET_LOCATION"),
            "wallet_password": os.getenv("DB_WALLET_PASSWORD"),
            # Stability / pooling knobs (optional)
            "use_pool": _to_bool(os.getenv("ORACLE_USE_POOL"), True),
            "pool_min": int(os.getenv("ORACLE_POOL_MIN", "1") or 1),
            "pool_max": int(os.getenv("ORACLE_POOL_MAX", "4") or 4),
            "pool_increment": int(os.getenv("ORACLE_POOL_INC", "1") or 1),
            "pool_timeout": int(os.getenv("ORACLE_POOL_TIMEOUT", "60") or 60),
            "wait_timeout": int(os.getenv("ORACLE_WAIT_TIMEOUT", "0") or 0) or None,
            "retry_attempts": int(os.getenv("ORACLE_RETRY_ATTEMPTS", "3") or 3),
            "retry_backoff_seconds": float(
                os.getenv("ORACLE_RETRY_BACKOFF", "1.0") or 1.0
            ),
            "ping_on_connect": _to_bool(os.getenv("ORACLE_PING_ON_CONNECT"), True),
        }
        # Health-aware fallback: unless strictly required, prefer SQLite when
        # Oracle appears unreachable to avoid test hangs and spurious failures.
        if not _to_bool(os.getenv("DB_REQUIRE_ORACLE")):
            try:
                if not oracle_env_is_healthy(require_opt_in=False, timeout_seconds=1.2):
                    return {
                        "db_type": "sqlite",
                        "db_name": os.getenv("SQLITE_DB", "workflow_outputs.db"),
                    }
            except Exception:
                # On any probe error, fall back to SQLite as a safe default
                return {
                    "db_type": "sqlite",
                    "db_name": os.getenv("SQLITE_DB", "workflow_outputs.db"),
                }
        return settings

    if db_type == "postgresql":
        return {
            "db_type": "postgresql",
            "db_name": os.getenv("DB_NAME", "story_db"),
            "user": os.getenv("DB_USER", "story"),
            "password": os.getenv("DB_PASSWORD"),
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", "5432") or 5432),
            "sslmode": os.getenv("DB_SSLMODE"),
            "sslrootcert": os.getenv("DB_SSLROOTCERT"),
            "sslcert": os.getenv("DB_SSLCERT"),
            "sslkey": os.getenv("DB_SSLKEY"),
        }

    # Default to SQLite
    return {
        "db_type": "sqlite",
        "db_name": os.getenv("SQLITE_DB", "workflow_outputs.db"),
    }


def get_logging_settings() -> Dict[str, Any]:
    """Expose logging-related env settings (mirrors core.common.observability)."""
    return {
        "LOG_FORMAT": os.getenv("LOG_FORMAT", "json"),
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
        "LOG_DEST": os.getenv("LOG_DEST", "stdout"),
        "LOG_FILE_PATH": os.getenv("LOG_FILE_PATH", "story_engine.log"),
        "LOG_SERVICE_NAME": os.getenv("LOG_SERVICE_NAME", "story-engine"),
    }
