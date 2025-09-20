#!/usr/bin/env python3
"""Lightweight Oracle healthcheck using story-engine's OracleConnection.

Usage:
  python scripts/oracle_healthcheck.py [--pool]

Reads .env.oracle if present, otherwise uses DB_* env vars.
Emits structured JSON logs; exit code 0 on success, 1 on failure.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # type: ignore

from llm_observability import get_logger
from story_engine.core.core.storage.database import OracleConnection


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", action="store_true", help="use session pool")
    args = parser.parse_args()

    # Logging
    init_logging_from_env()
    log = get_logger(__name__, component="db.oracle", workflow="healthcheck")

    # Load env from standard file if present
    load_dotenv(".env.oracle")

    user = os.getenv("DB_USER") or os.getenv("ORACLE_USER")
    password = os.getenv("DB_PASSWORD") or os.getenv("ORACLE_PASSWORD")
    # Prefer explicit ORACLE_DSN (full Easy Connect) if present, then DB_DSN alias
    dsn_candidates = [
        os.getenv("ORACLE_DSN"),
        os.getenv("DB_DSN"),
    ]
    wallet = os.getenv("DB_WALLET_LOCATION") or os.getenv("TNS_ADMIN")

    missing = [k for k, v in {"DB_USER": user, "DB_PASSWORD": password}.items() if not v]
    if missing:
        log.error("missing required env for healthcheck", extra={"missing": ",".join(missing)})
        return 1

    # Try each DSN candidate with short wait timeouts to fail fast
    last_err: Exception | None = None
    for dsn in [d for d in dsn_candidates if d]:
        try:
            t0 = time.time()
            conn = OracleConnection(
                user=user,  # type: ignore[arg-type]
                password=password,  # type: ignore[arg-type]
                dsn=dsn,  # type: ignore[arg-type]
                wallet_location=wallet,
                use_pool=args.pool,
                pool_min=1,
                pool_max=2,
                pool_timeout=10,
                wait_timeout=5,
                retry_attempts=2,
                retry_backoff_seconds=1.0,
            )
            log.info("attempting oracle connect", extra={"dsn": dsn, "use_pool": args.pool})
            conn.connect()
            ok = conn.healthy()
            conn.disconnect()
            log.info("oracle healthcheck ok", extra={"ok": ok, "elapsed_ms": int((time.time() - t0) * 1000), "dsn": dsn})
            return 0 if ok else 1
        except Exception as e:  # noqa: BLE001
            last_err = e
            log.error("oracle connect attempt failed", extra={"dsn": dsn, "error": str(e)})
            continue
    if last_err:
        log.error("oracle healthcheck failed", extra={"error": str(last_err)})
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
