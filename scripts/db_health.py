#!/usr/bin/env python3
"""
Oracle database health check.

- Loads env from `.env` and `.env.oracle` when available
- Supports both DB_* and ORACLE_* variables
- Optionally uses a session pool and retries (via OracleConnection)

Usage:
  python scripts/db_health.py --json           # machine-readable
  python scripts/db_health.py --verbose        # human-readable

Required (either DB_* or ORACLE_*):
  - DB_TYPE=oracle
  - DB_USER / ORACLE_USER
  - DB_PASSWORD / ORACLE_PASSWORD
  - DB_DSN / ORACLE_DSN (e.g., "mainbase_high")
  - DB_WALLET_LOCATION / ORACLE_WALLET_DIR (path containing tnsnames.ora, etc.)

Optional:
  - ORACLE_USE_POOL=1
  - ORACLE_POOL_MIN/ORACLE_POOL_MAX/ORACLE_POOL_INC/ORACLE_POOL_TIMEOUT
  - ORACLE_RETRY_ATTEMPTS / ORACLE_RETRY_BACKOFF
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_here = Path(__file__).resolve()
_root = _here.parents[1]
_src = _root / "src"
for p in (str(_src), str(_root)):
    if p not in sys.path:
        sys.path.insert(0, p)
from story_engine.core.storage.database import OracleConnection

try:
    # Optional dotenv support
    from story_engine.core.common.dotenv_loader import load_dotenv_keys
except Exception:  # pragma: no cover - optional dependency
    def load_dotenv_keys(path: str | None = None) -> None:  # type: ignore
        return


def _truthy(s: str | None) -> bool:
    if s is None:
        return False
    return str(s).strip().lower() in {"1", "true", "yes", "on"}


def resolve_env() -> dict:
    # Load env files if present
    try:
        load_dotenv_keys()
        load_dotenv_keys(path=".env.oracle")
    except Exception:
        pass

    # Support DB_* first, fall back to ORACLE_*
    user = os.getenv("DB_USER") or os.getenv("ORACLE_USER")
    password = os.getenv("DB_PASSWORD") or os.getenv("ORACLE_PASSWORD")
    dsn = os.getenv("DB_DSN") or os.getenv("ORACLE_DSN")
    wallet = os.getenv("DB_WALLET_LOCATION") or os.getenv("ORACLE_WALLET_DIR")

    cfg = {
        "user": user,
        "password": password,
        "dsn": dsn,
        "wallet_location": wallet,
    }

    # Pooling and retry settings
    cfg["use_pool"] = _truthy(os.getenv("ORACLE_USE_POOL", "1"))
    cfg["pool_min"] = int(os.getenv("ORACLE_POOL_MIN", "1"))
    cfg["pool_max"] = int(os.getenv("ORACLE_POOL_MAX", "4"))
    cfg["pool_increment"] = int(os.getenv("ORACLE_POOL_INC", "1"))
    cfg["pool_timeout"] = int(os.getenv("ORACLE_POOL_TIMEOUT", "60"))
    cfg["retry_attempts"] = int(os.getenv("ORACLE_RETRY_ATTEMPTS", "3"))
    cfg["retry_backoff_seconds"] = float(os.getenv("ORACLE_RETRY_BACKOFF", "1.0"))
    cfg["ping_on_connect"] = True
    return cfg


def check_oracle(verbose: bool = False) -> tuple[bool, dict]:
    env = resolve_env()
    missing = [k for k in ("user", "password", "dsn", "wallet_location") if not env.get(k)]
    if missing:
        return False, {
            "ok": False,
            "reason": "missing_env",
            "missing": missing,
            "hint": "Provide DB_USER/DB_PASSWORD/DB_DSN/DB_WALLET_LOCATION or ORACLE_* equivalents.",
        }

    # Ensure DB_TYPE=oracle for consistency (not strictly required by this script)
    if os.getenv("DB_TYPE", "oracle").lower() != "oracle":
        os.environ["DB_TYPE"] = "oracle"

    started = time.time()
    try:
        conn = OracleConnection(**env)
        conn.connect()
        healthy = conn.healthy()
        # minimal user/schema check
        cur = conn.conn.cursor()  # type: ignore[union-attr]
        cur.execute("SELECT USER FROM DUAL")
        user = cur.fetchone()[0]
        cur.close()
        conn.disconnect()
        finished = time.time()
        return healthy, {
            "ok": bool(healthy),
            "user": user,
            "latency_ms": round((finished - started) * 1000.0, 1),
            "pooled": bool(env.get("use_pool")),
        }
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        hint = None
        if "ORA-12506" in msg:
            hint = "ADB may be paused; resume instance or retry later."
        elif "ORA-12154" in msg:
            hint = "Check TNS_ADMIN/tnsnames.ora and DSN service name."
        elif "ORA-12514" in msg:
            hint = "Listener may not know the service; verify DSN service."
        elif "ORA-12541" in msg:
            hint = "No listener; verify network access and service endpoint."
        return False, {"ok": False, "error": msg, "hint": hint}


def main() -> int:
    ap = argparse.ArgumentParser(description="Oracle DB health check")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--verbose", action="store_true", help="verbose text output")
    args = ap.parse_args()

    ok, info = check_oracle(verbose=args.verbose)
    if args.json:
        print(json.dumps(info))
    else:
        if ok:
            print("✓ Oracle healthy:")
            print(f"  user={info.get('user')} pooled={info.get('pooled')} latency_ms={info.get('latency_ms')}")
        else:
            print("✗ Oracle unhealthy:")
            if "missing" in info:
                print(f"  Missing: {', '.join(info['missing'])}")
                print(f"  Hint: {info.get('hint')}")
            else:
                print(f"  Error: {info.get('error')}")
                if info.get("hint"):
                    print(f"  Hint: {info['hint']}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
