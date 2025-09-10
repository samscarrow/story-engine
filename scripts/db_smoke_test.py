#!/usr/bin/env python3
"""
Quick database smoke test using core.storage.

Reads env: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD.
Usage:
  python scripts/db_smoke_test.py --workflow test_run
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from story_engine.core.storage import get_database_connection
from story_engine.core.common.dotenv_loader import load_dotenv_keys


def main() -> None:
    p = argparse.ArgumentParser(description="DB smoke test")
    p.add_argument("--workflow", default="db_smoke")
    args = p.parse_args()

    # Load DB_* and STORE_ALL from .env, and try .env.oracle if present
    load_dotenv_keys()
    try:
        load_dotenv_keys(path=".env.oracle")
    except Exception:
        pass

    db_type = os.getenv("DB_TYPE", "postgresql").lower()
    if db_type == "oracle":
        db = get_database_connection(
            db_type="oracle",
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            dsn=os.getenv("DB_DSN"),
            wallet_location=os.getenv("DB_WALLET_LOCATION"),
            wallet_password=os.getenv("DB_WALLET_PASSWORD"),
        )
    else:
        db = get_database_connection(
            db_type="postgresql",
            db_name=os.getenv("DB_NAME", "postgres"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            sslmode=os.getenv("DB_SSLMODE"),
            sslrootcert=os.getenv("DB_SSLROOTCERT"),
            sslcert=os.getenv("DB_SSLCERT"),
            sslkey=os.getenv("DB_SSLKEY"),
        )
    db.connect()
    print("Connected")
    db.store_output(args.workflow, {"ok": True})
    print("Stored")
    print("Fetched:", db.get_outputs(args.workflow))
    db.disconnect()
    print("Closed")


if __name__ == "__main__":
    main()

