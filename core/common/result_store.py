from __future__ import annotations

import os
from typing import Any, Dict


def store_workflow_output(workflow_name: str, output: Dict[str, Any]) -> None:
    """Best-effort store of workflow output to PostgreSQL if env is configured.

    Reads DB settings from env: DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT.
    If credentials are missing or connection fails, this function is a no-op.
    """
    try:
        from core.storage import get_database_connection
    except Exception:
        return

    password = os.getenv("DB_PASSWORD")
    if not password:
        # No credentials provided; skip silently
        return

    try:
        db = get_database_connection(
            db_type="postgresql",
            db_name=os.getenv("DB_NAME", "story_db"),
            user=os.getenv("DB_USER", "story"),
            password=password,
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
        )
        db.connect()
        db.store_output(workflow_name, output)
    except Exception:
        # Best-effort; do not raise
        pass
    finally:
        try:
            if 'db' in locals() and getattr(db, 'conn', None):
                db.disconnect()
        except Exception:
            pass

