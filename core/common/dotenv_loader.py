from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional


def load_dotenv_keys(path: Optional[str | Path] = None, keys_prefixes: Optional[Iterable[str]] = None) -> None:
    """Lightweight .env loader that only loads selected prefixes.

    - Skips lines that are comments or malformed
    - Only sets env vars for keys matching provided prefixes (e.g., ["DB_", "STORE_ALL"]).
    - Does not raise if file is missing.
    """
    try:
        file = Path(path) if path else Path.cwd() / ".env"
        if not file.exists():
            return

        prefixes = tuple(keys_prefixes or ("DB_", "STORE_ALL"))
        for raw in file.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key or not key.startswith(prefixes):
                continue
            # Respect existing env unless empty
            if not os.environ.get(key):
                os.environ[key] = value
    except Exception:
        # Best-effort; ignore errors
        return

