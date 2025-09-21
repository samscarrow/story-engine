from __future__ import annotations

from pathlib import Path
import os
from typing import Dict, List


def validate_oracle_env() -> Dict[str, List[str]]:
    """Validate Oracle client environment minimally without connecting.

    Returns a dict with keys 'errors' and 'warnings'.
    """
    errors: List[str] = []
    warnings: List[str] = []

    wallet = os.getenv("DB_WALLET_LOCATION") or os.getenv("TNS_ADMIN")
    if not wallet:
        errors.append("Missing DB_WALLET_LOCATION or TNS_ADMIN")
        return {"errors": errors, "warnings": warnings}

    wp = Path(wallet).expanduser().resolve()
    if not wp.exists() or not wp.is_dir():
        errors.append(f"Wallet directory not found: {wp}")
        return {"errors": errors, "warnings": warnings}

    required = [
        "tnsnames.ora",
        "sqlnet.ora",
        "ewallet.p12",
        "cwallet.sso",
    ]
    for name in required:
        if not (wp / name).exists():
            warnings.append(f"Wallet file missing: {(wp / name)}")

    dsn = os.getenv("DB_DSN") or os.getenv("DB_CONNECT_STRING")
    if not dsn:
        warnings.append("Missing DB_DSN/DB_CONNECT_STRING")

    user = os.getenv("DB_USER")
    pwd = os.getenv("DB_PASSWORD")
    if not user or not pwd:
        warnings.append("Missing DB_USER/DB_PASSWORD")

    return {"errors": errors, "warnings": warnings}
