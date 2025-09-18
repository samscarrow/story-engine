from __future__ import annotations

from typing import Any, Dict, Optional


def normalize_kobold(data: Dict[str, Any], model: Optional[str]) -> Dict[str, Any]:
    """Return a minimal normalized meta for KoboldCpp responses.

    - model: taken from provider config or response (if present)
    - usage: not available; return None
    - reasoning: not available; return empty string
    """
    meta = {
        "effective_model": model or data.get("model") or None,
        "usage": None,
        "reasoning": "",
    }
    return meta

