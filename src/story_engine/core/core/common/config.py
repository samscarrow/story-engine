"""
Unified configuration loader for Story Engine.
Loads a single root YAML config (config.yaml) and exposes a dict.
Environment variables may override certain keys (e.g., endpoints, API keys).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


DEFAULT_CONFIG: Dict[str, Any] = {
    "llm": {
        "providers": [
            {
                "name": "lmstudio",
                "provider": "lmstudio",
                "endpoint": os.environ.get("LMSTUDIO_ENDPOINT", "http://localhost:1234"),
                "model": os.environ.get("LMSTUDIO_MODEL", "gemma-2-27b"),
                "defaults": {"temperature": 0.8, "max_tokens": 600},
            }
        ]
    },
    "orchestrator": {
        "timeout": 60,
        "retries": {"attempts": 2, "base_delay": 0.8},
    },
    "poml": {
        "template_paths": ["templates/", "components/", "gallery/"],
        "cache": {"enabled": True, "ttl_seconds": 3600},
        "strict_mode": False,
    },
    "simulation": {"max_concurrent": 5, "validate_schema": True, "use_poml": False},
    "narrative": {
        "plot_structure": {"temperature": 0.7, "max_tokens": 800},
        "scene_details": {"temperature": 0.8, "max_tokens": 1000},
        "dialogue": {"temperature": 0.9, "max_tokens": 500},
        "evaluation": {"temperature": 0.3, "max_tokens": 400},
    },
    "messaging": {
        "type": os.environ.get("MQ_TYPE", "inmemory"),
        "url": os.environ.get("MQ_URL", "amqp://localhost"),
        "queue_prefix": os.environ.get("QUEUE_PREFIX", "story"),
        "max_retries": int(os.environ.get("MAX_RETRIES", "3") or 3),
        "prefetch": int(os.environ.get("PREFETCH", "1") or 1),
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path = "config.yaml") -> Dict[str, Any]:
    """Load root YAML configuration with sane defaults.

    Args:
        path: Path to root config.yaml

    Returns:
        Merged configuration dictionary
    """
    cfg = DEFAULT_CONFIG.copy()

    config_path = Path(path)
    if yaml is not None and config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            try:
                loaded = yaml.safe_load(f) or {}
                if isinstance(loaded, dict):
                    _deep_merge(cfg, loaded)
            except Exception:
                # Keep defaults if parse fails
                pass

    # Env overrides for common keys
    lm_ep = os.environ.get("LM_ENDPOINT")
    if lm_ep:
        for p in cfg.get("llm", {}).get("providers", []):
            if p.get("provider") == "lmstudio":
                p["endpoint"] = lm_ep

    # Env override for LM Studio model
    lm_model = os.environ.get("LMSTUDIO_MODEL")
    if lm_model:
        for p in cfg.get("llm", {}).get("providers", []):
            if p.get("provider") == "lmstudio":
                p["model"] = lm_model

    return cfg
