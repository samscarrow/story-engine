"""
Helper to construct LLMOrchestrator from the root config.yaml.
Maps YAML structure to LLMConfig and registers providers.
"""

from __future__ import annotations

from typing import Any, Dict
import os

from .llm_orchestrator import LLMOrchestrator, LLMConfig, ModelProvider
from story_engine.core.common.config import load_config


def create_orchestrator_from_yaml(path: str = "config.yaml") -> LLMOrchestrator:
    cfg: Dict[str, Any] = load_config(path)
    orch = LLMOrchestrator(fail_on_all_providers=True)

    providers = cfg.get("llm", {}).get("providers", [])
    for p in providers:
        name = p.get("name")
        provider_key = p.get("provider")
        endpoint = p.get("endpoint")
        model = p.get("model")
        defaults = p.get("defaults", {})
        temperature = defaults.get("temperature", 0.7)
        max_tokens = defaults.get("max_tokens", 600)
        # Allow environment override to speed up tests or tighten SLAs
        timeout = int(os.environ.get("LLM_TIMEOUT_SECS", defaults.get("timeout", 60)))

        llm_conf = LLMConfig(
            provider=ModelProvider(provider_key),
            endpoint=str(endpoint),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        orch.register_provider(name, llm_conf)

    # Active provider (prefer explicit; fall back to first defined)
    active = cfg.get("llm", {}).get("active")
    if active:
        orch.set_active(active)
    elif providers:
        orch.set_active(providers[0].get("name"))

    # Prefer small models toggle from YAML or environment
    prefer_small_yaml = cfg.get("llm", {}).get("prefer_small_models")
    if isinstance(prefer_small_yaml, bool):
        orch.prefer_small_models = prefer_small_yaml
    else:
        env_val = os.environ.get("LM_PREFER_SMALL")
        if env_val is not None:
            orch.prefer_small_models = str(env_val).strip().lower() in {"1", "true", "yes", "on"}

    return orch

