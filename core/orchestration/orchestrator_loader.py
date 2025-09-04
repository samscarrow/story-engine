"""
Helper to construct LLMOrchestrator from the root config.yaml.
Maps YAML structure to LLMConfig and registers providers.
"""

from __future__ import annotations

from typing import Any, Dict

from .llm_orchestrator import LLMOrchestrator, LLMConfig, ModelProvider
from core.common.config import load_config


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
        timeout = defaults.get("timeout", 60)

        llm_conf = LLMConfig(
            provider=ModelProvider(provider_key),
            endpoint=str(endpoint),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        orch.register_provider(name, llm_conf)

    # Optional active provider
    active = cfg.get("llm", {}).get("active")
    if active:
        orch.set_active(active)

    return orch
