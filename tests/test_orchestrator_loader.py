"""
Tests for YAML-based orchestrator loader.
"""

from story_engine.core.orchestration.orchestrator_loader import create_orchestrator_from_yaml


def test_loader_registers_providers():
    orch = create_orchestrator_from_yaml("config.yaml")
    assert hasattr(orch, "providers")
    assert len(orch.providers) >= 1
    assert orch.active_provider is not None


