"""
Initial tests for the consolidated LLMOrchestrator.
"""

import pytest
from story_engine.core.orchestration.llm_orchestrator import (
    LLMOrchestrator,
    LLMConfig,
    ModelProvider,
)


def test_orchestrator_instantiation():
    """
    Tests if the LLMOrchestrator can be instantiated without errors.
    """
    try:
        orchestrator = LLMOrchestrator()
        assert orchestrator is not None
        assert isinstance(orchestrator, LLMOrchestrator)
    except Exception as e:
        pytest.fail(f"Failed to instantiate LLMOrchestrator: {e}")


def test_register_provider():
    """
    Tests if a provider can be registered successfully.
    """
    orchestrator = LLMOrchestrator()
    config = LLMConfig(
        provider=ModelProvider.KOBOLDCPP, endpoint="http://localhost:5001"
    )
    try:
        orchestrator.register_provider("test_kobold", config)
        assert "test_kobold" in orchestrator.providers
        assert orchestrator.active_provider == "test_kobold"
    except Exception as e:
        pytest.fail(f"Failed to register provider: {e}")
