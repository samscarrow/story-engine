"""
Ensure OrchestratedStoryEngine loads narrative profiles from config.yaml.
This test does not perform LLM calls.
"""

from story_engine.core.story_engine.story_engine_orchestrated import OrchestratedStoryEngine, StoryComponent


def test_profiles_loaded_from_config_yaml():
    engine = OrchestratedStoryEngine()
    profiles = engine.component_profiles
    # Check expected keys exist
    assert StoryComponent.CHARACTER_DIALOGUE in profiles
    assert "temperature" in profiles[StoryComponent.CHARACTER_DIALOGUE]
    assert "max_tokens" in profiles[StoryComponent.CHARACTER_DIALOGUE]
    # Assert values reflect default config.yaml (dialogue temp 0.9, max_tokens 500)
    assert abs(profiles[StoryComponent.CHARACTER_DIALOGUE]["temperature"] - 0.9) < 1e-6
    assert profiles[StoryComponent.CHARACTER_DIALOGUE]["max_tokens"] == 500


