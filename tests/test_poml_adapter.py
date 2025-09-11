"""
Basic tests for POML adapter rendering to ensure integration wiring works.
These tests do not validate full POML semantics, only that rendering returns strings.
"""

from story_engine.poml.lib.poml_integration import StoryEnginePOMLAdapter


def test_scene_prompt_renders_string():
    adapter = StoryEnginePOMLAdapter()
    beat = {"name": "Setup", "purpose": "Establish normal", "tension": 0.2}
    characters = [{"id": "c1", "name": "Alice", "role": "protagonist"}]

    prompt = adapter.get_scene_prompt(
        beat=beat, characters=characters, previous_context=""
    )
    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_dialogue_prompt_renders_string():
    adapter = StoryEnginePOMLAdapter()
    character = {
        "id": "c1",
        "name": "Alice",
        "role": "protagonist",
        "traits": ["brave"],
    }
    scene = type(
        "S",
        (),
        {
            "__dict__": {
                "name": "Setup",
                "situation": "A quiet room",
                "emphasis": {"c1": "neutral"},
                "goals": {"c1": "greet"},
                "sensory": {},
            }
        },
    )()
    ctx = {"emphasis": "neutral", "goal": "greet"}

    prompt = adapter.get_dialogue_prompt(
        character=character, scene=scene, dialogue_context=ctx
    )
    assert isinstance(prompt, str)
    assert len(prompt) > 0
