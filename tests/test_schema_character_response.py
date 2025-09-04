"""
Contract test for character response shape using a mocked LLM output.
Does not call real providers; validates parsing expectations at the edge.
"""

import json


def validate_character_response_shape(payload: dict) -> None:
    assert isinstance(payload, dict)
    assert "dialogue" in payload and isinstance(payload["dialogue"], str)
    assert "thought" in payload and isinstance(payload["thought"], str)
    assert "action" in payload and isinstance(payload["action"], str)
    assert "emotional_shift" in payload and isinstance(payload["emotional_shift"], dict)


def test_character_response_schema_minimal():
    # Simulate a model output used elsewhere in the codebase
    raw = json.dumps({
        "dialogue": "Hello.",
        "thought": "Stay calm.",
        "action": "Nods.",
        "emotional_shift": {"anger": 0, "doubt": 0.1, "fear": 0, "compassion": 0}
    })
    parsed = json.loads(raw)
    validate_character_response_shape(parsed)

