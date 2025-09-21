import pytest

from story_engine.core.core.orchestration.response_normalizer import normalize_openai_chat


def test_normalize_plain_content():
    data = {
        "model": "mock-model",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "Hello world"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 15},
    }

    result = normalize_openai_chat(data, headers={"X-Selected-Model": "mock-model"})
    assert result["text"] == "Hello world"
    assert result["reasoning"] == ""
    assert result["meta"]["effective_model"] == "mock-model"


def test_normalize_segmented_reasoning_content():
    data = {
        "model": "reasoning-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "Final answer."},
                    ],
                    "reasoning": [
                        {"type": "thought", "text": "Thinking step 1."},
                        {"type": "thought", "text": "Thinking step 2."},
                    ],
                },
                "finish_reason": "stop",
            }
        ],
    }

    result = normalize_openai_chat(data)
    assert result["text"] == "Final answer."
    assert "Thinking step" in result["reasoning"]
    assert "step 2" in result["reasoning"]


def test_normalize_falls_back_to_reasoning_when_content_missing():
    data = {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "reasoning": [{"type": "thought", "text": "Reasoned output."}],
                },
            }
        ]
    }

    result = normalize_openai_chat(data)
    assert result["text"] == "Reasoned output."
    assert result["reasoning"] == "Reasoned output."
