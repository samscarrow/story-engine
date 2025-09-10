"""
Topic map and validator registry for message contracts (v1).

This module centralizes topic names and per-topic validation so that
adapters (like the in-memory bus) can enforce contracts consistently.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from .plot import PlotRequest
from .scene import SceneRequest
from .dialogue import DialogueRequest
from .evaluation import EvaluationRequest

# Topic names
PLOT_REQUEST = "plot.request"
PLOT_DONE = "plot.done"

SCENE_REQUEST = "scene.request"
SCENE_DONE = "scene.done"

DIALOGUE_REQUEST = "dialogue.request"
DIALOGUE_DONE = "dialogue.done"

EVALUATION_REQUEST = "evaluation.request"
EVALUATION_DONE = "evaluation.done"

DLQ_PREFIX = "dlq."


def dlq_topic(topic: str) -> str:
    return f"{DLQ_PREFIX}{topic}"


# Validator registry: topic -> callable(payload) that raises on error
Validator = Callable[[Dict[str, Any]], Any]

VALIDATORS: Dict[str, Validator] = {
    PLOT_REQUEST: PlotRequest.validate,
    SCENE_REQUEST: SceneRequest.validate,
    DIALOGUE_REQUEST: DialogueRequest.validate,
    EVALUATION_REQUEST: EvaluationRequest.validate,
}

