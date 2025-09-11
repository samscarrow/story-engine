"""
Message contracts (v1) for scene generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class SceneRequest:
    job_id: str
    outline_id: str | None
    beat_name: str | None
    prompt: str
    characters: List[Dict[str, Any]]
    constraints: Dict[str, Any]

    @staticmethod
    def validate(payload: Dict[str, Any]) -> "SceneRequest":
        required = ("job_id", "prompt", "characters", "constraints")
        for key in required:
            if key not in payload:
                raise ValueError(f"scene.request missing '{key}'")
        if not isinstance(payload["characters"], list):
            raise ValueError("scene.request 'characters' must be a list")
        if not isinstance(payload["constraints"], dict):
            raise ValueError("scene.request 'constraints' must be an object")
        return SceneRequest(
            job_id=str(payload["job_id"]),
            outline_id=(
                payload.get("outline_id")
                if payload.get("outline_id") is not None
                else None
            ),
            beat_name=(
                payload.get("beat_name")
                if payload.get("beat_name") is not None
                else None
            ),
            prompt=str(payload["prompt"]),
            characters=list(payload["characters"]),
            constraints=dict(payload["constraints"]),
        )


@dataclass
class SceneDone:
    job_id: str
    scene_id: str
    scene_description: str
