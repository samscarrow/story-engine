"""
Message contracts (v1) for dialogue generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class DialogueRequest:
    job_id: str
    scene_id: str | None
    character_id: str
    opening_line: str | None
    context: Dict[str, Any]

    @staticmethod
    def validate(payload: Dict[str, Any]) -> "DialogueRequest":
        for key in ("job_id", "character_id", "context"):
            if key not in payload:
                raise ValueError(f"dialogue.request missing '{key}'")
        if not isinstance(payload["context"], dict):
            raise ValueError("dialogue.request 'context' must be an object")
        return DialogueRequest(
            job_id=str(payload["job_id"]),
            scene_id=(
                payload.get("scene_id") if payload.get("scene_id") is not None else None
            ),
            character_id=str(payload["character_id"]),
            opening_line=(
                payload.get("opening_line")
                if payload.get("opening_line") is not None
                else None
            ),
            context=dict(payload["context"]),
        )


@dataclass
class DialogueDone:
    job_id: str
    scene_id: str | None
    text: str
