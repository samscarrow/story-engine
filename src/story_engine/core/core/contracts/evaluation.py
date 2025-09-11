"""
Message contracts (v1) for content evaluation and scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class EvaluationRequest:
    job_id: str
    content: str
    criteria: List[str] | None
    options: Dict[str, Any]

    @staticmethod
    def validate(payload: Dict[str, Any]) -> "EvaluationRequest":
        for key in ("job_id", "content", "options"):
            if key not in payload:
                raise ValueError(f"evaluation.request missing '{key}'")
        if not isinstance(payload["options"], dict):
            raise ValueError("evaluation.request 'options' must be an object")
        criteria = payload.get("criteria")
        if criteria is not None and not isinstance(criteria, list):
            raise ValueError("evaluation.request 'criteria' must be a list if provided")
        return EvaluationRequest(
            job_id=str(payload["job_id"]),
            content=str(payload["content"]),
            criteria=(list(criteria) if criteria is not None else None),
            options=dict(payload["options"]),
        )


@dataclass
class EvaluationDone:
    job_id: str
    evaluation_text: str
