"""
Message contracts (v1) for plot generation.
Lightweight schema checks to keep runtime stdlib-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class PlotRequest:
    job_id: str
    prompt: str
    constraints: Dict[str, Any]

    @staticmethod
    def validate(payload: Dict[str, Any]) -> "PlotRequest":
        for key in ("job_id", "prompt", "constraints"):
            if key not in payload:
                raise ValueError(f"plot.request missing '{key}'")
        if not isinstance(payload["constraints"], dict):
            raise ValueError("plot.request 'constraints' must be an object")
        return PlotRequest(
            job_id=str(payload["job_id"]),
            prompt=str(payload["prompt"]),
            constraints=dict(payload["constraints"]),
        )


@dataclass
class PlotDone:
    job_id: str
    outline_id: str
    outline_ref: str

