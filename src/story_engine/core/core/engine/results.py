from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class StepStatus(Enum):
    SUCCESS = "success"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class StepResult:
    """Typed metadata for a step execution.

    Stores status, elapsed time, and optional payload summary/error. The raw
    payload remains available to steps via `EngineContext.results` to preserve
    backward compatibility with existing code.
    """

    status: StepStatus
    elapsed_ms: float = 0.0
    summary: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

