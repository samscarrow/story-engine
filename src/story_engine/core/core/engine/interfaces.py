from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, runtime_checkable

from ..common.observability import get_logger

# Forward ref to avoid cyclic import during type checking
try:
    from .results import StepResult
except Exception:  # pragma: no cover - during bootstrap
    StepResult = object  # type: ignore


class StepKind(Enum):
    AI_REQUEST = "ai_request"
    TRANSFORM = "transform"
    PERSIST = "persist"
    FETCH = "fetch"
    BRANCH = "branch"


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 1
    base_delay_sec: float = 0.2
    jitter: float = 0.1


@dataclass
class Step:
    """A unit of work in an engine plan.

    - key: unique step identifier within the plan
    - kind: type of step
    - params: runtime parameters; for AI steps, includes prompt/system/etc.
    - func: for TRANSFORM/PERSIST/FETCH steps, a callable to execute
    - depends_on: upstream step keys that must complete successfully
    - retry: retry policy for transient failures
    - timeout_sec: optional timeout per attempt
    - idempotency_key: if provided, used to short-circuit repeated work
    - metadata: arbitrary tags to propagate into logs/traces
    """

    key: str
    kind: StepKind
    params: Dict[str, Any] = field(default_factory=dict)
    func: Optional[Callable[["EngineContext", Dict[str, Any]], Any]] = None
    depends_on: List[str] = field(default_factory=list)
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    timeout_sec: Optional[float] = None
    idempotency_key: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Plan:
    """A directed acyclic graph (DAG) of steps to run."""

    steps: Mapping[str, Step]
    roots: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def linear(*steps: Step, metadata: Optional[Dict[str, Any]] = None) -> "Plan":
        mapping: Dict[str, Step] = {}
        prev: Optional[Step] = None
        for s in steps:
            mapping[s.key] = s
            if prev is not None:
                s.depends_on.append(prev.key)
            prev = s
        roots = [steps[0].key] if steps else []
        return Plan(mapping, roots, metadata or {})


@dataclass
class EngineResult:
    success: bool
    outputs: Dict[str, Any] = field(default_factory=dict)
    step_results: Dict[str, Any] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)


@dataclass
class ValidationReport:
    valid: bool
    issues: List[str] = field(default_factory=list)


@runtime_checkable
class AIClient(Protocol):
    async def generate(self, prompt: str, **kwargs: Any) -> Any:  # returns object with .text
        ...


@runtime_checkable
class ArtifactRepo(Protocol):
    def save(self, kind: str, key: str, payload: Dict[str, Any]) -> None: ...
    def load(self, kind: str, key: str) -> Optional[Dict[str, Any]]: ...


@runtime_checkable
class JobRepo(Protocol):
    def was_processed(self, idempotency_key: str) -> bool: ...
    def mark_processed(self, idempotency_key: str) -> None: ...


@runtime_checkable
class StoryRepo(Protocol):
    def save_story_state(self, story_id: str, state: Dict[str, Any]) -> None: ...
    def load_story_state(self, story_id: str) -> Optional[Dict[str, Any]]: ...


@dataclass
class EngineContext:
    """Runtime dependencies injected into engines and steps."""

    ai: Optional[AIClient] = None
    artifacts: Optional[ArtifactRepo] = None
    jobs: Optional[JobRepo] = None
    stories: Optional[StoryRepo] = None
    logger: logging.Logger = field(default_factory=lambda: get_logger("engine"))
    config: Dict[str, Any] = field(default_factory=dict)
    rng_seed: Optional[int] = None
    clock_fn: Optional[Callable[[], float]] = None
    # Results of previously completed steps, populated by the orchestrator
    results: Dict[str, Any] = field(default_factory=dict)
    # Typed metadata per step (status, elapsed, error), maintained by the orchestrator
    meta: Dict[str, StepResult] = field(default_factory=dict)  # type: ignore[type-arg]

    def get_result(self, step_key: str, default: Any | None = None) -> Any:
        """Read-only accessor for prior step results."""
        return self.results.get(step_key, default)


class BaseEngine(Protocol):
    """Contract for story engines executed by the EngineOrchestrator."""

    def describe(self) -> str: ...

    def plan(self, inputs: Dict[str, Any]) -> Plan: ...

    async def execute(self, plan: Plan, ctx: EngineContext) -> EngineResult: ...

    def validate(self, result: EngineResult) -> ValidationReport: ...

    def compensate(self, plan: Plan, ctx: EngineContext) -> None: ...

    def idempotency_key(self, inputs: Dict[str, Any]) -> Optional[str]: ...

    def estimate_cost(self, plan: Plan) -> Optional[Dict[str, Any]]: ...
