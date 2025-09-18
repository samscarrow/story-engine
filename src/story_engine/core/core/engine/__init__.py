"""
Engine framework: interfaces, orchestrator, and repository abstractions.

This package provides a light-weight, testable execution model for story
engines. Engines declare Plans of Steps (AI, Transform, Persist, etc.) and an
EngineOrchestrator executes them with retries, timeouts, and idempotency.

The design favors composability and determinism:
- Engines implement a stable BaseEngine interface
- EngineContext wires AI, repos, clock, RNG, and logging
- Plans/Steps are plain dataclasses that serialize cleanly
- Local-first persistence via in-memory and filesystem repos
"""

from .interfaces import (
    BaseEngine,
    EngineContext,
    StepKind,
    RetryPolicy,
    Step,
    Plan,
    EngineResult,
    ValidationReport,
)

from .orchestrator import EngineOrchestrator
from .repos import (
    ArtifactRepo,
    JobRepo,
    StoryRepo,
    InMemoryRepos,
    FileArtifactRepo,
    LocalRepos,
)
from .adapters import NarrativePipelineEngine

__all__ = [
    "BaseEngine",
    "EngineContext",
    "StepKind",
    "RetryPolicy",
    "Step",
    "Plan",
    "EngineResult",
    "ValidationReport",
    "EngineOrchestrator",
    "ArtifactRepo",
    "JobRepo",
    "StoryRepo",
    "InMemoryRepos",
    "FileArtifactRepo",
    "LocalRepos",
    "NarrativePipelineEngine",
]
