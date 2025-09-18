"""
Compatibility package for transitional layout.

Historically, modules lived under `story_engine.core.*`. During the Milestone 2
refactor, sources were moved under `story_engine.core.core.*`. To avoid breaking
imports in tests and user code, we alias the legacy package paths to the new
locations at import time.
"""

from __future__ import annotations

import importlib
import sys

__all__ = []


def _alias_pkg(subpkg: str) -> None:
    legacy = f"{__name__}.{subpkg}"
    target = f"{__name__}.core.{subpkg}"
    try:
        module = importlib.import_module(target)
        sys.modules[legacy] = module
    except Exception:
        # If a subpackage doesn't exist yet, ignore; only alias known ones
        pass


# Alias common subpackages used across tests and services
# Import foundational subpackages first so dependent ones (like engine) resolve
for _name in (
    "common",
    "contracts",
    "domain",
    "cache",
    "storage",
    "messaging",
    "orchestration",
    "story_engine",
    "character_engine",
    "engine",
):
    _alias_pkg(_name)
