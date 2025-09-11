from __future__ import annotations

import importlib
import sys

__all__ = []


def _alias_pkg(subpkg: str) -> None:
    legacy = f"{__name__}.{subpkg}"
    target = f"{__name__}.poml.{subpkg}"
    try:
        module = importlib.import_module(target)
        sys.modules[legacy] = module
    except Exception:
        pass


for _name in (
    "lib",
    "integration",
):
    _alias_pkg(_name)
