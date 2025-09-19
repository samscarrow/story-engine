"""Tooling utilities for Story Engine."""

__all__ = ["__version__"]

# Expose package version for tooling components
try:  # pragma: no cover - version only available in installed package
    from importlib.metadata import version

    __version__ = version("story-engine")
except Exception:  # pragma: no cover - fallback when metadata missing
    __version__ = "0.0.0"
