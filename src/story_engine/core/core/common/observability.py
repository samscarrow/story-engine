"""Backwards-compatible shim over :mod:`llm_observability`.

The legacy project exposed observability helpers from this module.  During the
refactor they moved into the standalone ``llm-observability-suite`` package;
the shim keeps the old import path working for consumers and documentation.
"""

from llm_observability import (  # re-export for consumers still on legacy path
    DBLogger,
    ErrorCodes,
    GenerationDBLogger,
    get_logger,
    inc_metric,
    init_logging_from_env,
    log_exception,
    metric_event,
    observe_metric,
)

from contextlib import contextmanager
from time import perf_counter


@contextmanager
def timing(metric: str, **tags):
    """Lightweight timing context that emits a metric on exit (ms)."""
    t0 = None
    try:
        try:
            t0 = perf_counter()
        except Exception:
            t0 = None
        yield
    finally:
        if t0 is not None:
            try:
                elapsed_ms = (perf_counter() - t0) * 1000.0
                observe_metric(metric, elapsed_ms, **tags)
            except Exception:
                pass


__all__ = [
    "DBLogger",
    "ErrorCodes",
    "GenerationDBLogger",
    "get_logger",
    "init_logging_from_env",
    "log_exception",
    "metric_event",
    "observe_metric",
    "inc_metric",
    "timing",
]
