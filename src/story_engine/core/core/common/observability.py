"""
Observability utilities: env-driven logging config, JSON formatting,
lightweight error taxonomy, and context helpers (trace/correlation IDs).

This module builds on core.common.logging and should be imported early
in CLI entrypoints and worker __main__ modules.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .logging import JsonLogFormatter, configure_json_logging, configure_logging


# Context IDs available to formatters via record attributes
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")


def _level_from_env(value: str | None) -> int:
    mapping = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }
    return mapping.get((value or "INFO").upper(), logging.INFO)


def init_logging_from_env() -> None:
    """Configure logging based on environment variables.

    LOG_FORMAT: json|text (default json)
    LOG_LEVEL: DEBUG|INFO|WARNING|ERROR (default INFO)
    LOG_DEST: stdout|stderr|file (default stdout)
    LOG_FILE_PATH: path if LOG_DEST=file
    LOG_SERVICE_NAME: logical service name for attribution
    TRACE_ID: preset trace id for correlation (else auto)
    """
    level = _level_from_env(os.getenv("LOG_LEVEL"))
    fmt = (os.getenv("LOG_FORMAT") or "json").lower()
    dest = (os.getenv("LOG_DEST") or "stdout").lower()
    service = os.getenv("LOG_SERVICE_NAME") or "story-engine"
    trace = os.getenv("TRACE_ID") or str(uuid.uuid4())

    # Set context vars
    trace_id_var.set(trace)

    # Base config
    if fmt == "json":
        configure_json_logging(level)
    else:
        configure_logging(level)

    root = logging.getLogger()

    # Switch handler destination if requested
    if dest in {"stdout", "stderr"}:
        stream = sys.stdout if dest == "stdout" else sys.stderr
        for h in list(root.handlers):
            root.removeHandler(h)
        handler = logging.StreamHandler(stream)
        if fmt == "json":
            handler.setFormatter(JsonLogFormatter())
        root.addHandler(handler)
    elif dest == "file":
        path = os.getenv("LOG_FILE_PATH") or "story_engine.log"
        for h in list(root.handlers):
            root.removeHandler(h)
        file_handler = logging.FileHandler(path)
        if fmt == "json":
            file_handler.setFormatter(JsonLogFormatter())
        root.addHandler(file_handler)

    # Attach a service name to logs via a LoggerAdapter factory
    global _service_name
    _service_name = service

    # Inject contextvars into all log records so standard loggers carry IDs
    class _ContextVarFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
            try:
                record.trace_id = trace_id_var.get()
                record.correlation_id = correlation_id_var.get()
                record.service = _service_name
            except Exception:
                pass
            return True

    root.addFilter(_ContextVarFilter())


_service_name = "story-engine"


def get_logger(name: str, **context: Any) -> logging.LoggerAdapter[Any]:
    """Get a LoggerAdapter injecting common context fields."""
    base = logging.getLogger(name)
    ctx = {
        "service": _service_name,
        "trace_id": trace_id_var.get(),
        "correlation_id": correlation_id_var.get(),
    }
    ctx.update({k: v for k, v in context.items() if v is not None})
    return logging.LoggerAdapter(base, ctx)


def set_correlation_id(value: Optional[str]) -> None:
    """Safely set the correlation ID for the current context."""
    try:
        if value is None:
            correlation_id_var.set("")
        else:
            correlation_id_var.set(str(value))
    except Exception:
        # Do not propagate context errors into business logic
        pass


@dataclass(frozen=True)
class ErrorEvent:
    code: str
    component: str
    message: str
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "code": self.code,
            "component": self.component,
            "message": self.message,
        }
        if self.details:
            d["details"] = self.details
        return d


class ErrorCodes:
    GEN_TIMEOUT = "GEN_TIMEOUT"
    GEN_PARSE_ERROR = "GEN_PARSE_ERROR"
    DB_CONN_FAIL = "DB_CONN_FAIL"
    AI_LB_UNAVAILABLE = "AI_LB_UNAVAILABLE"
    CONFIG_INVALID = "CONFIG_INVALID"


def log_exception(logger: logging.Logger | logging.LoggerAdapter, *, code: str, component: str, exc: BaseException, **context: Any) -> None:
    """Emit a structured error event with exception info."""
    evt = ErrorEvent(code=code, component=component, message=str(exc), details=context)
    if isinstance(logger, logging.LoggerAdapter):
        logger.error(json.dumps({"event": "error", **evt.to_dict()}), exc_info=exc)
    else:
        logger = get_logger(logger.name, **context)
        logger.error(json.dumps({"event": "error", **evt.to_dict()}), exc_info=exc)


def should_log(sample_rate: Optional[float] = None) -> bool:
    try:
        import random
        rate = sample_rate if sample_rate is not None else float(os.getenv("LOG_SAMPLING_RATE", "1.0") or 1.0)
        rate = max(0.0, min(1.0, rate))
        return random.random() < rate
    except Exception:
        return True


def info_sampled(logger: logging.Logger | logging.LoggerAdapter, message: str, sample_rate: Optional[float] = None, **extra: Any) -> None:
    if should_log(sample_rate):
        if isinstance(logger, logging.LoggerAdapter):
            logger.info(message, extra=extra)
        else:
            get_logger(logger.name, **extra).info(message)


# ---- Minimal metrics helpers (log-based) ----

def metric_event(name: str, value: Optional[float] = None, **tags: Any) -> None:
    """Emit a lightweight metric as a structured log event.

    Uses the root logger with JSON formatters already configured. This keeps
    the implementation dependency-free while enabling external scraping if desired.
    """
    try:
        logger = logging.getLogger("metrics")
        payload: Dict[str, Any] = {"event": "metric", "metric": name}
        if value is not None:
            payload["value"] = value
        if tags:
            payload.update({k: v for k, v in tags.items() if v is not None})
        logger.info(json.dumps(payload))
    except Exception:
        # Never break business logic due to metrics issues
        pass


def inc_metric(name: str, n: int = 1, **tags: Any) -> None:
    metric_event(name, value=n, type="counter", **tags)


def observe_metric(name: str, value_ms: float, **tags: Any) -> None:
    metric_event(name, value=value_ms, unit="ms", type="timer", **tags)


class timing:
    """Context manager for timing code blocks in milliseconds.

    Example:
        with timing("db.connect_ms", component="db.oracle"):
            connect()
    """

    def __init__(self, metric_name: str, **tags: Any) -> None:
        self.metric_name = metric_name
        self.tags = tags
        self._start: Optional[float] = None

    def __enter__(self):
        try:
            from time import perf_counter
            self._perf = perf_counter  # type: ignore[attr-defined]
            self._start = self._perf()
        except Exception:
            self._start = None
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._start is None:
                return False
            end = self._perf()  # type: ignore[attr-defined]
            elapsed_ms = (end - self._start) * 1000.0
            observe_metric(self.metric_name, elapsed_ms, **self.tags)
        except Exception:
            pass
        return False
