"""
Centralized logging configuration.
Call configure_logging() early in entrypoints or tests.
Includes a simple JSON formatter for structured logs.
"""

import json
import logging
from typing import Optional


def configure_logging(level: int = logging.INFO, fmt: Optional[str] = None) -> None:
    if fmt is None:
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=level, format=fmt)


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        payload = {
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
        }

        # Attach common structured context fields if present
        base_keys = (
            "correlation_id",
            "trace_id",
            "message_id",
            "job_id",
            "step",
            "service",
            "endpoint",
            "model",
            "provider",
            "beat",
            "character_id",
            "elapsed_ms",
            "ok",
            "len",
            "event",
            "attempt",
            "retry_in_s",
            "error",
            "error_code",
        )
        # Basic redaction of sensitive keys
        redact = {"api_key", "authorization", "password", "db_password", "oracle_password"}
        for key in base_keys:
            if hasattr(record, key):
                val = getattr(record, key)
                payload[key] = "[REDACTED]" if key in redact else val

        # Include any additional extras provided via LoggerAdapter/extra
        std_attrs = {
            "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
            "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
            "created", "msecs", "relativeCreated", "thread", "threadName", "processName",
            "process", "asctime",
        }
        for k, v in record.__dict__.items():
            if k in payload or k in std_attrs or k.startswith("_"):
                continue
            if callable(v):
                continue
            try:
                payload[k] = "[REDACTED]" if k in redact else v
                # Ensure JSON-serializable
                json.dumps(payload[k])
            except Exception:
                payload[k] = str(v)

        return json.dumps(payload, ensure_ascii=False)


def configure_json_logging(level: int = logging.INFO) -> None:
    """Configure root logger to emit JSON logs.

    This is stdlib-only and keeps existing handlers minimal.
    """
    root = logging.getLogger()
    root.setLevel(level)
    # Clear existing handlers if basicConfig ran
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    root.addHandler(handler)
