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
        # Attach common context fields if present in extra
        for key in ("correlation_id", "trace_id", "message_id", "job_id", "step"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
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
