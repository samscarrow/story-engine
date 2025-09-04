"""
Centralized logging configuration.
Call configure_logging() early in entrypoints or tests.
"""

import logging
from typing import Optional


def configure_logging(level: int = logging.INFO, fmt: Optional[str] = None) -> None:
    if fmt is None:
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=level, format=fmt)

