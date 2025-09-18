from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class TimeBudget:
    """Simple wall-clock time budget tracker in milliseconds."""

    total_ms: int
    start_monotonic: float | None = None

    def start(self) -> None:
        if self.start_monotonic is None:
            self.start_monotonic = time.monotonic()

    def remaining_ms(self) -> int:
        if self.start_monotonic is None:
            return self.total_ms
        elapsed = int((time.monotonic() - self.start_monotonic) * 1000)
        rem = self.total_ms - elapsed
        return max(0, rem)

    def remaining_seconds(self) -> float:
        return max(0.0, self.remaining_ms() / 1000.0)

