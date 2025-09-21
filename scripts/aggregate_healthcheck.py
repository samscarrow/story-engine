#!/usr/bin/env python3
"""Aggregate Oracle healthcheck NDJSON logs into simple SLI summary.

Usage:
  python scripts/aggregate_healthcheck.py oracle_health.ndjson

Outputs a markdown summary to stdout suitable for $GITHUB_STEP_SUMMARY.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path


def pct(p, total):
    return (100.0 * p / total) if total else 0.0


def main(path: str) -> int:
    p = Path(path)
    if not p.exists():
        print(f"No log file found at {path}")
        return 1
    success = 0
    failures = 0
    latencies: list[int] = []
    attempts: list[int] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        msg = obj.get("message", "")
        if msg == "oracle connect ok":
            success += 1
            if isinstance(obj.get("elapsed_ms"), int):
                latencies.append(obj["elapsed_ms"])
            if isinstance(obj.get("attempt"), int):
                attempts.append(obj["attempt"])
        elif obj.get("level") in {"ERROR", "CRITICAL"} and obj.get("message") == "error":
            # terminal DB_CONN_FAIL events are emitted with message="error" and code
            code = obj.get("code") or obj.get("error_code") or ""
            if code == "DB_CONN_FAIL":
                failures += 1

    total = success + failures
    p50 = int(statistics.median(latencies)) if latencies else 0
    p95 = int(statistics.quantiles(latencies, n=20)[18]) if len(latencies) >= 20 else (max(latencies) if latencies else 0)
    avg_attempt = round(statistics.mean(attempts), 2) if attempts else 0.0

    print("## Oracle Healthcheck Summary")
    print()
    print(f"- Runs: {total}")
    print(f"- Success: {success} ({pct(success, total):.2f}%)")
    print(f"- Failures: {failures} ({pct(failures, total):.2f}%)")
    print(f"- P50 connect latency: {p50} ms")
    print(f"- P95 connect latency: {p95} ms")
    print(f"- Mean attempts: {avg_attempt}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: aggregate_healthcheck.py <ndjson>")
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))

