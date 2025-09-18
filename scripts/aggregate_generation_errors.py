#!/usr/bin/env python3
"""
Aggregate generation_error_*.json files into a single ndjson ledger and print a summary.

Usage: python scripts/aggregate_generation_errors.py [--delete]

Writes/updates: lb_metrics.ndjson in repo root.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delete", action="store_true", help="Delete source error files after aggregation")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out = root / "lb_metrics.ndjson"
    files = sorted(root.glob("generation_error_*.json"))
    if not files:
        print("No generation_error_*.json files found.")
        return 0

    total = 0
    codes: Dict[str, int] = {}
    providers: Dict[str, int] = {}

    with out.open("a", encoding="utf-8") as f:
        for p in files:
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            rec: Dict[str, Any] = {
                "ts": data.get("timestamp"),
                "type": "generation_error",
                "providers_tried": data.get("providers_tried"),
                "prompt_preview": data.get("prompt_preview"),
                "all_failures": data.get("all_failures"),
            }
            f.write(json.dumps(rec) + "\n")
            total += 1
            # Tally best-effort fields
            for failure in (data.get("all_failures") or []):
                provider = failure.get("provider") or failure.get("provider_name") or "unknown"
                providers[provider] = providers.get(provider, 0) + 1
                code = failure.get("code") or failure.get("error_type") or "unknown"
                codes[code] = codes.get(code, 0) + 1

    print(f"Aggregated {total} error files into {out}")
    if codes:
        print("Top error codes:")
        for k, v in sorted(codes.items(), key=lambda kv: kv[1], reverse=True)[:10]:
            print(f"  {k}: {v}")
    if providers:
        print("Top providers:")
        for k, v in sorted(providers.items(), key=lambda kv: kv[1], reverse=True)[:10]:
            print(f"  {k}: {v}")

    if args.delete:
        for p in files:
            try:
                p.unlink()
            except Exception:
                pass
        print("Source files deleted.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

