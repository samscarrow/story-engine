#!/usr/bin/env python3
"""
HPQ Pipeline Demo CLI

Usage:
  python -m story_engine.scripts.run_hpq \
      --title "The Trial" \
      --premise "Roman prefect under pressure decides a prophet's fate" \
      --characters pilate caiaphas crowd \
      --beats 1

Environment knobs:
  LM_ENDPOINT=http://localhost:8000 (ai-lb) or direct LM Studio
  LM_HQ_MODEL=llama-3.1-70b-instruct (optional)
  LM_FAST_MODEL=gemma-2-9b-instruct (optional)
  HPQ_FORCE_24B=1 (optional)
  HPQ_CACHE_TTL=1800 (seconds)
  LOG_FORMAT=json LOG_LEVEL=INFO
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any, Dict, List

from llm_observability import get_logger, init_logging_from_env
from story_engine.core.core.story_engine.hpq_pipeline import HPQPipeline, HPQOptions


_log = get_logger("hpq.cli")


def _make_characters(names: List[str]) -> List[Dict[str, Any]]:
    # Basic mock character objects; real systems can pass full descriptors
    out = []
    for n in names:
        out.append(
            {
                "id": n.lower().replace(" ", "_"),
                "name": n,
                "role": "unknown",
                "traits": [],
            }
        )
    return out


async def main_async(args):
    init_logging_from_env()
    opts = HPQOptions(
        candidates=args.candidates,
        threshold_avg=args.threshold,
        threshold_high=getattr(args, "threshold_high", 8.3),
        max_tokens_fast=args.max_tokens_fast,
        max_tokens_hq=args.max_tokens_hq,
        temperature_fast=args.temp_fast,
        temperature_hq=args.temp_hq,
        canary_pct=args.canary,
        budget_ms=args.budget_ms,
        concurrency=getattr(args, "concurrency", 2),
        use_structured_scoring=getattr(args, "structured_scoring", False),
    )
    hpq = HPQPipeline(opts=opts)

    # Minimal beat spec based on flags
    beat = {"name": "HPQ Scene", "purpose": args.premise, "tension": 0.7}
    characters = _make_characters(args.characters)

    _log.info("hpq.start", extra={"candidates": opts.candidates})
    result = await hpq.craft_scene_hpq(beat, characters)
    _log.info(
        "hpq.done",
        extra={
            "best_avg": result.get("best_avg"),
            "used_model": result.get("used_model"),
        },
    )

    # Pretty print excerpt
    print("\n=== Situation ===\n")
    print(result.get("situation", "").strip())
    print("\n=== Final (excerpt) ===\n")
    final = (result.get("final") or "").strip()
    print(final[:2000])


def build_parser():
    ap = argparse.ArgumentParser(description="Run HPQ pipeline demo")
    ap.add_argument("--title", default="The Trial")
    ap.add_argument(
        "--premise", default="Roman prefect under pressure decides a prophet's fate"
    )
    ap.add_argument(
        "--characters",
        nargs="+",
        default=["Pontius Pilate", "Caiaphas", "Crowd Representative"],
    )
    ap.add_argument("--beats", type=int, default=1)
    ap.add_argument("--candidates", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=7.5)
    ap.add_argument("--threshold-high", dest="threshold_high", type=float, default=8.3)
    ap.add_argument("--max-tokens-fast", dest="max_tokens_fast", type=int, default=600)
    ap.add_argument("--max-tokens-hq", dest="max_tokens_hq", type=int, default=800)
    ap.add_argument("--temp-fast", dest="temp_fast", type=float, default=0.7)
    ap.add_argument("--temp-hq", dest="temp_hq", type=float, default=0.6)
    ap.add_argument(
        "--canary",
        type=float,
        default=0.0,
        help="0..1 fraction of calls escalated for canary",
    )
    ap.add_argument("--budget-ms", dest="budget_ms", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument(
        "--structured-scoring", action="store_true", help="Enable two-pass JSON scoring"
    )
    return ap


def main():
    args = build_parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
