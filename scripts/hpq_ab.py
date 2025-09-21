#!/usr/bin/env python3
"""
HPQ A/B Harness

Runs baseline (fast-only) vs HPQ on a small golden set of scenarios and reports:
- Average quality score (structured evaluator)
- Median latency per variant
- Escalation rate (HPQ)

Usage:
  uv run python scripts/hpq_ab.py --count 3 --candidates 2 --threshold-low 7.5 --threshold-high 8.3 --structured-scoring
  uv run python scripts/hpq_ab.py --input scenarios.jsonl --report report.json

Notes:
- Requires ai-lb at LM_ENDPOINT (config.yaml default http://localhost:8000)
- Uses HPQ structured scoring for consistent numeric evaluation
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from story_engine.core.core.story_engine.hpq_pipeline import HPQPipeline, HPQOptions


DEFAULT_SCENARIOS = [
    {
        "title": "The Trial",
        "premise": "A Roman prefect must decide a prophet's fate under political pressure.",
        "characters": ["Pontius Pilate", "Caiaphas", "Crowd Representative"],
    },
    {
        "title": "Senate Intrigue",
        "premise": "A junior senator uncovers a plot and must choose between loyalty and truth.",
        "characters": ["Senator Marcus", "Praetor Lucilla", "Informer"],
    },
]


def _mk_chars(names: List[str]) -> List[Dict[str, Any]]:
    return [{"id": n.lower().replace(" ", "_"), "name": n, "role": "unknown", "traits": []} for n in names]


async def _score_quality(hpq: HPQPipeline, text: str) -> float:
    avg, _ = await hpq._evaluate_quality(text)  # type: ignore[attr-defined]
    return float(avg or 0.0)


@dataclass
class RunResult:
    score: float
    elapsed_ms: int
    escalated: bool


async def run_one(hpq_opts_fast: HPQOptions, hpq_opts_full: HPQOptions, scenario: Dict[str, Any]) -> Dict[str, RunResult]:
    # Baseline (fast-only): set threshold_low=0 to avoid escalation
    hpq_fast = HPQPipeline(opts=hpq_opts_fast)
    beat = {"name": scenario.get("title", "Scene"), "purpose": scenario["premise"], "tension": 0.7}
    chars = _mk_chars(scenario["characters"])

    t0 = time.perf_counter()
    out_f = await hpq_fast.craft_scene_hpq(beat, chars)
    t1 = time.perf_counter()
    score_f = await _score_quality(hpq_fast, out_f.get("final") or "")
    res_fast = RunResult(score=score_f, elapsed_ms=int((t1 - t0) * 1000), escalated=False)

    # HPQ full
    hpq_full = HPQPipeline(opts=hpq_opts_full)
    t2 = time.perf_counter()
    out_h = await hpq_full.craft_scene_hpq(beat, chars)
    t3 = time.perf_counter()
    score_h = await _score_quality(hpq_full, out_h.get("final") or "")
    escalated = (out_h.get("used_model") or "") != (out_h.get("model_fast") or out_h.get("used_model"))
    res_hpq = RunResult(score=score_h, elapsed_ms=int((t3 - t2) * 1000), escalated=bool(escalated))

    return {"fast": res_fast, "hpq": res_hpq}


async def main_async(args):
    # Build scenarios
    scenarios: List[Dict[str, Any]] = []
    if args.input:
        with open(args.input) as f:
            for line in f:
                if not line.strip():
                    continue
                scenarios.append(json.loads(line))
    if not scenarios:
        scenarios = DEFAULT_SCENARIOS[: args.count]

    # Options
    fast_opts = HPQOptions(
        candidates=1,
        threshold_avg=0.0,  # disable escalation
        threshold_high=10.0,
        canary_pct=0.0,
        use_structured_scoring=True,
        concurrency=max(1, args.concurrency // 2),
        budget_ms=args.budget_ms,
    )
    hpq_opts = HPQOptions(
        candidates=args.candidates,
        threshold_avg=args.threshold_low,
        threshold_high=args.threshold_high,
        canary_pct=args.canary,
        use_structured_scoring=args.structured_scoring,
        concurrency=args.concurrency,
        budget_ms=args.budget_ms,
    )

    # Run
    results = []
    for sc in scenarios:
        r = await run_one(fast_opts, hpq_opts, sc)
        results.append({"scenario": sc, "fast": r["fast"].__dict__, "hpq": r["hpq"].__dict__})

    # Aggregate
    fast_scores = [r["fast"]["score"] for r in results]
    hpq_scores = [r["hpq"]["score"] for r in results]
    fast_lat = [r["fast"]["elapsed_ms"] for r in results]
    hpq_lat = [r["hpq"]["elapsed_ms"] for r in results]
    hpq_escalated = [1 if r["hpq"]["escalated"] else 0 for r in results]

    summary = {
        "count": len(results),
        "fast": {"avg_score": statistics.mean(fast_scores) if fast_scores else 0.0, "median_ms": int(statistics.median(fast_lat) if fast_lat else 0)},
        "hpq": {"avg_score": statistics.mean(hpq_scores) if hpq_scores else 0.0, "median_ms": int(statistics.median(hpq_lat) if hpq_lat else 0), "escalation_rate": (sum(hpq_escalated) / len(hpq_escalated)) if hpq_escalated else 0.0},
        "win_rate": (sum(1 for f, h in zip(fast_scores, hpq_scores) if h > f) / len(results)) if results else 0.0,
        "details": results,
    }

    out = json.dumps(summary, indent=2)
    if args.report:
        with open(args.report, "w") as f:
            f.write(out)
    print(out)


def build_parser():
    ap = argparse.ArgumentParser(description="HPQ A/B harness")
    ap.add_argument("--input", help="JSONL scenarios file", default=None)
    ap.add_argument("--count", type=int, default=2)
    ap.add_argument("--candidates", type=int, default=2)
    ap.add_argument("--threshold-low", dest="threshold_low", type=float, default=7.5)
    ap.add_argument("--threshold-high", dest="threshold_high", type=float, default=8.3)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--canary", type=float, default=0.0)
    ap.add_argument("--budget-ms", dest="budget_ms", type=int, default=65000)
    ap.add_argument("--structured-scoring", action="store_true")
    ap.add_argument("--report", help="Write summary JSON to path", default=None)
    return ap


def main():
    args = build_parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

