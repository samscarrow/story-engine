#!/usr/bin/env python3
"""
Run meta-narrative pipeline:
  simulations (multi) → reviewer throughlines → select best → synthesize meta → screenplay draft

Usage (LM Studio live):
  LM_ENDPOINT=http://100.82.243.100:1234 LMSTUDIO_MODEL=google/gemma-3n-e4b STORY_ENGINE_LIVE=1 \
  python scripts/run_meta_pipeline.py --character pilate --runs 2 \
    --situations "Crowd demands decision" "Private counsel with Claudia"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from core.character_engine.meta_narrative_pipeline import MetaNarrativePipeline  # noqa: E402
from core.common.dotenv_loader import load_dotenv_keys  # noqa: E402


def load_character(name: str) -> dict:
    # Minimal pilate default
    if name.lower() in ("pilate", "pontius", "pontius_pilate"):
        return {
            "id": "pontius_pilate",
            "name": "Pontius Pilate",
            "backstory": {"origin": "Roman equestrian from Samnium", "career": "Prefect of Judaea"},
            "traits": ["pragmatic", "ambitious", "anxious"],
            "values": ["order", "duty", "Roman law"],
            "fears": ["rebellion", "imperial disfavor"],
            "desires": ["peace", "advancement"],
            "emotional_state": {"anger": 0.4, "doubt": 0.6, "fear": 0.5, "compassion": 0.3, "confidence": 0.5},
            "memory": {"recent_events": ["Wife's warning dream", "Crowd agitation"]},
            "current_goal": "Maintain order without rebellion",
            "internal_conflict": "Duty to Rome vs. sense of justice",
        }
    raise SystemExit(f"Unknown character preset: {name}")


async def run(character_name: str, situations: List[str], runs: int, out: str, store_db: bool = False, workflow_name: str = "meta_pipeline") -> None:
    # Load DB_* from .env so auto-store works without manual export
    load_dotenv_keys()
    pipeline = MetaNarrativePipeline(use_poml=True)
    char_dict = load_character(character_name)
    char_state = pipeline.character_from_dict(char_dict)

    # Round 1: Baseline simulations and review
    sims_round1 = await pipeline.simulate(char_state, situations, runs_per=runs)
    review_round1 = await pipeline.review_throughlines(char_dict, situations, sims_round1)
    throughlines1 = review_round1.get("throughlines", [])
    recommended = review_round1.get("recommended_params", {}) or {}

    # Apply reviewer params (if any) and run Round 2
    sims_round2 = []
    review_round2 = {"throughlines": []}
    if recommended:
        pipeline.apply_reviewer_params(recommended)
        sims_round2 = await pipeline.simulate(char_state, situations, runs_per=runs)
        review_round2 = await pipeline.review_throughlines(char_dict, situations, sims_round1 + sims_round2)
        throughlines2 = review_round2.get("throughlines", [])
        throughlines_final = throughlines2 or throughlines1
    else:
        throughlines_final = throughlines1

    best = pipeline.select_best_throughline(throughlines_final)
    if not best:
        raise SystemExit("No throughlines produced")
    meta_outline = await pipeline.synthesize_meta(char_dict, best)
    evaluation_meta = await pipeline.evaluate_meta(meta_outline)
    enhanced_meta = await pipeline.enhance_meta(meta_outline, evaluation_meta)
    draft = await pipeline.draft_screenplay(enhanced_meta or meta_outline)

    result = {
        "character": char_dict,
        "situations": situations,
        "simulations_round1": sims_round1,
        "review_round1": review_round1,
        "simulations_round2": sims_round2,
        "review_round2": review_round2,
        "selected_throughline": best,
        "meta_outline": meta_outline,
        "meta_evaluation": evaluation_meta,
        "meta_enhanced": enhanced_meta,
        "screenplay_draft": draft,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved:", out)

    if store_db or os.getenv("STORE_ALL") == "1" or os.getenv("DB_PASSWORD"):
        try:
            from core.storage import get_database_connection
            db = get_database_connection(
                db_type="postgresql",
                db_name=os.getenv("DB_NAME", "story_db"),
                user=os.getenv("DB_USER", "story"),
                password=os.getenv("DB_PASSWORD"),
                host=os.getenv("DB_HOST", "localhost"),
                port=int(os.getenv("DB_PORT", "5432")),
                sslmode=os.getenv("DB_SSLMODE"),
                sslrootcert=os.getenv("DB_SSLROOTCERT"),
                sslcert=os.getenv("DB_SSLCERT"),
                sslkey=os.getenv("DB_SSLKEY"),
            )
            db.connect()
            db.store_output(workflow_name, result)
            print(f"Stored result in DB under workflow '{workflow_name}'")
        except Exception as e:
            print(f"DB store failed: {e}")
        finally:
            try:
                if 'db' in locals() and getattr(db, 'conn', None):
                    db.disconnect()
                    print("DB connection closed.")
            except Exception:
                pass


def main() -> None:
    p = argparse.ArgumentParser(description="Run meta-narrative pipeline over character simulations")
    p.add_argument("--character", default="pilate")
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--situations", nargs="+", required=True)
    p.add_argument("--out", default="feedback/meta_pipeline_result.json")
    p.add_argument("--store-db", action="store_true", help="Store the final result JSON into PostgreSQL using DB_* env vars")
    p.add_argument("--workflow-name", default="meta_pipeline", help="Workflow name key under which to store the output")
    args = p.parse_args()

    asyncio.get_event_loop().run_until_complete(
        run(
            args.character,
            args.situations,
            args.runs,
            args.out,
            store_db=args.store_db,
            workflow_name=args.workflow_name,
        )
    )


if __name__ == "__main__":
    main()
