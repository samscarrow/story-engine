#!/usr/bin/env python3
"""
Run a live Pontius Pilate simulation via OrchestratedStoryEngine using POML.

Env overrides:
  LM_ENDPOINT:       LM Studio endpoint (e.g., http://localhost:1234)
  LMSTUDIO_MODEL:    Model id (e.g., google/gemma-3n-e4b)
  STORY_ENGINE_LIVE: Set to 1 to enable live tests (skips health failure)

Usage:
  LM_ENDPOINT=http://100.82.243.100:1234 LMSTUDIO_MODEL=google/gemma-3n-e4b \
  STORY_ENGINE_LIVE=1 \
  python scripts/run_pilate_simulation.py
"""

import asyncio
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from story_engine.core.story_engine.story_engine_orchestrated import (
    OrchestratedStoryEngine,
    StoryComponent,
)  # noqa: E402
from story_engine.core.domain.models import StoryRequest  # noqa: E402
from story_engine.core.common.result_store import store_workflow_output  # noqa: E402
from story_engine.core.common.dotenv_loader import load_dotenv_keys  # noqa: E402
from story_engine.core.common.cli_utils import (
    add_model_client_args,
    get_model_and_client_config,
    print_connection_status,
)  # noqa: E402


async def run_sim(
    character_flags: dict | None = None, model_config: dict | None = None
):
    # Load DB_* so that auto-store can work without manual export
    load_dotenv_keys()

    # Configure environment for model/client if provided
    if model_config:
        if model_config.get("endpoint"):
            os.environ["LM_ENDPOINT"] = model_config["endpoint"]
        if model_config.get("model"):
            os.environ["LMSTUDIO_MODEL"] = model_config["model"]

    engine = OrchestratedStoryEngine(use_poml=True, runtime_flags=character_flags)

    # Trim defaults to keep the run snappy
    profiles = engine.component_profiles
    profiles[StoryComponent.PLOT_STRUCTURE]["max_tokens"] = min(
        400, profiles[StoryComponent.PLOT_STRUCTURE]["max_tokens"]
    )  # noqa: E501
    profiles[StoryComponent.SCENE_DETAILS]["max_tokens"] = min(
        400, profiles[StoryComponent.SCENE_DETAILS]["max_tokens"]
    )  # noqa: E501
    profiles[StoryComponent.CHARACTER_DIALOGUE]["max_tokens"] = min(
        200, profiles[StoryComponent.CHARACTER_DIALOGUE]["max_tokens"]
    )  # noqa: E501
    profiles[StoryComponent.QUALITY_EVALUATION]["max_tokens"] = min(
        300, profiles[StoryComponent.QUALITY_EVALUATION]["max_tokens"]
    )  # noqa: E501
    profiles[StoryComponent.ENHANCEMENT]["max_tokens"] = min(
        300, profiles[StoryComponent.ENHANCEMENT]["max_tokens"]
    )  # noqa: E501

    request = StoryRequest(
        title="The Trial Before Dawn",
        premise=(
            "A Roman governor must choose between political stability and his sense of justice "
            "when a controversial prophet is brought before him."
        ),
        genre="Historical Drama",
        tone="Grave",
        characters=[
            {"id": "pilate", "name": "Pontius Pilate", "role": "conflicted judge"},
            {"id": "caiaphas", "name": "Caiaphas", "role": "antagonist"},
        ],
        setting="Jerusalem",
        structure="three_act",
    )

    # Health check providers; skip if none healthy and not explicitly live
    health = await engine.orchestrator.health_check_all()
    if not any(v.get("healthy") for v in health.values()):
        raise SystemExit(f"No healthy providers: {health}")

    # Plot structure
    plot = await engine.generate_plot_structure(request)
    beats = plot.get("beats") or []
    if not beats:
        raise RuntimeError("No beats generated from plot structure")

    # First scene
    scene = await engine.generate_scene(beats[0], request.characters)

    # Dialogue for Pilate
    dialogue = await engine.generate_dialogue(
        scene, request.characters[0], "Opening line"
    )

    # Evaluation and enhancement
    evaluation = await engine.evaluate_quality(scene["scene_description"])
    enhanced = await engine.enhance_content(
        scene["scene_description"], evaluation, "pacing and emotion"
    )

    result = {
        "plot": plot,
        "scene": scene,
        "dialogue": dialogue,
        "evaluation": evaluation,
        "enhanced": enhanced,
    }

    # Save and pretty print
    out_path = Path("feedback").mkdir(exist_ok=True)
    out_file = Path("feedback").joinpath("pilate_simulation_result.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("Saved:", out_file)
    print("\n--- Scene (excerpt) ---\n")
    print(scene["scene_description"][:800])
    print("\n--- Dialogue ---\n")
    print(dialogue[:400])
    print("\n--- Evaluation (scores) ---\n")
    print(json.dumps(evaluation.get("scores", {}), indent=2))

    # Best-effort store
    if os.getenv("STORE_ALL") == "1" or os.getenv("DB_PASSWORD"):
        try:
            store_workflow_output("pilate_simulation", result)
            print("Stored result to DB (pilate_simulation)")
        except Exception as e:
            print(f"Store skipped: {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run a Pilate demo simulation")

    # Add standardized model/client arguments
    add_model_client_args(p)

    p.add_argument(
        "--char-flag",
        action="append",
        default=[],
        help="Per-character runtime flags id:key=value; repeatable",
    )

    args = p.parse_args()

    # Get model/client configuration
    model_config = get_model_and_client_config(args)
    print_connection_status(model_config)

    # Parse character flags
    cflags: dict[str, dict] = {}
    for spec in args.char_flag or []:
        try:
            if ":" not in spec or "=" not in spec:
                continue
            ident, kv = spec.split(":", 1)
            key, val = kv.split("=", 1)
            ident = ident.strip().lower().replace(" ", "_")
            cflags.setdefault(ident, {})[key.strip()] = val.strip()
        except Exception:
            continue

    asyncio.run(run_sim(cflags or None, model_config))
