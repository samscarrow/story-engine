#!/usr/bin/env python3
"""
Select a scene from the scene bank and run a simulation step with the engine.

Usage:
  # List scenes
  python scripts/simulate_from_scene_bank.py --list

  # Simulate from a specific scene id or title slug
  LM_ENDPOINT=http://100.82.243.100:1234 LMSTUDIO_MODEL=google/gemma-3n-e4b STORY_ENGINE_LIVE=1 \
  python scripts/simulate_from_scene_bank.py --id act-one-the-standards-crisis-ext-jerusalem-streets-day
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from story_engine.core.story_engine.scene_bank import SceneBank  # noqa: E402
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


async def simulate_from_scene(
    scene: Dict[str, Any], character_flags: dict | None = None
) -> Dict[str, Any]:
    engine = OrchestratedStoryEngine(use_poml=True, runtime_flags=character_flags)

    # Slightly larger budgets for drama
    profiles = engine.component_profiles
    profiles[StoryComponent.SCENE_DETAILS]["max_tokens"] = max(
        1200, profiles[StoryComponent.SCENE_DETAILS]["max_tokens"]
    )  # noqa: E501
    profiles[StoryComponent.CHARACTER_DIALOGUE]["max_tokens"] = max(
        600, profiles[StoryComponent.CHARACTER_DIALOGUE]["max_tokens"]
    )  # noqa: E501

    req = StoryRequest(
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

    # Seed a scene from the bank as previous context
    plot_point = {
        "name": scene.get("title", "Scene"),
        "purpose": "Use pre-authored scene context",
        "tension": 6,
    }
    characters = req.characters
    previous_context = scene.get("body", "")

    # Expand the scene details with the engine
    sim_scene = await engine.generate_scene(plot_point, characters, previous_context)

    # Dialogue for Pilate
    dialogue = await engine.generate_dialogue(
        sim_scene, characters[0], "Continue the exchange"
    )

    return {
        "seed_scene": scene,
        "simulated_scene": sim_scene,
        "dialogue": dialogue,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Simulate from a scene bank entry")

    # Add standardized model/client arguments
    add_model_client_args(p)

    p.add_argument("--bank", default="scene_bank/pilate_md_scenes.json")
    p.add_argument("--list", action="store_true", help="List available scenes")
    p.add_argument("--id", help="Scene id or title slug to simulate from", default=None)
    p.add_argument("--out", help="Write JSON result to this file", default=None)
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

    # Configure environment for model/client
    if model_config.get("endpoint"):
        os.environ["LM_ENDPOINT"] = model_config["endpoint"]
    if model_config.get("model"):
        os.environ["LMSTUDIO_MODEL"] = model_config["model"]

    # Load DB_* env vars from .env for auto-store
    load_dotenv_keys()
    bank = SceneBank(args.bank)

    if args.list or not args.id:
        for item in bank.list()[:100]:
            print(f"{item['id']}	{item['title']}	{item['act']}")
        if not args.id:
            return

    entry = bank.get(args.id)
    if not entry:
        raise SystemExit(f"Scene not found: {args.id}")

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

    res = asyncio.run(simulate_from_scene(entry.__dict__, cflags or None))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print("Saved:", args.out)
    else:
        print(json.dumps(res, indent=2, ensure_ascii=False))

    # Best-effort store
    if os.getenv("STORE_ALL") == "1" or os.getenv("DB_PASSWORD"):
        try:
            store_workflow_output("scene_bank_simulation", res)
            print("Stored result to DB (scene_bank_simulation)")
        except Exception as e:
            print(f"Store skipped: {e}")


if __name__ == "__main__":
    main()
