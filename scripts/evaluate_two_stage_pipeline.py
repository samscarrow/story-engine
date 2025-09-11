import asyncio
import argparse
import yaml
import json
import sys
from pathlib import Path

# Add project root to path to allow direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from story_engine.core.orchestration.orchestrator_loader import (
    create_orchestrator_from_yaml,
)
from story_engine.poml.lib.poml_integration import StoryEnginePOMLAdapter

STATE_FILE = Path(__file__).parent / ".last_model_used"


def load_character(character_id: str) -> dict:
    """Loads a character definition from a YAML file."""
    char_path = (
        Path(__file__).resolve().parents[1]
        / f"poml/config/characters/{character_id}.yaml"
    )
    if not char_path.exists():
        raise FileNotFoundError(f"Character file not found: {char_path}")
    with open(char_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


async def evaluate_two_stage_pipeline(model_identifier: str, character_id: str):
    """Runs a two-stage storytelling evaluation."""
    print(
        f"--- Starting Two-Stage Pipeline Evaluation for model: {model_identifier} ---"
    )
    print(f"Character: {character_id}")

    try:
        # --- Setup ---
        character_data = load_character(character_id)
        adapter = StoryEnginePOMLAdapter()
        orchestrator = create_orchestrator_from_yaml("config.yaml")
        print(f"Successfully loaded character: {character_data.get('name')}")

        # --- Execute Two-Stage Pipeline ---
        final_structured_response = await adapter.get_two_stage_character_response(
            character=character_data,
            situation=(
                "You are in your tent, cleaning your gladius. A young, nervous legionary, "
                "barely a man, enters. The legionary, Lucius, informs you that a Germanic "
                "chieftain has been spotted near the camp, alone and seemingly unarmed, "
                "requesting to speak with the commander. This is highly unusual and potentially a trap."
            ),
            orchestrator=orchestrator,
            model_identifier=model_identifier,
        )

        # --- FINAL OUTPUT ---
        print("\n" + "=" * 80)
        print("                          FINAL STRUCTURED RESPONSE (JSON)")
        print("=" * 80 + "\n")

        print(json.dumps(final_structured_response, indent=2))
        print("\n--- Two-Stage Pipeline Evaluation complete ---")

    except FileNotFoundError as e:
        print(f"\n🛑 ERROR: {e}")
    except Exception as e:
        print(f"\n🛑 An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a two-stage storytelling pipeline using the POML framework."
    )
    parser.add_argument(
        "--model", required=True, help="The model identifier to evaluate."
    )
    parser.add_argument(
        "--character", default="centurion", help="The character ID to use."
    )
    args = parser.parse_args()

    asyncio.run(evaluate_two_stage_pipeline(args.model, args.character))
