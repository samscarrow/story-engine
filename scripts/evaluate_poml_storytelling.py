import asyncio
import argparse
import yaml
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path to allow direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from story_engine.core.orchestration.orchestrator_loader import (
    create_orchestrator_from_yaml,
)
from story_engine.poml.lib.poml_integration import StoryEnginePOMLAdapter
from story_engine.core.common.cli_utils import (
    add_model_client_args,
    get_model_and_client_config,
    print_connection_status,
)

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


def handle_model_change(current_model: str, force_sync: bool = False):
    """Checks for a model change and prompts the user to act."""
    last_model = None
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            last_model = f.read().strip()

    if force_sync:
        print("🔄 --sync flag detected. State will be updated to match current model.")
        return

    if last_model and last_model != current_model:
        print("=" * 60)
        print("                      MODEL CHANGE DETECTED")
        print("-" * 60)
        print(f"Previous model: {last_model}")
        print(f"New model:      {current_model}")
        print("\nACTION REQUIRED:")
        print("1. In LM Studio, unload the current model.")
        print("2. Load the new model.")
        print("3. Ensure the local server is running.")
        print("-" * 60)
        print("PAUSING FOR 5 SECONDS TO ALLOW MANUAL MODEL CHANGE...")
        time.sleep(5)
        print("Continuing with evaluation...")


def update_last_model(model: str):
    """Updates the state file with the last used model."""
    with open(STATE_FILE, "w") as f:
        f.write(model)


async def evaluate_poml_model(
    model_identifier: str,
    character_id: str,
    emphasis: str,
    sync_mode: bool,
    model_config: dict = None,
):
    """Runs a storytelling evaluation using the POML framework."""

    handle_model_change(model_identifier, force_sync=sync_mode)

    print(f"--- Starting POML evaluation for model: {model_identifier} ---")
    print(f"Character: {character_id}, Emphasis: {emphasis}")

    # Configure environment for model/client if provided
    if model_config:
        if model_config.get("endpoint"):
            os.environ["LM_ENDPOINT"] = model_config["endpoint"]
        if model_config.get("model"):
            os.environ["LMSTUDIO_MODEL"] = model_config["model"]

    try:
        # 1. Load Character Data
        character_data = load_character(character_id)
        print(f"Successfully loaded character: {character_data.get('name')}")

        # 2. Initialize POML Adapter and Orchestrator
        adapter = StoryEnginePOMLAdapter()
        orchestrator = create_orchestrator_from_yaml("config.yaml")

        # 3. Define the Scenario
        situation = (
            "Marcus is in his tent, cleaning his gladius. A young, nervous legionary, "
            "barely a man, enters. The legionary, Lucius, informs Marcus that a Germanic "
            "chieftain has been spotted near the camp, alone and seemingly unarmed, "
            "requesting to speak with the commander. This is highly unusual and potentially a trap."
        )

        # 4. Get the rendered prompt from the POML template
        prompt = adapter.get_character_prompt(character_data, situation, emphasis)
        print("\n--- Rendered POML Prompt (Truncated) ---")
        print(prompt[:1000] + "...")
        print("-----------------------------------------")

        # 5. Generate the response using the orchestrator
        print("\nSending prompt to the model...")
        response = await orchestrator.generate(
            prompt,
            system="You are an expert character simulation AI.",
            temperature=0.7,
            max_tokens=1000,
            model=model_identifier,
            stop=["<|EOR|>"],
            response_format={"type": "json_object"},
        )

        # 6. Print and validate the structured response
        print("\n" + "=" * 80)
        print("                          MODEL RESPONSE (JSON)")
        print("=" * 80 + "\n")

        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:-3].strip()
        elif response_text.startswith("```"):
            response_text = response_text[3:-3].strip()

        try:
            parsed_response = json.loads(response_text)
            print(json.dumps(parsed_response, indent=2))
            # If successful, update the state file
            update_last_model(model_identifier)
            print("\n--- POML Evaluation complete ---")
        except json.JSONDecodeError:
            print("ERROR: Model did not return valid JSON even after cleaning.")
            print("--- Raw Model Output ---")
            print(response.text)

        print("\n" + "=" * 80)

    except FileNotFoundError as e:
        print(f"\n🛑 ERROR: {e}")
    except Exception as e:
        print(f"\n🛑 An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a storytelling model using the POML framework."
    )

    # Add standardized model/client arguments
    add_model_client_args(parser)

    parser.add_argument(
        "--character", default="centurion", help="The character ID to use."
    )
    parser.add_argument(
        "--emphasis", default="pragmatic", help="The emphasis mode for the response."
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Force update of the script's last-used model state.",
    )

    args = parser.parse_args()

    # Get model/client configuration with auto-detection if --model not specified
    model_config = get_model_and_client_config(args)
    print_connection_status(model_config)

    # Use the detected/specified model
    model_to_use = model_config["model"]

    asyncio.run(
        evaluate_poml_model(
            model_to_use, args.character, args.emphasis, args.sync, model_config
        )
    )
