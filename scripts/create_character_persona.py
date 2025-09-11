import asyncio
import argparse
import yaml
import sys
from pathlib import Path

# Add project root to path to allow direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from story_engine.core.orchestration.orchestrator_loader import (
    create_orchestrator_from_yaml,
)
from story_engine.poml.lib.poml_integration import StoryEnginePOMLAdapter


async def create_character_persona(
    model_identifier: str, concept: str, character_id: str
):
    """Runs a two-stage character creation pipeline."""
    print(f"--- Starting Two-Stage Character Creation for concept: {concept} ---")
    print(f"Model: {model_identifier}, Character ID: {character_id}")

    try:
        # --- Setup ---
        adapter = StoryEnginePOMLAdapter()
        orchestrator = create_orchestrator_from_yaml("config.yaml")

        # --- STAGE 1: Free-Form Persona Generation ---
        print("\n--- Stage 1: Generating Free-Form Biography ---")

        freeform_system_prompt = adapter.engine.render_roles(
            "characters/persona_generation_freeform.poml", {"concept": concept}
        )

        freeform_response = await orchestrator.generate(
            prompt="",
            system=freeform_system_prompt["system"],
            model=model_identifier,
            temperature=0.7,
            max_tokens=1500,
        )
        freeform_biography_text = freeform_response.text

        print("\n" + "-" * 25 + " STAGE 1 OUTPUT (Free-Form Biography) " + "-" * 26)
        print(freeform_biography_text)
        print("-" * 70)

        # --- STAGE 2: Structuring the Biography ---
        print("\n--- Stage 2: Structuring the Biography into YAML ---")

        structuring_system_prompt = adapter.engine.render_roles(
            "characters/persona_generation_structured.poml", {}
        )

        structured_response = await orchestrator.generate(
            prompt=freeform_biography_text,
            system=structuring_system_prompt["system"],
            model=model_identifier,
            temperature=0.1,
            max_tokens=2000,
            response_format={
                "type": "json_object"
            },  # Assuming YAML can be represented as JSON
        )

        # --- FINAL OUTPUT ---
        print("\n" + "=" * 80)
        print("                          FINAL STRUCTURED PERSONA (YAML)")
        print("=" * 80 + "\n")

        # The response text should be a YAML string, so we parse it
        try:
            # The model will likely return a JSON representation of the YAML
            parsed_yaml = yaml.safe_load(structured_response.text)
            # Convert to YAML string for display and saving
            yaml_output = yaml.dump(parsed_yaml, sort_keys=False)
            print(yaml_output)

            # Save the final YAML file
            output_path = (
                Path(__file__).resolve().parents[1]
                / f"poml/config/characters/{character_id}.yaml"
            )
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(yaml_output)
            print(f"\nSuccessfully saved persona to: {output_path}")

            print("\n--- Two-Stage Character Creation complete ---")
        except yaml.YAMLError as e:
            print(f"ERROR: Model did not return valid YAML/JSON.\n{e}")
            print("--- Raw Model Output ---")
            print(structured_response.text)

        print("\n" + "=" * 80)

    except Exception as e:
        print(f"\n🛑 An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a character persona using a two-stage pipeline."
    )
    parser.add_argument("--model", required=True, help="The model identifier to use.")
    parser.add_argument(
        "--concept", required=True, help="A brief concept for the character."
    )
    parser.add_argument(
        "--id",
        required=True,
        help="The character ID to use for the output file (e.g., 'caiaphas').",
    )
    args = parser.parse_args()

    asyncio.run(create_character_persona(args.model, args.concept, args.id))
