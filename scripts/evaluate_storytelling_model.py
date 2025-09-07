import openai
import os
import argparse
import sys
from pathlib import Path

# Add project root to path for cli_utils
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.common.cli_utils import add_model_client_args, get_model_and_client_config, print_connection_status

# --- Configuration ---
# Ensure your LM Studio server is running.
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
API_KEY = "lm-studio"

# --- Prompt Definition ---
STORY_PROMPT = """
You are a world-class storytelling AI. Your task is to simulate a character in a given scenario, writing from their perspective.

**Character:** Marcus, a grizzled Roman Centurion stationed at the edge of the empire in Germania. He is weary of war, cynical, but still holds a flicker of loyalty to his men and to the idea of Rome. He has a dry wit and a pragmatic outlook.

**Scenario:** Marcus is in his tent, cleaning his gladius. A young, nervous legionary, barely a man, enters. The legionary, Lucius, informs Marcus that a Germanic chieftain has been spotted near the camp, alone and seemingly unarmed, requesting to speak with the commander. This is highly unusual and potentially a trap.

**Your Task:** Write the scene from Marcus's perspective. Reveal his thoughts, what he says, and what he does. Capture his personality and his immediate reaction to this strange news.
"""

def evaluate_model(model_identifier, endpoint_url=None):
    """
    Connects to LM Studio, sends the storytelling prompt, and prints the response.
    """
    print(f"--- Starting evaluation for model: {model_identifier} ---")
    
    # Use provided endpoint or default
    base_url = endpoint_url or LM_STUDIO_BASE_URL
    if not base_url.endswith('/v1'):
        base_url = f"{base_url.rstrip('/')}/v1"

    # Initialize the OpenAI client to connect to LM Studio
    try:
        client = openai.OpenAI(base_url=base_url, api_key=API_KEY)
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        print("Please ensure the 'openai' library is installed (`pip install openai`).")
        return

    print("Sending prompt to the model...")

    try:
        # Create the chat completion request
        completion = client.chat.completions.create(
            model=model_identifier,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant specialized in storytelling and character simulation."
                },
                {
                    "role": "user",
                    "content": STORY_PROMPT
                }
            ],
            temperature=0.7,  # A balanced value for creative but coherent output
            max_tokens=1000,
        )

        # --- Output ---
        print("\n" + "="*80)
        print("                          MODEL RESPONSE")
        print("="*80 + "\n")
        print(completion.choices[0].message.content)
        print("\n" + "="*80)
        print("--- Evaluation complete ---")

    except openai.APIConnectionError as e:
        print("\n" + "="*80)
        print("🛑 ERROR: Could not connect to LM Studio.")
        print(f"   Please ensure LM Studio is running and the server is active at {LM_STUDIO_BASE_URL}.")
        print(f"   Details: {e.__cause__}")
        print("="*80)
    except openai.NotFoundError as e:
        print("\n" + "="*80)
        print("🛑 ERROR: Model not found.")
        print(f"   The model identifier '{model_identifier}' may be incorrect or not loaded.")
        print("   Please verify the model name in LM Studio.")
        print(f"   Details: {e}")
        print("="*80)
    except Exception as e:
        print("\n" + "="*80)
        print(f"🛑 An unexpected error occurred: {e}")
        print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a storytelling model in LM Studio.")
    
    # Add standardized model/client arguments
    add_model_client_args(parser)
    
    args = parser.parse_args()
    
    # Get model/client configuration with auto-detection
    model_config = get_model_and_client_config(args)
    print_connection_status(model_config)

    # Set the console to handle UTF-8 output, useful for special characters
    if os.name == 'nt':
        try:
            import _winapi
            _winapi.SetConsoleOutputCP(65001)
        except (ImportError, AttributeError):
            print("Warning: Could not set console to UTF-8. Some characters may not display correctly.")
    
    evaluate_model(model_config["model"], model_config["endpoint"])