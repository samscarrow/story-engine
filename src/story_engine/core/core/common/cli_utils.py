"""
Common CLI utilities for Story Engine scripts.
Provides standardized argument parsing for model and client parameters.
"""

import argparse
import os
import requests
from typing import Optional, Dict, Any
from story_engine.core.orchestration.model_filters import filter_models, choose_first_id


def add_model_client_args(parser: argparse.ArgumentParser) -> None:
    """
    Add standardized --model and --client arguments to an argument parser.

    Args:
        parser: The argparse ArgumentParser to add arguments to
    """
    parser.add_argument(
        "--client",
        default="lmstudio",
        choices=["lmstudio", "koboldcpp", "openai", "anthropic"],
        help="LLM client type to use (default: lmstudio)",
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Model identifier to use. If not specified, uses loaded model in LMStudio or falls back to gemma-2-27b",
    )

    parser.add_argument(
        "--endpoint",
        default=None,
        help="LLM endpoint URL (default: http://localhost:1234 for lmstudio, http://localhost:5001 for koboldcpp)",
    )


def get_model_and_client_config(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Get standardized model and client configuration from parsed arguments.
    Handles auto-detection of loaded models in LMStudio.

    Args:
        args: Parsed command line arguments from argparse

    Returns:
        Dictionary with client, model, and endpoint configuration
    """

    # Set default endpoint based on client
    if args.endpoint is None:
        if args.client == "lmstudio":
            args.endpoint = os.environ.get("LM_ENDPOINT", "http://localhost:1234")
        elif args.client == "koboldcpp":
            args.endpoint = os.environ.get("KOBOLD_ENDPOINT", "http://localhost:5001")
        else:
            args.endpoint = "http://localhost:1234"  # Default fallback

    # Auto-detect model if not specified
    if args.model is None:
        detected_model = detect_loaded_model(args.client, args.endpoint)
        if detected_model:
            args.model = detected_model
            print(f"🎯 Auto-detected loaded model: {detected_model}")
        else:
            # Use environment variable or fallback
            args.model = os.environ.get("LMSTUDIO_MODEL", "gemma-2-27b")
            print(f"⚡ No model detected, using fallback: {args.model}")
    else:
        print(f"🔧 Using specified model: {args.model}")

    config = {
        "client": args.client,
        "model": args.model,
        "endpoint": args.endpoint,
        "provider": args.client,  # For compatibility with existing orchestrator
    }

    print(f"📡 Client: {config['client']}")
    print(f"🤖 Model: {config['model']}")
    print(f"🌐 Endpoint: {config['endpoint']}")

    return config


def detect_loaded_model(client: str, endpoint: str) -> Optional[str]:
    """
    Attempt to detect the currently loaded model from the LLM server.

    Args:
        client: The client type (lmstudio, koboldcpp, etc.)
        endpoint: The endpoint URL

    Returns:
        The loaded model identifier if detected, None otherwise
    """

    try:
        if client == "lmstudio":
            return detect_lmstudio_model(endpoint)
        elif client == "koboldcpp":
            return detect_koboldcpp_model(endpoint)
        else:
            return None
    except Exception as e:
        print(f"⚠️  Model detection failed: {e}")
        return None


def detect_lmstudio_model(endpoint: str) -> Optional[str]:
    """
    Detect loaded model in LMStudio via API.

    Args:
        endpoint: LMStudio endpoint URL

    Returns:
        Loaded model identifier or None
    """

    try:
        # Try the /v1/models endpoint
        models_url = f"{endpoint.rstrip('/')}/v1/models"
        response = requests.get(models_url, timeout=5)

        if response.status_code == 200:
            data = response.json()
            models = data.get("data", [])
            # Filter to text-gen and optionally prefer small models
            prefer_small = str(
                os.environ.get("LM_PREFER_SMALL", "")
            ).strip().lower() in {"1", "true", "yes", "on"}
            filtered = filter_models(
                models, require_text=True, prefer_small=prefer_small
            )
            choice = choose_first_id(filtered)
            if choice:
                return choice

        # Fallback: try a simple completion to see if model is loaded
        completion_url = f"{endpoint.rstrip('/')}/v1/chat/completions"
        test_payload = {
            "model": "any",  # LMStudio usually ignores this if model is loaded
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 1,
        }

        response = requests.post(completion_url, json=test_payload, timeout=5)
        if response.status_code == 200:
            # Model is responsive, but we can't detect the exact name
            return "loaded-model"

    except Exception:
        pass

    return None


def detect_koboldcpp_model(endpoint: str) -> Optional[str]:
    """
    Detect loaded model in KoboldCpp via API.

    Args:
        endpoint: KoboldCpp endpoint URL

    Returns:
        Loaded model identifier or None
    """

    try:
        # Try the /api/v1/model endpoint
        model_url = f"{endpoint.rstrip('/')}/api/v1/model"
        response = requests.get(model_url, timeout=5)

        if response.status_code == 200:
            data = response.json()
            model_name = data.get("result", "")
            if model_name:
                return model_name

        # Fallback: try status endpoint
        status_url = f"{endpoint.rstrip('/')}/api/v1/info/version"
        response = requests.get(status_url, timeout=5)

        if response.status_code == 200:
            return "kobold-model"  # Generic identifier

    except Exception:
        pass

    return None


def validate_model_connection(client: str, model: str, endpoint: str) -> bool:
    """
    Validate that we can connect to the specified model and endpoint.

    Args:
        client: Client type
        model: Model identifier
        endpoint: Endpoint URL

    Returns:
        True if connection is valid, False otherwise
    """

    try:
        if client == "lmstudio":
            return validate_lmstudio_connection(model, endpoint)
        elif client == "koboldcpp":
            return validate_koboldcpp_connection(model, endpoint)
        else:
            return False
    except Exception:
        return False


def validate_lmstudio_connection(model: str, endpoint: str) -> bool:
    """Validate LMStudio connection"""

    try:
        url = f"{endpoint.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 1,
        }

        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 200

    except Exception:
        return False


def validate_koboldcpp_connection(model: str, endpoint: str) -> bool:
    """Validate KoboldCpp connection"""

    try:
        url = f"{endpoint.rstrip('/')}/api/v1/generate"
        payload = {"prompt": "test", "max_length": 1}

        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 200

    except Exception:
        return False


def print_connection_status(config: Dict[str, Any]) -> None:
    """
    Print connection status information for debugging.

    Args:
        config: Configuration dictionary from get_model_and_client_config
    """

    print("\n🔍 CONNECTION STATUS:")
    print(f"   Client: {config['client']}")
    print(f"   Model: {config['model']}")
    print(f"   Endpoint: {config['endpoint']}")

    # Test connection
    is_valid = validate_model_connection(
        config["client"], config["model"], config["endpoint"]
    )

    if is_valid:
        print("   Status: ✅ Connection validated")
    else:
        print("   Status: ❌ Connection failed - check if server is running")

        if config["client"] == "lmstudio":
            print("   💡 Start LMStudio and load a model, then try again")
        elif config["client"] == "koboldcpp":
            print(
                "   💡 Start KoboldCpp with: python koboldcpp.py --model your_model.gguf"
            )


# Convenience function for quick setup
def setup_standard_args() -> argparse.ArgumentParser:
    """
    Create a new ArgumentParser with standard model/client arguments pre-added.

    Returns:
        ArgumentParser with --model, --client, and --endpoint arguments
    """
    parser = argparse.ArgumentParser()
    add_model_client_args(parser)
    return parser
