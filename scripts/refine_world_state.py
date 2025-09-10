#!/usr/bin/env python3
"""
Refine the latest world state using LMStudio via the orchestrator and persist the result.

Usage:
  python scripts/refine_world_state.py --characters pontius_pilate --location Praetorium --workflow world_refined

Ensure config.yaml orchestrator is set to LMStudio (endpoint http://localhost:1234) or otherwise reachable.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root to path to allow direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from story_engine.core.character_engine.meta_narrative_pipeline import MetaNarrativePipeline
from story_engine.core.story_engine.world_state_refiner import WorldStateRefiner
from story_engine.core.common.cli_utils import add_model_client_args, get_model_and_client_config, print_connection_status


async def main():
    parser = argparse.ArgumentParser(description='Refine world state via LMStudio and save the result')
    
    # Add standardized model/client arguments
    add_model_client_args(parser)
    
    parser.add_argument('--characters', nargs='*', default=[], help='Focus characters (IDs). Optional')
    parser.add_argument('--location', default='', help='Focus location. Optional')
    parser.add_argument('--last-n-events', type=int, default=8, help='How many recent timeline events to consider')
    parser.add_argument('--workflow', default='world_state_refined', help='Workflow name to store refined world state')
    parser.add_argument('--mode', default='json_patch', choices=['json_patch', 'poml_enhance'], help='Refinement mode')
    parser.add_argument('--system-prompt-file', default='', help='Optional path to a system prompt file (for poml_enhance)')
    parser.add_argument('--output-poml', default='refined_world_state.poml', help='Output path for enhanced POML (poml_enhance mode)')
    
    args = parser.parse_args()
    
    # Get model/client configuration with auto-detection
    model_config = get_model_and_client_config(args)
    print_connection_status(model_config)
    
    # Configure environment for model/client
    if model_config.get("endpoint"):
        os.environ["LM_ENDPOINT"] = model_config["endpoint"]
    if model_config.get("model"):
        os.environ["LMSTUDIO_MODEL"] = model_config["model"]

    # Reuse existing pipeline infra to get orchestrator + POML adapter + config
    pipeline = MetaNarrativePipeline(use_poml=True)
    refiner = WorldStateRefiner(pipeline.orchestrator, pipeline.poml, manager=pipeline.world_manager)
    # Load system prompt if provided
    system_prompt = ''
    if args.system_prompt_file:
        try:
            with open(args.system_prompt_file, 'r', encoding='utf-8') as f:
                system_prompt = f.read()
        except Exception:
            system_prompt = ''

    refined = await refiner.refine(
        focus_characters=args.characters or None,
        location=(args.location or None),
        last_n_events=args.last_n_events,
        workflow_name=args.workflow,
        mode=args.mode,
        system_prompt=(system_prompt or None),
        output_poml_path=(args.output_poml if args.mode == 'poml_enhance' else None),
    )
    print(json.dumps({
        'status': 'ok',
        'mode': args.mode,
        'saved_workflow': args.workflow,
        'facts': len(refined.facts),
        'timeline': len(refined.timeline),
        'poml_output': (args.output_poml if args.mode == 'poml_enhance' else None)
    }, indent=2))


if __name__ == '__main__':
    asyncio.run(main())

