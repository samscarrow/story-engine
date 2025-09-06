#!/usr/bin/env python3
"""
Run an iterative persona-constrained simulation and persist results to the DB.

Uses MetaNarrativePipeline orchestration and the SimulationEngine iterative loop.
Stores final, history, and reviews JSON into the storage backend.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict

import yaml

from core.character_engine.meta_narrative_pipeline import MetaNarrativePipeline
from core.storage import get_database_connection


def load_persona_yaml(char_id: str) -> Dict[str, Any]:
    base = Path(__file__).resolve().parents[1] / 'poml' / 'config' / 'characters' / f'{char_id}.yaml'
    if not base.exists():
        raise FileNotFoundError(f"Persona YAML not found for id='{char_id}': {base}")
    with open(base, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_db():
    """Build a DB connection from environment variables; defaults to PostgreSQL if DB_USER present, otherwise SQLite."""
    db_user = os.getenv('DB_USER')
    if db_user:
        # Prefer PostgreSQL
        db_name = os.getenv('DB_NAME', 'story_db')
        db_password = os.getenv('DB_PASSWORD')
        db_host = os.getenv('DB_HOST', '127.0.0.1')
        db_port = int(os.getenv('DB_PORT', '5432'))
        # Optional SSL
        sslmode = os.getenv('DB_SSLMODE')
        sslrootcert = os.getenv('DB_SSLROOTCERT')
        sslcert = os.getenv('DB_SSLCERT')
        sslkey = os.getenv('DB_SSLKEY')
        return get_database_connection(
            'postgresql',
            db_name=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port,
            sslmode=sslmode,
            sslrootcert=sslrootcert,
            sslcert=sslcert,
            sslkey=sslkey,
        )
    # Fallback to SQLite local file inside repo
    return get_database_connection('sqlite', db_name=str(Path('workflow_outputs.db')))


async def main():
    parser = argparse.ArgumentParser(description='Run iterative persona simulation and store results')
    parser.add_argument('--character-id', default='pontius_pilate', help='Character ID matching a persona YAML filename')
    parser.add_argument('--situation', required=False, default='The crowd demands a decision; tensions are high in the Praetorium.', help='Situation prompt')
    parser.add_argument('--emphasis', default='duty', help='Emphasis mode')
    parser.add_argument('--iterations', type=int, default=3, help='Number of iterative attempts')
    parser.add_argument('--window', type=int, default=3, help='Number of previous attempts to include')
    parser.add_argument('--workflow-name', default='iterative_persona_sim', help='Storage workflow name')
    args = parser.parse_args()

    # Load persona
    persona = load_persona_yaml(args.character_id)

    # Construct pipeline (uses config.yaml and orchestrator)
    pipeline = MetaNarrativePipeline(use_poml=True, use_iterative_persona=True)
    character = pipeline.character_from_dict(persona)

    # Run iterative simulation directly to capture full history + reviews
    result = await pipeline.engine.run_iterative_simulation(
        character,
        args.situation,
        emphasis=args.emphasis,
        iterations=args.iterations,
        window=args.window,
    )

    # Prepare record for storage
    record = {
        'character_id': character.id,
        'character_name': character.name,
        'situation': args.situation,
        'emphasis': args.emphasis,
        'iterations': args.iterations,
        'window': args.window,
        'result': result,
    }

    # Store in DB
    db = build_db()
    db.connect()
    try:
        db.store_output(args.workflow_name, record)
        print(json.dumps({'status': 'ok', 'stored': True, 'workflow': args.workflow_name}, indent=2))
    finally:
        db.disconnect()


if __name__ == '__main__':
    asyncio.run(main())

