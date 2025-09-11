#!/usr/bin/env python3
"""
Choose and set an LLM model via the engine's capability-aware filter.

Usage examples:
  - Print chosen model id (no env changes):
      python scripts/choose_model.py
  - Prefer small models:
      python scripts/choose_model.py --prefer-small
  - Emit a shell export line you can eval in your shell:
      eval "$(python scripts/choose_model.py --prefer-small --export)"
  - Write to a dotenv file:
      python scripts/choose_model.py --prefer-small --write-env .env
"""

from __future__ import annotations

import argparse
import sys
import asyncio

from story_engine.core.story_engine.story_engine_orchestrated import (
    OrchestratedStoryEngine,
)


async def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Choose and set LLM model via engine filter"
    )
    parser.add_argument(
        "--prefer-small", action="store_true", help="Prefer smaller text models"
    )
    parser.add_argument(
        "--export", action="store_true", help="Print export line for LM_MODEL"
    )
    parser.add_argument(
        "--write-env", metavar="PATH", help="Write LM_MODEL into a dotenv-style file"
    )
    args = parser.parse_args()

    engine = OrchestratedStoryEngine(use_poml=False)
    chosen = await engine.choose_and_set_model(prefer_small=args.prefer_small)

    if not chosen:
        print("No viable model found.", file=sys.stderr)
        return 1

    # Default: print the chosen model id
    if not args.export and not args.write_env:
        print(chosen)
        return 0

    if args.export:
        print(f"export LM_MODEL={chosen}")

    if args.write_env:
        try:
            with open(args.write_env, "a", encoding="utf-8") as f:
                f.write(f"\nLM_MODEL={chosen}\n")
        except Exception as e:
            print(f"Failed to write env file: {e}", file=sys.stderr)
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
