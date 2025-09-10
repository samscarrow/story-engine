#!/usr/bin/env python3
"""
Parse the Pontius Pilate screenplay into a scene bank for future simulations.

Usage:
  python scripts/build_scene_bank.py \
    --source pontius-pilate-screenplay/pontius_pilate_screenplay.txt \
    --out scene_bank/pilate_scenes.json
"""

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from story_engine.core.story_engine.scene_bank import parse_screenplay_to_scenes, SceneBank  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Build a scene bank from a screenplay text (txt or md)")
    p.add_argument("--source", default="pontius-pilot-screenplay.md")
    p.add_argument("--out", default="scene_bank/pilate_md_scenes.json")
    args = p.parse_args()

    src_path = Path(args.source)
    if not src_path.exists():
        raise SystemExit(f"Source not found: {src_path}")

    text = src_path.read_text(encoding="utf-8")
    scenes = parse_screenplay_to_scenes(text, source=str(src_path))

    bank = SceneBank(args.out)
    # Merge: update or add; prefer new parsed content
    for s in scenes:
        bank.add(s)
    bank.save()

    print(f"Saved {len(bank.scenes)} scenes to {args.out}")
    # Print a quick listing
    listing = bank.list()[:10]
    for item in listing:
        print(f"- {item['id']} :: {item['title']} :: {item['act']}")


if __name__ == "__main__":
    main()

