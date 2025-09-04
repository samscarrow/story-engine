#!/usr/bin/env python3
"""
Evaluate the quality of story content using the POML evaluation template.

Usage examples:

  # Evaluate inline text (non-live uses defaults; live requires env vars)
  python scripts/evaluate_poml.py --text "Pilate weighed justice against order in Jerusalem."

  # Evaluate from files and write JSONL
  LM_ENDPOINT=http://100.82.243.100:1234 LMSTUDIO_MODEL=google/gemma-3n-e4b \
  STORY_ENGINE_LIVE=1 \
  python scripts/evaluate_poml.py --file samples/*.txt --out results.jsonl

Environment:
  LM_ENDPOINT       - LM Studio endpoint (e.g., http://localhost:1234)
  LMSTUDIO_MODEL    - Model id (e.g., google/gemma-3n-e4b)
  STORY_ENGINE_LIVE - If set to 1, run live against provider
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import glob
import hashlib
import json
import os
import sys
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from core.story_engine.story_engine_orchestrated import OrchestratedStoryEngine  # noqa: E402
from core.common.result_store import store_workflow_output  # noqa: E402


def _read_inputs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    if args.text:
        items.append({"source": "inline", "content": args.text})

    for pattern in args.file or []:
        for path in glob.glob(pattern):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    items.append({"source": path, "content": f.read()})
            except Exception as e:
                print(f"! Skipping {path}: {e}", file=sys.stderr)

    if args.stdin:
        content = sys.stdin.read()
        if content.strip():
            items.append({"source": "stdin", "content": content})

    if not items:
        raise SystemExit("No input provided. Use --text, --file, or --stdin.")

    return items


async def _evaluate_items(items: List[Dict[str, Any]], use_poml: bool) -> List[Dict[str, Any]]:
    engine = OrchestratedStoryEngine(use_poml=use_poml)
    out: List[Dict[str, Any]] = []

    for it in items:
        content = (it.get("content") or "").strip()
        if not content:
            out.append({"source": it["source"], "error": "empty content"})
            continue

        res = await engine.evaluate_quality(content)
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
        out.append({
            "source": it["source"],
            "content_hash": content_hash,
            "evaluation_text": res.get("evaluation_text", ""),
            "scores": res.get("scores", {}),
            "meta": res.get("meta", {}),
        })

    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate story content quality using POML")
    g_in = p.add_argument_group("inputs")
    g_in.add_argument("--text", help="Inline text to evaluate", default=None)
    g_in.add_argument("--file", nargs="*", help="One or more file globs to evaluate", default=[])
    g_in.add_argument("--stdin", action="store_true", help="Read content from STDIN")

    g_out = p.add_argument_group("output")
    g_out.add_argument("--out", help="Write JSONL to this file; default prints to stdout", default=None)

    g_cfg = p.add_argument_group("config")
    g_cfg.add_argument("--live", action="store_true", help="Use live provider (LM Studio) via env vars")

    args = p.parse_args()

    use_poml = True  # Always use POML evaluation template here

    # Optional live mode prompt
    if args.live and (not os.environ.get("LM_ENDPOINT") or not os.environ.get("LMSTUDIO_MODEL")):
        print("! --live is set but LM_ENDPOINT/LMSTUDIO_MODEL not found in env; using defaults.", file=sys.stderr)

    items = _read_inputs(args)
    results = asyncio.get_event_loop().run_until_complete(_evaluate_items(items, use_poml))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {len(results)} results to {args.out}")
    else:
        print(json.dumps(results, indent=2, ensure_ascii=False))

    # Best-effort store
    if os.getenv("STORE_ALL") == "1" or os.getenv("DB_PASSWORD"):
        try:
            store_workflow_output("evaluate_poml", {"items": results})
            print("Stored evaluation batch to DB (evaluate_poml)")
        except Exception as e:
            print(f"Store skipped: {e}")


if __name__ == "__main__":
    main()
