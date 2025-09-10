import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def _ensure_outdir() -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path("dist") / f"run-{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _default_characters() -> List[Dict[str, Any]]:
    return [
        {
            "id": "pontius_pilate",
            "name": "Pontius Pilate",
            "traits": ["pragmatic", "ambitious", "anxious"],
            "values": ["order", "duty", "Roman law"],
            "fears": ["rebellion", "imperial disfavor"],
        },
        {
            "id": "high_priest",
            "name": "High Priest",
            "traits": ["authoritative", "political", "pious"],
            "values": ["religious law", "tradition", "authority"],
            "fears": ["heresy", "unrest"],
        },
    ]


async def run(args: argparse.Namespace) -> int:
    # Build unified orchestrator (points to ai-lb via config)
    from story_engine.core.core.orchestration.unified_llm_orchestrator import (
        UnifiedLLMOrchestrator,
        LLMPersona,
    )

    orch = UnifiedLLMOrchestrator.from_env_and_config({})

    # Load characters
    chars: List[Dict[str, Any]] = _default_characters()
    if args.characters and Path(args.characters).exists():
        import yaml
        raw = yaml.safe_load(Path(args.characters).read_text()) or []
        if isinstance(raw, dict) and raw.get("characters"):
            chars = list(raw.get("characters") or [])
        elif isinstance(raw, list):
            chars = list(raw)

    situation = args.situation or "A charged public confrontation where a verdict must be decided."
    emphasis = args.emphasis or "neutral"

    # Concurrency
    sem = asyncio.Semaphore(args.max_concurrent)

    async def _simulate(ch: Dict[str, Any]) -> Dict[str, Any]:
        async with sem:
            data = {
                "character": ch,
                "situation": situation,
                "emphasis": emphasis,
            }
            # Route with sticky session per character id
            resp = await orch.generate_with_persona(
                LLMPersona.CHARACTER_SIMULATOR,
                data,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                provider_name=None,
                allow_fallback=True,
                session_id=ch.get("id") or ch.get("name"),
            )
            text = getattr(resp, "text", None) or getattr(resp, "content", "") or ""
            return {"character": ch, "response": text}

    results = await asyncio.gather(*[_simulate(c) for c in chars], return_exceptions=False)

    # Write outputs
    outdir = _ensure_outdir()
    (outdir / "scene_multi.json").write_text(json.dumps({
        "situation": situation,
        "emphasis": emphasis,
        "results": results,
    }, indent=2))

    # Console sample
    lines: List[str] = []
    lines.append("# Multi-Character Scene\n")
    lines.append(f"Situation: {situation}")
    lines.append(f"Emphasis: {emphasis}\n")
    for r in results:
        ch = r.get("character", {})
        lines.append(f"## {ch.get('name','Character')} ({ch.get('id','')})")
        lines.append(r.get("response", "").strip() + "\n")
    (outdir / "console.multi.md").write_text("\n".join(lines))

    print(f"Multi-character outputs written to: {outdir}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Run a simple multi-character scene via ai-lb + POML personas")
    p.add_argument("--characters", help="Path to YAML with a list at 'characters:'", default=None)
    p.add_argument("--situation", help="Situation prompt", default=None)
    p.add_argument("--emphasis", help="Emphasis tag", default="neutral")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-tokens", type=int, default=600)
    p.add_argument("--max-concurrent", type=int, default=4)
    args = p.parse_args()

    try:
        raise SystemExit(asyncio.run(run(args)))
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()

