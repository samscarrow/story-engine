import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml  # type: ignore
except Exception:  # Optional dependency for config
    yaml = None  # type: ignore

# Local imports from the engine
from story_engine.core.core.character_engine.character_simulation_engine_v2 import (
    SimulationEngine,
    MockLLM,
    RetryHandler,
    EmotionalState,
    CharacterMemory,
    CharacterState,
)


def load_config(path: Path | None) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    if path and path.exists() and yaml is not None:
        with path.open("r") as f:
            cfg = (yaml.safe_load(f) or {})  # type: ignore[attr-defined]
    # Merge env toggles
    prefer_small = os.getenv("LM_PREFER_SMALL") in ("1", "true", "True")
    if prefer_small:
        cfg.setdefault("llm", {}).setdefault("prefer_small_models", True)
    return cfg


def ensure_outdir() -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path("dist") / f"run-{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_snapshot(outdir: Path, config: Dict[str, Any]) -> None:
    # Save config snapshot and env capture for reproducibility
    if yaml is not None:
        (outdir / "config.snapshot.yaml").write_text(yaml.safe_dump(config, sort_keys=False))  # type: ignore[attr-defined]
    else:
        (outdir / "config.snapshot.json").write_text(json.dumps(config, indent=2))
    capture_keys = [
        "LM_PREFER_SMALL",
        "LM_MODEL",
        "OPENAI_API_KEY",
        "STORY_ENGINE_LIVE",
    ]
    env = {k: os.getenv(k, "") for k in capture_keys}
    (outdir / "env.capture").write_text("\n".join(f"{k}={v}" for k, v in env.items()))


def default_beats() -> List[Dict[str, Any]]:
    # A tiny beat list used by the planner/graph in SimulationEngine
    return [
        {"id": "b1", "event": "Initial encounter", "stakes": "Political"},
        {"id": "b2", "event": "Crowd demands verdict", "stakes": "Public order"},
        {"id": "b3", "event": "Wife's warning", "stakes": "Personal risk"},
    ]


def make_character(profile: str | None) -> CharacterState:
    if profile == "pilate" or profile is None:
        return CharacterState(
            id="pontius_pilate",
            name="Pontius Pilate",
            backstory={"origin": "Roman equestrian", "career": "Prefect of Judaea"},
            traits=["pragmatic", "ambitious", "anxious"],
            values=["order", "duty", "Roman law"],
            fears=["rebellion", "imperial disfavor"],
            desires=["peace", "advancement"],
            emotional_state=EmotionalState(anger=0.3, doubt=0.7, fear=0.6, compassion=0.3, confidence=0.6),
            memory=CharacterMemory(recent_events=["Tense crowd", "Dream warning"]),
            current_goal="Maintain order without inciting revolt",
            internal_conflict="Duty vs. justice",
        )
    # Future profiles can be added here
    return make_character("pilate")


async def run(args: argparse.Namespace) -> int:
    cfg_path = Path(args.config) if args.config else Path("config.yaml")
    config = load_config(cfg_path if cfg_path.exists() else None)

    # Feature flags from CLI override config
    # Respect config.yaml defaults; only enable POML explicitly if requested.
    config.setdefault("simulation", {})
    if bool(args.use_poml):
        config["simulation"]["use_poml"] = True
    config.setdefault("features", {})
    config["features"]["strict_persona_mode"] = bool(args.strict_persona)
    if args.persona_threshold is not None:
        config["features"]["persona_adherence_threshold"] = int(args.persona_threshold)

    # LLM selection: mock by default; if --live, use unified orchestrator if available
    llm = MockLLM()
    orchestrator = None
    if args.live:
        try:
            from story_engine.core.core.orchestration.unified_llm_orchestrator import (
                UnifiedLLMOrchestrator,
            )

            orchestrator = UnifiedLLMOrchestrator.from_env_and_config(config)
        except Exception:
            orchestrator = None

    # Engine
    retry = RetryHandler(
        max_retries=config.get("simulation", {}).get("retry", {}).get("max_attempts", 2),
        base_delay=config.get("simulation", {}).get("retry", {}).get("base_delay", 0.6),
    )
    engine = SimulationEngine(
        llm_provider=llm if orchestrator is None else None,
        orchestrator=orchestrator,
        max_concurrent=config.get("simulation", {}).get("max_concurrent", 4),
        retry_handler=retry,
        config=config,
    )

    # Inputs
    character = make_character(args.profile)
    situation = (
        args.situation
        or "You face a moral decision under public pressure; choose and justify."
    )

    # Run a few simulations for diversity
    results = await engine.run_multiple_simulations(
        character, situation, num_runs=args.runs, emphases=[args.emphasis]
    )

    # Optional planning + continuity check if the engine provides it
    beats = default_beats()
    plan: Dict[str, Any] = {}
    continuity: Dict[str, Any] = {"ok": True, "violations": [], "summary": "skipped (POML disabled)"}
    graph_dict: Dict[str, Any] = {}
    if getattr(engine, 'use_poml', False) and getattr(engine, 'poml_adapter', None) is not None:
        try:
            plan = await engine.plan_scene(beats, objective="Resolve the conflict publicly", style="historical drama")
            continuity = await engine.continuity_check_scene(plan, world_state={"locale": "Judaea", "era": "1st century"})
            graph = engine.graph_from(beats, plan)
            graph_dict = graph.to_dict() if hasattr(graph, 'to_dict') else {}
        except Exception:
            continuity = {"ok": False, "violations": [], "summary": "planning failed"}
            graph_dict = {}

    # Compose outputs
    outdir = ensure_outdir()
    save_snapshot(outdir, config)

    (outdir / "story.json").write_text(
        json.dumps(
            {
                "character": character.name,
                "situation": situation,
                "emphasis": args.emphasis,
                "runs": results,
                "scene_plan": plan,
                "graph": graph_dict,
            },
            indent=2,
        )
    )

    (outdir / "continuity_report.json").write_text(json.dumps(continuity, indent=2))

    # Minimal metrics for evaluation
    def ngram_repetition(dialogues: List[str], n: int = 3) -> float:
        from collections import Counter

        tokens: List[str] = " ".join(dialogues).split()
        if len(tokens) < n:
            return 0.0
        grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        c = Counter(grams)
        rep = sum(v for v in c.values() if v > 1)
        return rep / max(1, len(grams))

    dialogues = []
    for r in results:
        resp = (r.get("response") or {})
        s = resp.get("dialogue") or ""
        if s:
            dialogues.append(s)
    metrics = {
        "runs": len(results),
        "schema_valid": all(
            isinstance(r.get("response"), dict)
            and {"dialogue", "thought", "action", "emotional_shift"}.issubset(r["response"].keys())
            for r in results
        ),
        "continuity_ok": bool(continuity.get("ok")),
        "continuity_violations": len(continuity.get("violations") or []),
        "ngram_repetition_3": round(ngram_repetition(dialogues, 3), 4),
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Human-readable console.md
    lines: List[str] = []
    lines.append("# Story Engine Demo\n")
    lines.append(f"Character: {character.name}")
    lines.append(f"Emphasis: {args.emphasis}")
    lines.append(f"Runs: {len(results)}\n")
    lines.append("## Samples\n")
    for i, r in enumerate(results, 1):
        rsp = r.get("response") or {}
        lines.append(f"### Run {i}")
        lines.append(f"Dialogue: {rsp.get('dialogue','').strip()}")
        lines.append(f"Thought: {rsp.get('thought','').strip()}")
        lines.append(f"Action: {rsp.get('action','').strip()}\n")
    lines.append("## Continuity\n")
    lines.append(json.dumps(continuity, indent=2))
    lines.append("\n## Metrics\n")
    lines.append(json.dumps(metrics, indent=2))
    (outdir / "console.md").write_text("\n".join(lines))

    print(f"Demo outputs written to: {outdir}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Run Story Engine demo")
    p.add_argument("--config", help="Path to config.yaml", default=None)
    p.add_argument("--use-poml", action="store_true", help="Enable POML adapter prompts")
    p.add_argument("--live", action="store_true", help="Use live orchestrator if configured")
    p.add_argument("--strict-persona", action="store_true", help="Enable strict persona mode")
    p.add_argument("--persona-threshold", type=int, default=None, help="Persona adherence threshold (0-100)")
    p.add_argument("--profile", default="pilate", help="Character profile id (default: pilate)")
    p.add_argument("--situation", default=None, help="Situation to simulate")
    p.add_argument("--emphasis", default="neutral", help="Emphasis: power|doubt|fear|compassion|duty|neutral")
    p.add_argument("--runs", type=int, default=3, help="Number of simulation runs")
    args = p.parse_args()

    try:
        raise SystemExit(asyncio.run(run(args)))
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
