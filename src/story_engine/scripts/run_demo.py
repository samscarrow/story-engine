import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from llm_observability import get_logger, init_logging_from_env

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
from story_engine.core.core.common.anchors import load_anchors, compute_decisions_id


def load_config(path: Path | None) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    if path and path.exists() and yaml is not None:
        with path.open("r") as f:
            cfg = yaml.safe_load(f) or {}  # type: ignore[attr-defined]
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
        "SEED",
        "DEMO_SEED",
        "STORY_ENGINE_SEED",
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
            emotional_state=EmotionalState(
                anger=0.3, doubt=0.7, fear=0.6, compassion=0.3, confidence=0.6
            ),
            memory=CharacterMemory(recent_events=["Tense crowd", "Dream warning"]),
            current_goal="Maintain order without inciting revolt",
            internal_conflict="Duty vs. justice",
        )
    # Future profiles can be added here
    return make_character("pilate")


async def run(args: argparse.Namespace) -> int:
    init_logging_from_env()
    # Global seed for more deterministic behavior (optional via env)
    try:
        seed_str = os.getenv("DEMO_SEED") or os.getenv("STORY_ENGINE_SEED") or os.getenv("SEED")
        if seed_str:
            import random
            os.environ["PYTHONHASHSEED"] = str(int(seed_str))
            random.seed(int(seed_str))
            try:
                import numpy as _np  # type: ignore
                _np.random.seed(int(seed_str))  # type: ignore[attr-defined]
            except Exception:
                pass
    except Exception:
        pass
    if args.job_id:
        try:
            correlation_id_var.set(str(args.job_id))
        except Exception:
            pass
    log = get_logger("story_engine.demo", job_id=args.job_id)
    # Anchors (decisions)
    anchors = load_anchors()
    decisions_id = compute_decisions_id(anchors) if anchors else None
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
        max_retries=config.get("simulation", {})
        .get("retry", {})
        .get("max_attempts", 2),
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
        character,
        situation,
        num_runs=args.runs,
        emphases=[args.emphasis],
        world_pov=(args.world_pov or None),
    )

    # Optional planning + continuity check if the engine provides it
    beats = default_beats()
    plan: Dict[str, Any] = {}
    continuity: Dict[str, Any] = {
        "ok": True,
        "violations": [],
        "summary": "skipped (POML disabled)",
    }
    graph_dict: Dict[str, Any] = {}
    if (
        getattr(engine, "use_poml", False)
        and getattr(engine, "poml_adapter", None) is not None
    ):
        try:
            world_state = {"locale": "Judaea", "era": "1st century CE"}
            if bool(args.iterative_world):
                config.setdefault("simulation", {})["iterative_world_state"] = True
            plan = await engine.plan_scene_with_continuity(
                beats,
                world_state=world_state,
                objective="Resolve the conflict publicly",
                style="historical drama",
                tolerance=0,
                max_attempts=2,
            )
            continuity = plan.get("continuity_report") or {
                "ok": True,
                "violations": [],
                "summary": "no report",
            }
            graph = engine.graph_from(beats, plan)
            graph_dict = graph.to_dict() if hasattr(graph, "to_dict") else {}
        except Exception:
            continuity = {"ok": False, "violations": [], "summary": "planning failed"}
            graph_dict = {}

    # Compose outputs
    outdir = ensure_outdir()
    save_snapshot(outdir, config)
    # Save CLI args for reproducibility
    try:
        # Convert argparse.Namespace to a plain dict
        args_dict = {k: getattr(args, k) for k in vars(args)}
        (outdir / "args.snapshot.json").write_text(json.dumps(args_dict, indent=2))
    except Exception:
        pass

    # Orchestrator health and available models (live mode)
    if orchestrator is not None:
        try:
            health = await orchestrator.health_check_all()
            (outdir / "orchestrator_health.json").write_text(json.dumps(health, indent=2))
        except Exception:
            pass
        try:
            models = await orchestrator.list_models_filtered()
            (outdir / "lb_models.json").write_text(json.dumps(models, indent=2))
        except Exception:
            pass
        # Optional: scrape /metrics from ai-lb if reachable
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:8000/metrics", timeout=5) as resp:
                    if resp.status == 200:
                        (outdir / "lb_metrics.txt").write_text(await resp.text())
        except Exception:
            pass

    story_payload = {
        "character": character.name,
        "situation": situation,
        "emphasis": args.emphasis,
        "runs": results,
        "scene_plan": plan,
        "graph": graph_dict,
    }
    if decisions_id:
        story_payload["decisions_id"] = decisions_id
    (outdir / "story.json").write_text(
        json.dumps(story_payload, indent=2)
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
        resp = r.get("response") or {}
        s = resp.get("dialogue")
        if isinstance(s, list):
            try:
                s = " ".join([str(x) for x in s])
            except Exception:
                s = str(s)
        elif not isinstance(s, str):
            s = str(s or "")
        if s:
            dialogues.append(s)
    schema_valid_runs = 0
    degraded_runs = 0
    for r in results:
        meta = r.get("metadata") or {}
        if meta.get("schema_valid") is True:
            schema_valid_runs += 1
        if meta.get("parse_fallback_used") is True:
            degraded_runs += 1
    metrics = {
        "runs": len(results),
        "schema_valid": schema_valid_runs == len(results) and len(results) > 0,
        "schema_valid_runs": schema_valid_runs,
        "degraded_runs": degraded_runs,
        "continuity_ok": bool(continuity.get("ok")),
        "continuity_violations": len(continuity.get("violations") or []),
        "ngram_repetition_3": round(ngram_repetition(dialogues, 3), 4),
    }
    if decisions_id:
        metrics["decisions_id"] = decisions_id
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Trace file: minimal per-run metadata for audits (no raw content)
    try:
        with (outdir / "trace.ndjson").open("w") as tf:
            for r in results:
                m = (r.get("metadata") or {})
                lb = m.get("lb") or {}
                tf.write(json.dumps({
                    "ts": r.get("timestamp"),
                    "character": r.get("character_id"),
                    "emphasis": r.get("emphasis"),
                    "temperature": r.get("temperature"),
                    "provider_name": m.get("provider_name"),
                    "provider_type": m.get("provider_type"),
                    "model": m.get("model"),
                    "lb": {
                        "x_selected_model": lb.get("x_selected_model"),
                        "x_routed_node": lb.get("x_routed_node"),
                        "x_request_id": lb.get("x_request_id"),
                    }
                }) + "\n")
    except Exception:
        pass

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

    # Strict gating (optional): fail on degraded or continuity violations
    if getattr(args, "strict_output", False):
        degraded = metrics.get("degraded_runs", 0) > 0
        cont_bad = not metrics.get("continuity_ok", True) and metrics.get(
            "continuity_violations", 0
        ) > 0
        if degraded or cont_bad:
            print(
                f"Strict output gating failed: degraded_runs={metrics.get('degraded_runs')}, "
                f"continuity_ok={metrics.get('continuity_ok')}, "
                f"violations={metrics.get('continuity_violations')}"
            )
            print(f"Demo outputs written to: {outdir}")
            return 2

    log.info("demo.outputs_written", extra={"outdir": str(outdir)})
    print(f"Demo outputs written to: {outdir}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Run Story Engine demo")
    p.add_argument("--config", help="Path to config.yaml", default=None)
    p.add_argument(
        "--use-poml", action="store_true", help="Enable POML adapter prompts"
    )
    p.add_argument(
        "--live", action="store_true", help="Use live orchestrator if configured"
    )
    p.add_argument(
        "--strict-persona", action="store_true", help="Enable strict persona mode"
    )
    p.add_argument(
        "--persona-threshold",
        type=int,
        default=None,
        help="Persona adherence threshold (0-100)",
    )
    p.add_argument(
        "--profile", default="pilate", help="Character profile id (default: pilate)"
    )
    p.add_argument("--job-id", default=None, help="Correlation/job id for logs")
    p.add_argument("--situation", default=None, help="Situation to simulate")
    p.add_argument(
        "--emphasis",
        default="neutral",
        help="Emphasis: power|doubt|fear|compassion|duty|neutral",
    )
    p.add_argument(
        "--world-pov",
        default=None,
        help=(
            "Short world perspective to ground simulation language and setting. "
            "Keep English as the narrative language; at most occasional transliterated "
            "period terms with brief gloss."
        ),
    )
    p.add_argument("--runs", type=int, default=3, help="Number of simulation runs")
    p.add_argument(
        "--strict-output",
        action="store_true",
        help="Fail if degraded outputs are detected or continuity violations persist",
    )
    p.add_argument(
        "--iterative-world",
        action="store_true",
        help="Enable sequential world-state updates per scene (disables scene parallelism)",
    )
    args = p.parse_args()

    try:
        raise SystemExit(asyncio.run(run(args)))
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
