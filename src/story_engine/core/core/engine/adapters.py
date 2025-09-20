from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from llm_observability import get_logger
from ..story_engine.narrative_pipeline import NarrativePipeline
from story_engine.poml.lib.poml_integration import StoryEnginePOMLAdapter
from .interfaces import (
    BaseEngine,
    EngineContext,
    EngineResult,
    Plan,
    RetryPolicy,
    Step,
    StepKind,
    ValidationReport,
)
from .orchestrator import EngineOrchestrator


class _CtxAIOrchestrator:
    """Thin orchestrator that forwards generate() to EngineContext.ai.

    Satisfies NarrativePipeline's expectation of an object with `generate`.
    """

    def __init__(self, ctx: EngineContext):
        self._ctx = ctx

    async def generate(self, prompt: str, **kwargs):
        if not self._ctx.ai:
            raise RuntimeError("EngineContext.ai is not configured")
        return await self._ctx.ai.generate(prompt=prompt, **kwargs)


@dataclass
class NarrativePipelineEngine(BaseEngine):
    """Adapter that exposes NarrativePipeline as a BaseEngine.

    The sample plan crafts one scene from the first beat and persists it.
    It uses EngineContext.ai for LLM calls through a thin orchestrator shim.
    """

    use_poml: bool = False

    def describe(self) -> str:
        return "NarrativePipelineEngine: crafts a scene and persists output"

    def plan(self, inputs: Dict[str, Any]) -> Plan:
        title = inputs.get("title") or "Untitled"
        premise = inputs.get("premise") or ""
        characters: List[Dict[str, Any]] = list(inputs.get("characters") or [])
        num_beats = int(inputs.get("num_beats", 5))
        artifact_key = inputs.get("artifact_key") or self._default_artifact_key(
            title, premise, characters
        )

        # Build scene prompt using POML or fallback
        def _build_scene_prompt(ctx: EngineContext, params: Dict[str, Any]) -> Any:
            try:
                adapter = StoryEnginePOMLAdapter()
            except Exception:
                adapter = None

            if self.use_poml and adapter is not None:
                prompt = adapter.get_scene_prompt(
                    beat={
                        "id": 0,
                        "name": "Setup",
                        "purpose": "Establish normal",
                        "tension": 0.2,
                    },
                    characters=characters,
                    previous_context="",
                )
            else:
                prompt = (
                    f"Create a dramatic scene for:\n"
                    f"Beat: Setup - Establish normal\n"
                    f"Tension level: 0.2\n"
                    f"Characters: {', '.join([c['name'] for c in characters])}\n"
                    f"Previous context: Opening scene\n\n"
                    "Provide a detailed situation description (2-3 sentences) that gives characters clear dramatic opportunities.\n"
                    "Include: location, time of day, immediate conflict or tension, and what's at stake.\n\n"
                    "Scene situation:"
                )
            return {"prompt": prompt}

        # AI step: generate situation text from prompt
        async def _generate_situation(ctx: EngineContext, params: Dict[str, Any]) -> Any:
            if not ctx.ai:
                raise RuntimeError("EngineContext.ai is not configured")
            p = ctx.results.get("build_scene_prompt", {}).get("prompt", "")
            resp = await ctx.ai.generate(prompt=p, temperature=0.8)
            text = getattr(resp, "text", "")
            return {"text": text}

        # Build sensory prompt from generated situation
        def _build_sensory_prompt(ctx: EngineContext, params: Dict[str, Any]) -> Any:
            situation = ctx.results.get("generate_situation", {}).get("text", "")
            base = situation[:100]
            prompt = (
                f"For this scene: {base}... Provide ONE sensory detail each: "
                "sight, sound, atmosphere. Brief, evocative."
            )
            return {"prompt": prompt}

        # AI step: generate sensory response
        async def _generate_sensory(ctx: EngineContext, params: Dict[str, Any]) -> Any:
            if not ctx.ai:
                raise RuntimeError("EngineContext.ai is not configured")
            p = ctx.results.get("build_sensory_prompt", {}).get("prompt", "")
            resp = await ctx.ai.generate(prompt=p, temperature=0.9)
            return {"text": getattr(resp, "text", "")}

        # Assemble final scene dict
        def _assemble_scene(ctx: EngineContext, params: Dict[str, Any]) -> Any:
            situation = ctx.results.get("generate_situation", {}).get("text", "")
            sensory_resp = ctx.results.get("generate_sensory", {}).get("text", "")
            sensory = _parse_sensory(sensory_resp)
            location = _extract_location(situation)
            return {
                "id": 0,
                "name": "Setup",
                "location": location,
                "tension": 0.2,
                "situation": situation,
                "characters": [c["id"] for c in characters],
                "sensory": sensory,
            }

        # Persist step writes artifact if repo configured
        def _persist(ctx: EngineContext, params: Dict[str, Any]) -> Any:
            data = ctx.results.get("assemble_scene") or {}
            if ctx.artifacts:
                ctx.artifacts.save("scene", artifact_key, data)
            return {"saved": True, "key": artifact_key}

        steps = {
            "build_scene_prompt": Step(
                key="build_scene_prompt",
                kind=StepKind.TRANSFORM,
                func=_build_scene_prompt,
                retry=RetryPolicy(max_attempts=1),
                timeout_sec=10,
                metadata={"engine": "narrative_pipeline", "stage": "build_prompt"},
            ),
            "generate_situation": Step(
                key="generate_situation",
                kind=StepKind.AI_REQUEST,
                func=_generate_situation,
                depends_on=["build_scene_prompt"],
                retry=RetryPolicy(max_attempts=2, base_delay_sec=0.3),
                timeout_sec=inputs.get("timeout_sec", 120),
                idempotency_key=(self.idempotency_key(inputs) or "") + ":situation",
                metadata={"engine": "narrative_pipeline", "stage": "llm_situation"},
            ),
            "build_sensory_prompt": Step(
                key="build_sensory_prompt",
                kind=StepKind.TRANSFORM,
                func=_build_sensory_prompt,
                depends_on=["generate_situation"],
                retry=RetryPolicy(max_attempts=1),
                timeout_sec=10,
                metadata={"engine": "narrative_pipeline", "stage": "build_sensory"},
            ),
            "generate_sensory": Step(
                key="generate_sensory",
                kind=StepKind.AI_REQUEST,
                func=_generate_sensory,
                depends_on=["build_sensory_prompt"],
                retry=RetryPolicy(max_attempts=2, base_delay_sec=0.3),
                timeout_sec=inputs.get("timeout_sec", 60),
                idempotency_key=(self.idempotency_key(inputs) or "") + ":sensory",
                metadata={"engine": "narrative_pipeline", "stage": "llm_sensory"},
            ),
            "assemble_scene": Step(
                key="assemble_scene",
                kind=StepKind.TRANSFORM,
                func=_assemble_scene,
                depends_on=["generate_situation", "generate_sensory"],
                retry=RetryPolicy(max_attempts=1),
                timeout_sec=10,
                metadata={"engine": "narrative_pipeline", "stage": "assemble"},
            ),
            "persist": Step(
                key="persist",
                kind=StepKind.PERSIST,
                func=_persist,
                depends_on=["assemble_scene"],
                retry=RetryPolicy(max_attempts=1),
                timeout_sec=inputs.get("timeout_sec", 30),
                idempotency_key=(self.idempotency_key(inputs) or "") + ":persist",
                metadata={"engine": "narrative_pipeline", "stage": "persist"},
            ),
        }
        return Plan(steps=steps, roots=["build_scene_prompt"], metadata={"artifact_key": artifact_key})

    async def execute(self, plan: Plan, ctx: EngineContext) -> EngineResult:
        orch = EngineOrchestrator()
        return await orch.run(plan, ctx)

    def validate(self, result: EngineResult) -> ValidationReport:
        ok = result.success and "craft_scene" in result.step_results
        issues: List[str] = []
        if not ok:
            issues.append("craft_scene missing or failed")
        else:
            scene = result.step_results["craft_scene"]
            if not scene or not scene.get("situation"):
                ok = False
                issues.append("scene.situation empty")
        return ValidationReport(valid=ok, issues=issues)

    def compensate(self, plan: Plan, ctx: EngineContext) -> None:
        # No-op: artifacts are append-only; cleanup could remove files by key
        return None

    def idempotency_key(self, inputs: Dict[str, Any]) -> Optional[str]:
        base = json.dumps(
            {
                "title": inputs.get("title"),
                "premise": inputs.get("premise"),
                "characters": [c.get("id") for c in (inputs.get("characters") or [])],
                "use_poml": bool(self.use_poml),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]

    def estimate_cost(self, plan: Plan) -> Optional[Dict[str, Any]]:
        # Rough estimate: one scene generation ~ N tokens
        return {"calls": 2, "approx_tokens": 1200}

    def _default_artifact_key(self, title: str, premise: str, chars: List[Dict[str, Any]]) -> str:
        raw = f"{title}:{premise}:{','.join([c.get('id','') for c in chars])}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


# --- Local helpers mirroring NarrativePipeline heuristics ---

def _parse_sensory(response: str) -> Dict[str, str]:
    sensory = {
        "sight": "harsh lighting",
        "sound": "tense silence",
        "atmosphere": "oppressive",
    }
    if response:
        lines = response.lower().split("\n")
        for line in lines:
            if "sight" in line or "see" in line or "visual" in line:
                sensory["sight"] = line.split(":")[-1].strip()[:50]
            elif "sound" in line or "hear" in line or "audio" in line:
                sensory["sound"] = line.split(":")[-1].strip()[:50]
            elif "atmosphere" in line or "feel" in line or "mood" in line:
                sensory["atmosphere"] = line.split(":")[-1].strip()[:50]
    return sensory


def _extract_location(situation: str) -> str:
    location_words = [
        "room",
        "hall",
        "chamber",
        "office",
        "street",
        "building",
        "courtyard",
        "palace",
        "temple",
        "house",
        "plaza",
    ]
    for word in location_words:
        if word in situation.lower():
            return word.capitalize()
    return "Unknown Location"
