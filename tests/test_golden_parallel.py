from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Any, List

import pytest

from story_engine.core.story_engine.story_engine_orchestrated import (
    OrchestratedStoryEngine,
    StoryComponent,
)
from story_engine.core.domain.models import StoryRequest


# Mark these as slow/live; exclude by default with -m "not slow"
pytestmark = [
    pytest.mark.slow,
    pytest.mark.golden,
    pytest.mark.skipif(
        os.getenv("STORY_ENGINE_LIVE") != "1",
        reason="golden live tests are opt-in (set STORY_ENGINE_LIVE=1)",
    ),
]


def _load_prompts() -> List[Dict[str, Any]]:
    here = Path(__file__).parent
    path = here / "golden" / "prompts.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


PROMPTS = _load_prompts()


def _parse_eval(text: str) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    if not isinstance(text, str):
        return scores
    for line in text.splitlines():
        if ":" in line and "/" in line:
            try:
                metric, rest = line.split(":", 1)
                num = rest.strip().split("/")[0]
                score = float(num)
                scores[metric.strip()] = score
            except Exception:
                continue
    return scores


@pytest.mark.parametrize(
    "spec", PROMPTS, ids=[p.get("id", str(i)) for i, p in enumerate(PROMPTS)]
)
def test_golden_parallel_minimal(spec: Dict[str, Any]) -> None:
    engine = OrchestratedStoryEngine(use_poml=True)

    # Reduce token usage for quick, parallel-friendly runs
    profiles = engine.component_profiles
    profiles[StoryComponent.PLOT_STRUCTURE]["max_tokens"] = 200
    profiles[StoryComponent.SCENE_DETAILS]["max_tokens"] = 200
    profiles[StoryComponent.CHARACTER_DIALOGUE]["max_tokens"] = 150
    profiles[StoryComponent.QUALITY_EVALUATION]["max_tokens"] = 200
    profiles[StoryComponent.ENHANCEMENT]["max_tokens"] = 200

    # Optionally reduce provider timeout from env
    try:
        active = engine.orchestrator.active_provider or next(
            iter(engine.orchestrator.providers)
        )
        prov = engine.orchestrator.providers.get(active)
        if prov:
            t = os.getenv("LLM_TEST_TIMEOUT")
            if t:
                prov.config.timeout = int(t)
    except Exception:
        pass

    req = StoryRequest(
        title=spec.get("title", "Untitled"),
        premise=spec.get("premise", ""),
        genre=spec.get("genre", "Historical Drama"),
        tone=spec.get("tone", "Grave"),
        characters=spec.get("characters", []),
        setting=spec.get("setting", "Jerusalem"),
        structure=spec.get("structure", "three_act"),
    )

    async def run() -> None:
        # Health gate
        health = await engine.orchestrator.health_check_all()
        if not any(v.get("healthy") for v in health.values()):
            pytest.skip(f"No healthy providers: {health}")

        # Plot → beats
        plot = await engine.generate_plot_structure(req)
        assert isinstance(plot.get("raw_text", ""), str) and len(plot["raw_text"]) > 0
        beats = plot.get("beats") or []
        assert len(beats) >= 1

        # Scene from first beat
        scene = await engine.generate_scene(beats[0], req.characters)
        assert (
            isinstance(scene.get("scene_description", ""), str)
            and len(scene["scene_description"]) > 0
        )

        # Dialogue for the first character when available
        if req.characters:
            dlg = await engine.generate_dialogue(
                scene, req.characters[0], "Opening line"
            )
            assert isinstance(dlg, dict)
            assert "dialogue" in dlg and len(dlg["dialogue"]) > 0
            assert isinstance(dlg["dialogue"][0]["line"], str)

        # Evaluation (ensure 3 metrics with 0<score<=10 via fallback if needed)
        ev = await engine.evaluate_quality(scene["scene_description"])  # type: ignore[index]
        assert isinstance(ev.get("evaluation_text", ""), str)
        scores = _parse_eval(ev["evaluation_text"])  # type: ignore[index]
        assert len([s for s in scores.values() if 0 < s <= 10]) >= 3

    asyncio.run(run())
