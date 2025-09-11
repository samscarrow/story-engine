from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

import pytest

from story_engine.core.story_engine.story_engine_orchestrated import (
    OrchestratedStoryEngine,
    StoryComponent,
)
from story_engine.core.domain.models import StoryRequest
from story_engine.core.orchestration.model_filters import (
    filter_models,
    choose_first_id,
)


pytestmark = [
    pytest.mark.slow,
    pytest.mark.silver,
    pytest.mark.skipif(
        os.getenv("STORY_ENGINE_LIVE") != "1",
        reason="silver live tests are opt-in (set STORY_ENGINE_LIVE=1)",
    ),
]


def _load_prompts() -> List[Dict[str, Any]]:
    here = Path(__file__).parent
    path = here / "silver" / "prompts.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


PROMPTS = _load_prompts()


def _choose_small_model(models: List[Dict[str, Any]]) -> Optional[str]:
    # Use capability-aware filter, preferring small models when available
    filtered = filter_models(models, require_text=True, prefer_small=True)
    return choose_first_id(filtered)


@pytest.mark.parametrize(
    "spec", PROMPTS, ids=[p.get("id", str(i)) for i, p in enumerate(PROMPTS)]
)
def test_silver_parallel_small_models(spec: Dict[str, Any]) -> None:
    engine = OrchestratedStoryEngine(use_poml=True)

    # Reduce token usage further for small models
    profiles = engine.component_profiles
    profiles[StoryComponent.PLOT_STRUCTURE]["max_tokens"] = 150
    profiles[StoryComponent.SCENE_DETAILS]["max_tokens"] = 180
    profiles[StoryComponent.CHARACTER_DIALOGUE]["max_tokens"] = 120
    profiles[StoryComponent.QUALITY_EVALUATION]["max_tokens"] = 150

    # Reduce provider timeout if requested
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
        # Health & model selection
        health = await engine.orchestrator.health_check_all()
        if not any(v.get("healthy") for v in health.values()):
            pytest.skip(f"No healthy providers: {health}")
        # Pick a small model if available
        models = []
        for v in health.values():
            ms = v.get("models") or []
            if isinstance(ms, list):
                models.extend(ms)
        chosen = _choose_small_model(models)
        if not chosen:
            pytest.skip("No <=4B-class models found via ai-lb /v1/models")
        os.environ["LM_MODEL"] = str(chosen)

        # Plot → beats (allow minimal beats)
        plot = await engine.generate_plot_structure(req)
        beats = plot.get("beats") or ["Setup"]

        # Scene from first beat
        scene = await engine.generate_scene(beats[0], req.characters)
        assert isinstance(scene.get("scene_description", ""), str)
        assert len(scene["scene_description"]) > 0

        # Optional dialogue if at least one character
        if req.characters:
            dlg = await engine.generate_dialogue(
                scene, req.characters[0], "Opening line"
            )
            assert isinstance(dlg, dict) and "dialogue" in dlg
            assert len(dlg["dialogue"]) > 0

    asyncio.run(run())
