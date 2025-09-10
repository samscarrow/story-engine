"""
Live-fire test for POML flow using the real orchestrator from config.yaml.
Skips unless STORY_ENGINE_LIVE=1 and at least one provider is healthy.

This test is designed to be light on tokens and run quickly.
"""

import os
import asyncio
import pytest

from story_engine.core.story_engine.story_engine_orchestrated import OrchestratedStoryEngine, StoryComponent
from story_engine.core.domain.models import StoryRequest

# Opt-in and marker for slow/live tests
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(os.getenv("STORY_ENGINE_LIVE") != "1", reason="live test opt-in (set STORY_ENGINE_LIVE=1)"),
]


def _parse_eval(text: str):
    scores = {}
    if not isinstance(text, str):
        return scores
    for line in text.splitlines():
        if ':' in line and '/' in line:
            try:
                metric, rest = line.split(':', 1)
                num = rest.strip().split('/')[0]
                score = float(num)
                scores[metric.strip()] = score
            except Exception:
                continue
    return scores


def test_live_poml_pilate_flow_minimal():
    engine = OrchestratedStoryEngine(use_poml=True)

    # Optionally reduce provider timeouts for test runs via env
    try:
        active = engine.orchestrator.active_provider or next(iter(engine.orchestrator.providers))
        prov = engine.orchestrator.providers.get(active)
        if prov:
            t = os.getenv("LLM_TEST_TIMEOUT")
            if t:
                prov.config.timeout = int(t)
    except Exception:
        pass

    # Reduce token usage for a quick live test
    profiles = engine.component_profiles
    profiles[StoryComponent.PLOT_STRUCTURE]["max_tokens"] = 200
    profiles[StoryComponent.SCENE_DETAILS]["max_tokens"] = 200
    profiles[StoryComponent.CHARACTER_DIALOGUE]["max_tokens"] = 150
    profiles[StoryComponent.QUALITY_EVALUATION]["max_tokens"] = 200
    profiles[StoryComponent.ENHANCEMENT]["max_tokens"] = 200

    request = StoryRequest(
        title="The Trial Before Dawn",
        premise=(
            "A Roman governor must choose between political stability and his sense of justice "
            "when a controversial prophet is brought before him."
        ),
        genre="Historical Drama",
        tone="Grave",
        characters=[
            {"id": "pilate", "name": "Pontius Pilate", "role": "conflicted judge"},
            {"id": "caiaphas", "name": "Caiaphas", "role": "antagonist"},
        ],
        setting="Jerusalem",
        structure="three_act",
    )

    async def run():
        # Health check
        health = await engine.orchestrator.health_check_all()
        if not any(v.get("healthy") for v in health.values()):
            pytest.skip(f"No healthy providers: {health}")

        # Plot → beats
        plot = await engine.generate_plot_structure(request)
        assert plot.get("raw_text")
        beats = plot.get("beats") or []
        assert len(beats) >= 3

        # Scene from first beat
        scene = await engine.generate_scene(beats[0], request.characters)
        assert isinstance(scene.get("scene_description", ""), str) and len(scene["scene_description"]) > 0
        # Content should reference setting/characters roughly
        assert any(k in scene["scene_description"].lower() for k in ["pilate", "jerusalem", "crowd"]) 

        # Dialogue for Pilate
        dlg = await engine.generate_dialogue(scene, request.characters[0], "Opening line")
        assert isinstance(dlg, dict)
        assert "dialogue" in dlg and len(dlg["dialogue"]) > 0
        assert isinstance(dlg["dialogue"][0]["line"], str) and len(dlg["dialogue"][0]["line"]) > 0

        # Evaluation and enhancement
        ev = await engine.evaluate_quality(scene["scene_description"]) 
        assert isinstance(ev.get("evaluation_text", ""), str) and len(ev["evaluation_text"]) > 0
        scores = _parse_eval(ev["evaluation_text"]) 
        # At least 3 metrics detected with plausible scores
        assert len([s for s in scores.values() if 0 < s <= 10]) >= 3
        enhanced = await engine.enhance_content(scene["scene_description"], ev, "pacing and emotion")
        assert isinstance(enhanced, str) and len(enhanced) > 0

    asyncio.run(run())
