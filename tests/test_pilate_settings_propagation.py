"""
Verify that Pontius Pilate settings and profiles propagate through POML prompts
and LLM parameters (temperature, max_tokens) for each component.
"""

import asyncio

from story_engine.core.story_engine.story_engine_orchestrated import (
    OrchestratedStoryEngine,
)
from story_engine.core.domain.models import StoryRequest


class SpyOrchestrator:
    def __init__(self, reply_text: str = "OK"):
        self.prompts = []
        self.all_kwargs = []
        self.calls = 0
        self.reply_text = reply_text

    async def generate(self, prompt: str, **kwargs):
        self.calls += 1
        self.prompts.append(prompt)
        self.all_kwargs.append(kwargs)

        class R:
            def __init__(self, t):
                self.text = t
                self.metadata = {}
                self.timestamp = ""

        # Return structured JSON if the system prompt indicates a structuring task
        system_prompt = kwargs.get("system", "").lower()
        if "json" in system_prompt:
            if "dialogue" in system_prompt:
                return R(
                    '{"dialogue": [{"speaker": "Spy", "line": "Test dialogue", "tone": "neutral", "recipient": "All"}]}'
                )
            if "plot" in system_prompt:
                return R(
                    '{"structure_type": "three_act", "beats": [{"name": "Spy Beat", "description": "A test beat", "tension": 5, "purpose": "testing"}]}'
                )
            if "evaluation" in system_prompt:
                return R('{"evaluation_text": "Good", "scores": {"Coherence": 8}}')
            if "enhancement" in system_prompt:
                return R('{"enhanced_content": "An enhanced scene."}')

        return R(self.reply_text)


def build_request():
    return StoryRequest(
        title="The Trial Before Dawn",
        premise="Pilate faces a prophet under political and moral pressure",
        genre="Historical Drama",
        tone="Grave",
        characters=[
            {"id": "pilate", "name": "Pontius Pilate", "role": "conflicted judge"},
            {"id": "caiaphas", "name": "Caiaphas", "role": "antagonist"},
        ],
        setting="Jerusalem",
        structure="three_act",
    )


def test_pilate_profiles_and_prompts_propagate_with_poml():
    spy = SpyOrchestrator()
    engine = OrchestratedStoryEngine(orchestrator=spy, use_poml=True)
    req = build_request()

    async def run():
        # Plot structure → Two-stage call
        plot = await engine.generate_plot_structure(req)
        assert spy.all_kwargs[-2].get("temperature") == 0.7
        assert spy.all_kwargs[-2].get("max_tokens") == 800
        assert "Title:" in spy.prompts[-2] and "Jerusalem" in spy.prompts[-2]
        assert spy.all_kwargs[-1].get("temperature") == 0.1  # Structuring call
        assert "beats" in plot and len(plot["beats"]) >= 1

        # Scene crafting → temperature 0.8, max_tokens 1000; prompt includes characters
        beat = plot["beats"][0]
        spy.reply_text = (
            "Courtyard before the praetorium; the crowd murmurs as Pilate listens."
        )
        scene = await engine.generate_scene(beat, req.characters)
        assert spy.all_kwargs[-1].get("temperature") == 0.8
        assert spy.all_kwargs[-1].get("max_tokens") == 1000
        assert "Characters Present" in spy.prompts[-1]
        assert "Pontius Pilate" in spy.prompts[-1]
        assert "scene_description" in scene

        # Dialogue → Two-stage call
        # Stage 1 (Creative): temperature 0.9, max_tokens 300
        # Stage 2 (Structuring): temperature 0.1
        spy.reply_text = "What is truth?"
        dlg = await engine.generate_dialogue(scene, req.characters[0], "Opening line")

        # Assert against Stage 1 (Creative call, second to last)
        assert spy.all_kwargs[-2].get("temperature") == 0.9
        assert spy.all_kwargs[-2].get("max_tokens") == 300
        assert "dialogue" in spy.prompts[-2].lower()

        # Assert against Stage 2 (Structuring call, last)
        assert spy.all_kwargs[-1].get("temperature") == 0.1
        assert "json" in spy.all_kwargs[-1].get("system", "").lower()

        # Assert against Stage 2 (Structuring call, last)
        assert spy.all_kwargs[-1].get("temperature") == 0.1
        assert "json" in spy.all_kwargs[-1].get("system", "").lower()

        assert "dialogue" in dlg.lower() and len(dlg) > 0

        # Evaluation → Two-stage call
        spy.reply_text = "Narrative Coherence: 7/10 - OK"
        ev = await engine.evaluate_quality(scene["scene_description"])
        assert spy.all_kwargs[-2].get("temperature") == 0.5
        assert spy.all_kwargs[-2].get("max_tokens") == 1000
        assert (
            "Rate the following" in spy.prompts[-1]
            or "Story Content (Excerpt)" in spy.prompts[-1]
        )
        assert "evaluation_text" in ev

        # Enhancement → Two-stage call
        spy.reply_text = "Enhanced scene"
        enhanced = await engine.enhance_content(
            scene["scene_description"], ev, "pacing and emotion"
        )
        assert spy.all_kwargs[-2].get("temperature") == 0.6
        assert (
            "Focus area:" in spy.prompts[-2] or "Enhancement Focus:" in spy.prompts[-2]
        )
        assert len(enhanced) > 0

    asyncio.run(run())
