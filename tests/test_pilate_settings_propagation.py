"""
Verify that Pontius Pilate settings and profiles propagate through POML prompts
and LLM parameters (temperature, max_tokens) for each component.
"""

import asyncio

from core.story_engine.story_engine_orchestrated import OrchestratedStoryEngine
from core.domain.models import StoryRequest


class SpyOrchestrator:
    def __init__(self, reply_text: str = "OK"):
        self.last_prompt = None
        self.last_kwargs = None
        self.calls = 0
        self.reply_text = reply_text

    async def generate(self, prompt: str, **kwargs):
        self.calls += 1
        self.last_prompt = prompt
        self.last_kwargs = kwargs
        class R:
            def __init__(self, t):
                self.text = t
                self.metadata = {}
                self.timestamp = ""
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
        # Plot structure → temperature 0.7, max_tokens 800
        spy.reply_text = (
            "Setup: Early dawn in Jerusalem; rumors of a prophet spread.\n\n"
            "Rising Action: Leaders bring Jesus; crowd gathers, pressure mounts.\n\n"
            "Climax: The fateful decision before the crowd.\n\n"
            "Falling Action: Washing of hands; unrest simmers.\n\n"
            "Resolution: Order returns, doubt remains."
        )
        plot = await engine.generate_plot_structure(req)
        assert spy.last_kwargs.get("temperature") == 0.7
        assert spy.last_kwargs.get("max_tokens") == 800
        assert "Title:" in spy.last_prompt and "Jerusalem" in spy.last_prompt
        assert "beats" in plot and len(plot["beats"]) >= 3

        # Scene crafting → temperature 0.8, max_tokens 1000; prompt includes characters
        beat = plot["beats"][0]
        spy.reply_text = "Courtyard before the praetorium; the crowd murmurs as Pilate listens."
        scene = await engine.generate_scene(beat, req.characters)
        assert spy.last_kwargs.get("temperature") == 0.8
        assert spy.last_kwargs.get("max_tokens") == 1000
        assert "Characters Present" in spy.last_prompt
        assert "Pontius Pilate" in spy.last_prompt
        assert scene["name"] in {"Setup", "Rising Action", "Climax", "Falling Action", "Resolution"}

        # Dialogue → temperature 0.9, max_tokens override 300; prompt includes dialogue instructions
        spy.reply_text = "What is truth?"
        dlg = await engine.generate_dialogue(scene, req.characters[0], "Opening line")
        assert spy.last_kwargs.get("temperature") == 0.9
        assert spy.last_kwargs.get("max_tokens") == 300
        assert "dialogue" in spy.last_prompt.lower()
        assert isinstance(dlg, str)

        # Evaluation → temperature 0.3, max_tokens 400; prompt includes metrics list
        spy.reply_text = "Narrative Coherence: 7/10 - OK"
        ev = await engine.evaluate_quality(scene["scene_description"])
        assert spy.last_kwargs.get("temperature") == 0.3
        assert spy.last_kwargs.get("max_tokens") == 400
        assert "Rate the following" in spy.last_prompt or "Story Content (Excerpt)" in spy.last_prompt
        assert "evaluation_text" in ev

        # Enhancement → temperature 0.6; prompt contains focus/evaluation
        spy.reply_text = "Enhanced scene"
        enhanced = await engine.enhance_content(scene["scene_description"], ev, "pacing and emotion")
        assert spy.last_kwargs.get("temperature") == 0.6
        assert "Focus area:" in spy.last_prompt or "Enhancement Focus:" in spy.last_prompt
        assert isinstance(enhanced, str)

    asyncio.run(run())
