"""
Ensure OrchestratedStoryEngine can render POML prompts and call the orchestrator.
This uses a stub orchestrator and does not require real providers.
"""

import asyncio

from story_engine.core.story_engine.story_engine_orchestrated import (
    OrchestratedStoryEngine,
)


class StubResp:
    def __init__(self, text: str):
        self.text = text
        self.metadata = {}
        self.timestamp = ""


class StubOrchestrator:
    async def generate(self, prompt: str, **kwargs):
        # Ensure a non-empty prompt was produced
        assert isinstance(prompt, str) and len(prompt) > 0
        return StubResp("ok")


def test_orchestrated_scene_and_dialogue_with_poml():
    orch = StubOrchestrator()
    engine = OrchestratedStoryEngine(orchestrator=orch)
    characters = [{"id": "c1", "name": "Alice", "description": "hero"}]

    async def run():
        scene = await engine.generate_scene("A plot point", characters)
        assert scene["scene_description"] == "ok"
        dlg = await engine.generate_dialogue(scene, characters[0], "Opening line")
        assert dlg == "ok"
        enhanced = await engine.enhance_content(
            scene["scene_description"], {"evaluation_text": "Good pacing"}, "pacing"
        )
        assert enhanced == "ok"

    asyncio.run(run())
