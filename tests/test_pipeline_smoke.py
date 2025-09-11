"""
Smoke test for NarrativePipeline wired with a stub orchestrator and POML enabled.
Validates that generate_with_llm returns a string and craft_scene produces a SceneDescriptor.
"""

import asyncio

from story_engine.core.story_engine.narrative_pipeline import NarrativePipeline
from story_engine.core.domain.models import SceneDescriptor


class StubResponse:
    def __init__(self, text: str):
        self.text = text
        self.metadata = {}
        self.timestamp = ""


class StubOrchestrator:
    async def generate(self, prompt: str, **kwargs):
        # Always return a canned response
        return StubResponse('{"dialogue": "Hi", "thought": "Okay", "action": "Nods"}')


def test_pipeline_generate_with_stub_orchestrator():
    orch = StubOrchestrator()
    pipeline = NarrativePipeline(orchestrator=orch, use_poml=True)

    async def run():
        out = await pipeline.generate_with_llm(
            "Test prompt", context="", temperature=0.8
        )
        assert isinstance(out, str)
        assert len(out) > 0

    asyncio.run(run())


def test_pipeline_craft_scene_shape():
    orch = StubOrchestrator()
    pipeline = NarrativePipeline(orchestrator=orch, use_poml=False)

    async def run():
        beat = {"id": 0, "name": "Setup", "purpose": "Establish normal", "tension": 0.2}
        characters = [{"id": "c1", "name": "Alice"}]
        scene = await pipeline.craft_scene(beat, characters)
        assert isinstance(scene, SceneDescriptor)
        assert isinstance(scene.situation, str)
        assert scene.name == "Setup"

    asyncio.run(run())
