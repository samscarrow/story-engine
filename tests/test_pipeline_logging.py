"""
Smoke tests to ensure pipeline logging instrumentation does not raise and the
pipeline surfaces basic outputs while passing context.
"""

import asyncio
from story_engine.core.core.story_engine.narrative_pipeline import NarrativePipeline


class StubResponse:
    def __init__(self, text: str):
        self.text = text
        self.metadata = {}
        self.timestamp = ""


class StubOrchestrator:
    async def generate(self, prompt: str, **kwargs):
        return StubResponse('{"dialogue": "Hi", "thought": "Okay", "action": "Nods"}')


def test_pipeline_logging_context():
    orch = StubOrchestrator()
    pipeline = NarrativePipeline(orchestrator=orch, use_poml=False, job_id="job-ctx-1")

    async def run():
        # generate_with_llm directly
        out = await pipeline.generate_with_llm("hello", context="world", temperature=0.7, context_extra={"beat": "Setup"})
        assert isinstance(out, str)

        # craft a minimal scene
        beat = {"id": 0, "name": "Setup", "purpose": "Establish normal", "tension": 0.2}
        characters = [{"id": "c1", "name": "Alice"}]
        scene = await pipeline.craft_scene(beat, characters)
        assert scene.name == "Setup"

    asyncio.run(run())

