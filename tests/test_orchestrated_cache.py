"""
Test that OrchestratedStoryEngine caches repeated generation calls.
"""

import asyncio

from story_engine.core.story_engine.story_engine_orchestrated import OrchestratedStoryEngine, StoryComponent


class StubResp:
    def __init__(self, text):
        self.text = text
        self.metadata = {}
        self.timestamp = ""


class CountingOrchestrator:
    def __init__(self):
        self.calls = 0

    async def generate(self, prompt: str, **kwargs):
        self.calls += 1
        return StubResp("ok")


def test_orchestrated_caches_by_prompt_and_params():
    orch = CountingOrchestrator()
    engine = OrchestratedStoryEngine(orchestrator=orch)

    async def run():
        text1 = await engine.generate_component(StoryComponent.SCENE_DETAILS, "prompt A", temperature=0.8)
        text2 = await engine.generate_component(StoryComponent.SCENE_DETAILS, "prompt A", temperature=0.8)
        assert text1 == text2 == "ok"
        # Only first call should hit orchestrator due to cache
        assert orch.calls == 1

    asyncio.run(run())

