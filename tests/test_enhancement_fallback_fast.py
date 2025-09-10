"""
Fast, isolated tests for enhancement fallbacks.

These tests avoid live providers and complete in under a second.
"""

import asyncio

from story_engine.core.story_engine.story_engine_orchestrated import OrchestratedStoryEngine


class FailingOrchestrator:
    async def generate(self, *args, **kwargs):
        raise Exception("HTTP 502: All upstream nodes failed.")


class TwoCallFailingOrchestrator:
    def __init__(self):
        self.calls = 0

    async def generate(self, *args, **kwargs):
        self.calls += 1
        # First call (Stage 1 freeform) succeeds with a minimal text
        if self.calls == 1:
            class R:
                text = "Freeform enhancement text"
                metadata = {}
                timestamp = ""
            return R()
        # Second call (Stage 2 structuring) fails
        raise Exception("HTTP 502: structuring failed")


def test_enhancement_fallback_stage1_failure():
    async def run():
        engine = OrchestratedStoryEngine(orchestrator=FailingOrchestrator(), use_poml=True)
        original = "A scene paragraph about Pilate in Jerusalem."
        out = await engine.enhance_content(original, {"evaluation_text": "ok"}, "pacing and emotion")
        assert isinstance(out, str) and len(out) > 0
        # With Stage 1 failure, fallback should include or equal the original
        assert out == original or original in out

    asyncio.run(run())


def test_enhancement_fallback_stage2_failure():
    async def run():
        engine = OrchestratedStoryEngine(orchestrator=TwoCallFailingOrchestrator(), use_poml=True)
        original = "A scene paragraph about Pilate in Jerusalem."
        out = await engine.enhance_content(original, {"evaluation_text": "ok"}, "pacing")
        assert isinstance(out, str) and len(out) > 0
        # Stage 2 fails, so we should get the freeform enhancement text from Stage 1
        assert out == "Freeform enhancement text"

    asyncio.run(run())

