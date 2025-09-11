import asyncio
from story_engine.core.story_engine.story_engine_orchestrated import (
    OrchestratedStoryEngine,
)


class FailingOrchestrator:
    async def generate(self, *args, **kwargs):
        raise Exception("HTTP 502: All upstream nodes failed.")


async def main():
    engine = OrchestratedStoryEngine(orchestrator=FailingOrchestrator(), use_poml=True)
    original = "A scene paragraph about Pilate in Jerusalem."
    enhanced = await engine.enhance_content(
        original, {"evaluation_text": "ok"}, "pacing and emotion"
    )
    assert isinstance(enhanced, str) and len(enhanced) > 0
    assert enhanced == original or original in enhanced
    print("Enhancement fallback OK:", repr(enhanced[:80]))


if __name__ == "__main__":
    asyncio.run(main())
