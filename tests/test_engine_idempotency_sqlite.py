import asyncio

from story_engine.core.engine import (
    EngineContext,
    EngineOrchestrator,
    FileArtifactRepo,
    NarrativePipelineEngine,
)
from story_engine.core.engine.repos import SQLiteJobRepo


class _CountingAI:
    def __init__(self):
        self.calls = 0

    async def generate(self, prompt: str, **kwargs):
        self.calls += 1
        class R:
            def __init__(self, t: str):
                self.text = t
                self.metadata = {}
                self.timestamp = ""
        return R("Scene situation: a quiet room.\nSound: wind\nAtmosphere: calm")


def test_idempotency_skips_ai_steps(tmp_path):
    ai = _CountingAI()
    jobs = SQLiteJobRepo(db_path=str(tmp_path / "engine.db"))
    artifacts = FileArtifactRepo(root=str(tmp_path / "artifacts"))
    engine = NarrativePipelineEngine(use_poml=False)
    inputs = {
        "title": "Idem Story",
        "premise": "Simple",
        "characters": [{"id": "c1", "name": "Alice"}],
        "artifact_key": "scene1",
    }

    async def run_once():
        ctx = EngineContext(ai=ai, artifacts=artifacts, jobs=jobs)
        plan = engine.plan(inputs)
        res = await EngineOrchestrator().run(plan, ctx)
        assert res.success
        return res

    # First run should invoke AI twice (situation + sensory)
    asyncio.run(run_once())
    first_calls = ai.calls
    assert first_calls >= 2

    # Second run should skip AI steps due to idempotency
    asyncio.run(run_once())
    second_calls = ai.calls
    assert second_calls == first_calls

