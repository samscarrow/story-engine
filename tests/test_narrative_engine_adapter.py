import asyncio

from story_engine.core.engine import (
    EngineContext,
    EngineOrchestrator,
    LocalRepos,
    FileArtifactRepo,
    NarrativePipelineEngine,
)
from story_engine.core.engine.determinism import RecordReplayAI, ReplayConfig


class _StubAI:
    async def generate(self, prompt: str, **kwargs):
        class R:
            def __init__(self, t: str):
                self.text = t
                self.metadata = {}
                self.timestamp = ""

        # Return a trivial JSON-like description for scene situation
        return R("Scene situation: a room with tense silence.")


def test_narrative_pipeline_engine_sample_plan(tmp_path):
    # Arrange
    ai = RecordReplayAI(_StubAI(), ReplayConfig(mode="record", dir=str(tmp_path / "rr")))
    ctx = EngineContext(ai=ai, artifacts=FileArtifactRepo(root=str(tmp_path / 'artifacts')))
    engine = NarrativePipelineEngine(use_poml=False)

    inputs = {
        "title": "Test Story",
        "premise": "A simple premise",
        "characters": [{"id": "c1", "name": "Alice"}],
        "num_beats": 3,
        "artifact_key": "demo_scene",
    }

    plan = engine.plan(inputs)

    # Act
    async def run():
        orch = EngineOrchestrator()
        res = await orch.run(plan, ctx)
        assert res.success
        scene = res.step_results.get("assemble_scene")
        assert isinstance(scene, dict)
        assert "situation" in scene and len(scene["situation"]) > 0

    asyncio.run(run())
