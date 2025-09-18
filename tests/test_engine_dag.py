import asyncio

from story_engine.core.engine import (
    EngineContext,
    EngineOrchestrator,
    Plan,
    Step,
    StepKind,
)
from story_engine.core.engine.determinism import RecordReplayAI, ReplayConfig


class _StubAI:
    async def generate(self, prompt: str, **kwargs):
        class R:
            def __init__(self, t: str):
                self.text = t
                self.metadata = {}
                self.timestamp = ""

        return R(f"echo:{prompt}")


def test_engine_orchestrator_executes_dag(tmp_path):
    # Build a small linear plan: t1 -> ai2 -> t3
    t1 = Step(
        key="t1",
        kind=StepKind.TRANSFORM,
        func=lambda ctx, p: {"value": 1},
    )
    ai2 = Step(
        key="ai2",
        kind=StepKind.AI_REQUEST,
        params={"prompt": "hello"},
        depends_on=["t1"],
    )
    t3 = Step(
        key="t3",
        kind=StepKind.TRANSFORM,
        func=lambda ctx, p: {"sum": 42},
        depends_on=["ai2"],
    )

    plan = Plan(steps={s.key: s for s in [t1, ai2, t3]}, roots=["t1"])

    # Configure record-replay to write under tmp_path
    rr = RecordReplayAI(_StubAI(), ReplayConfig(mode="record", dir=str(tmp_path / "replays")))
    ctx = EngineContext(ai=rr)

    orch = EngineOrchestrator()

    async def run():
        res = await orch.run(plan, ctx)
        assert res.success
        assert "ai2" in res.step_results
        ai_out = res.step_results["ai2"]
        # when recorded, generate returns the real object, but we normalize text if present
        text = ai_out.get("text") if isinstance(ai_out, dict) else getattr(ai_out, "text", "")
        assert "echo:hello" in text

    asyncio.run(run())

