"""
Tests that engines honor the POML feature flag from config.yaml
when an explicit argument is not provided.
"""

from core.story_engine.narrative_pipeline import NarrativePipeline
from core.common.config import load_config


def test_pipeline_uses_config_poml_flag_by_default():
    cfg = load_config("config.yaml")
    expected = bool(cfg.get("simulation", {}).get("use_poml", False))

    pipeline = NarrativePipeline(orchestrator=None, use_poml=None)

    assert pipeline.use_poml == expected

