import os

import pytest

from story_engine.tooling.storyctl.config import EnvironmentRegistry


@pytest.fixture
def registry() -> EnvironmentRegistry:
    return EnvironmentRegistry()


def test_local_environment_defaults(registry: EnvironmentRegistry) -> None:
    local = registry.get("local")
    resolution = local.resolve()
    assert resolution.values["DB_TYPE"] == "sqlite"
    assert resolution.values["LM_ENDPOINT"].startswith("http://localhost")
    assert not resolution.has_missing_required()


def test_staging_requires_password(registry: EnvironmentRegistry, monkeypatch: pytest.MonkeyPatch) -> None:
    staging = registry.get("staging")
    resolution_missing = staging.resolve(environ={"PLACEHOLDER": "1"})
    assert resolution_missing.has_missing_required()
    assert "STAGING_DB_PASSWORD" in resolution_missing.missing_required

    monkeypatch.setenv("STAGING_DB_PASSWORD", "top-secret")
    try:
        resolution = staging.resolve()
        assert not resolution.has_missing_required()
        assert resolution.values["DB_PASSWORD"] == "top-secret"
    finally:
        monkeypatch.delenv("STAGING_DB_PASSWORD", raising=False)


def test_inheritance_overrides_checks(registry: EnvironmentRegistry) -> None:
    base = registry.get("base")
    staging = registry.get("staging")

    base_check = next(chk for chk in base.checks if chk.name == "llm-endpoint")
    staging_check = next(chk for chk in staging.checks if chk.name == "llm-endpoint")

    assert base_check.params["url"] == "${LM_ENDPOINT}/v1/models"
    assert staging_check.params["url"] == "${AI_LB_ENDPOINT}/health"


def test_export_helpers(registry: EnvironmentRegistry) -> None:
    local = registry.get("local")
    resolution = local.resolve()
    exports = resolution.as_shell()
    assert any(line.startswith("export LM_ENDPOINT=") for line in exports)
    assert resolution.as_dict()["STORY_ENV"] == "local"
