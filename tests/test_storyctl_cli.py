from __future__ import annotations

from click.testing import CliRunner

from story_engine.tooling.storyctl.cli import app


def test_env_show_unknown_environment() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["env", "show", "--env", "does-not-exist"])
    assert result.exit_code == 1
    assert "Unknown environment 'does-not-exist'" in result.output


def test_run_aborts_when_required_env_missing(monkeypatch) -> None:
    runner = CliRunner()
    # Ensure required secret is absent
    monkeypatch.delenv("STAGING_DB_PASSWORD", raising=False)
    result = runner.invoke(app, ["run", "--env", "staging", "echo", "ok"])
    assert result.exit_code == 2
    assert "Missing required environment inputs for staging" in result.output
    assert "ok" not in result.output
