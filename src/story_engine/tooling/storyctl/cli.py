"""storyctl – environment-aware CLI entry point."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Iterable, Optional

import click

from . import config
from .checks import CheckResult, run_preflight_checks


@dataclass
class CLIContext:
    registry: config.EnvironmentRegistry
    env_name: str


def _build_registry(env_dir: Optional[str]) -> config.EnvironmentRegistry:
    extra_path = env_dir or os.getenv("STORYCTL_ENV_DIR")
    if extra_path:
        return config.EnvironmentRegistry([config._default_env_dir(), extra_path])
    return config.EnvironmentRegistry()


@click.group()
@click.option("--env", "env_name", default=None, help="Environment name to use")
@click.option(
    "--env-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=str),
    default=None,
    help="Directory containing additional environment definitions",
)
@click.pass_context
def app(ctx: click.Context, env_name: Optional[str], env_dir: Optional[str]) -> None:
    """Story Engine control CLI."""

    registry = _build_registry(env_dir)
    resolved_env_name = env_name or os.getenv("STORY_ENV") or registry.default
    ctx.obj = CLIContext(registry=registry, env_name=resolved_env_name)


def _resolve_environment(
    ctx: CLIContext,
    override: Optional[str] = None,
) -> tuple[config.EnvironmentDefinition, config.EnvironmentResolution]:
    env_name = override or ctx.env_name
    definition = ctx.registry.get(env_name)
    resolution = definition.resolve()
    return definition, resolution


@app.group()
@click.pass_obj
def env(ctx: CLIContext) -> None:
    """Environment management commands."""

    ctx  # pragma: no cover - ensures ctx is referenced


@env.command("list")
@click.pass_obj
def env_list(ctx: CLIContext) -> None:
    """List available environments."""

    active = ctx.env_name
    for definition in ctx.registry.list():
        marker = "*" if definition.name == active else ""
        click.echo(f"{marker:1} {definition.name:15} {definition.description}")


@env.command("show")
@click.option("--env", "env_name", default=None, help="Environment name to show")
@click.option("--reveal", is_flag=True, help="Show sensitive values when present")
@click.pass_obj
def env_show(ctx: CLIContext, env_name: Optional[str], reveal: bool) -> None:
    """Show resolved environment variables."""

    definition, resolution = _resolve_environment(ctx, env_name)
    click.echo(f"Environment: {definition.name}")
    click.echo(definition.description or "(no description)")
    click.echo("""
Key                  Value                        Source
-------------------- --------------------------- ----------------
""".rstrip())

    for key in sorted(resolution.statuses.keys()):
        status = resolution.statuses[key]
        value = status.value
        if value is None:
            rendered = "<missing>"
        elif status.sensitive and not reveal:
            rendered = "<hidden>"
        else:
            rendered = value
        source = status.source or ""
        click.echo(f"{key:20} {rendered:27} {source}")

    if resolution.missing_required:
        missing = ", ".join(sorted(set(resolution.missing_required)))
        click.echo(f"Missing required environment variables: {missing}", err=True)


@env.command("export")
@click.option("--env", "env_name", default=None, help="Environment name to export")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["shell", "json"], case_sensitive=False),
    default="shell",
)
@click.pass_obj
def env_export(ctx: CLIContext, env_name: Optional[str], fmt: str) -> None:
    """Export resolved environment variables."""

    definition, resolution = _resolve_environment(ctx, env_name)
    if resolution.has_missing_required():
        missing = ", ".join(sorted(set(resolution.missing_required)))
        click.echo(
            f"warning: missing required variables for {definition.name}: {missing}",
            err=True,
        )

    if fmt.lower() == "json":
        click.echo(json.dumps(resolution.as_dict(), indent=2))
    else:
        for line in resolution.as_shell():
            click.echo(line)


@app.command()
@click.option("--env", "env_name", default=None, help="Environment to validate")
@click.option(
    "--check",
    "check_filters",
    multiple=True,
    help="Filter checks by name or type",
)
@click.option("--fail-fast/--no-fail-fast", default=True, help="Stop on first failure")
@click.option("--json", "json_output", is_flag=True, help="Emit JSON output")
@click.pass_obj
def check(
    ctx: CLIContext,
    env_name: Optional[str],
    check_filters: Iterable[str],
    fail_fast: bool,
    json_output: bool,
) -> None:
    """Run preflight checks for an environment."""

    definition, resolution = _resolve_environment(ctx, env_name)
    filters = {f.lower() for f in check_filters}

    if resolution.has_missing_required():
        missing = ", ".join(sorted(set(resolution.missing_required)))
        click.echo(
            f"Missing required environment inputs for {definition.name}: {missing}",
            err=True,
        )
        raise SystemExit(2)

    checks_to_run = definition.checks
    if filters:
        checks_to_run = [
            chk for chk in definition.checks if chk.name.lower() in filters or chk.check_type.lower() in filters
        ]
        if not checks_to_run:
            click.echo("No checks matched the provided filters", err=True)
            raise SystemExit(3)

    results = run_preflight_checks(
        definition,
        resolution,
        checks=checks_to_run,
        fail_fast=fail_fast,
    )

    if json_output:
        payload = [
            {
                "name": result.check.name,
                "type": result.check.check_type,
                "success": result.success,
                "optional": result.optional,
                "message": result.message,
                "details": result.details,
            }
            for result in results
        ]
        click.echo(json.dumps(payload, indent=2))
    else:
        for result in results:
            label = "OK" if result.success else ("WARN" if result.optional else "FAIL")
            click.echo(f"[{label}] {result.check.name}: {result.message}")

    if any((not result.success and not result.optional) for result in results):
        raise SystemExit(1)


@app.command()
@click.argument("command", nargs=-1, required=True)
@click.option("--env", "env_name", default=None, help="Environment context to apply")
@click.option("--dry-run", is_flag=True, help="Print command and exports without running")
@click.pass_obj
def run(ctx: CLIContext, command: tuple[str, ...], env_name: Optional[str], dry_run: bool) -> None:
    """Run a command with environment variables applied."""

    if not command:
        raise click.UsageError("COMMAND is required")

    definition, resolution = _resolve_environment(ctx, env_name)
    exports = resolution.as_shell()
    if dry_run:
        for line in exports:
            click.echo(line)
        click.echo("Command: " + " ".join(command))
        return

    env = os.environ.copy()
    env.update(resolution.as_dict())

    try:
        completed = subprocess.run(command, env=env, check=False)
    except FileNotFoundError as exc:  # pragma: no cover - depends on external binaries
        raise SystemExit(str(exc)) from exc
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    app(standalone_mode=True)


__all__ = ["app", "main"]
