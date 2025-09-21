"""Preflight check execution for storyctl."""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Dict, Iterable, List, Mapping, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from story_engine import cli as story_cli

from .config import EnvironmentDefinition, EnvironmentResolution, PreflightCheck


@dataclass
class CheckResult:
    """Outcome of a single preflight check."""

    check: PreflightCheck
    success: bool
    message: str
    details: Optional[Mapping[str, object]] = None

    @property
    def optional(self) -> bool:
        return self.check.optional


def run_preflight_checks(
    environment: EnvironmentDefinition,
    resolution: EnvironmentResolution,
    *,
    checks: Optional[Iterable[PreflightCheck]] = None,
    fail_fast: bool = True,
) -> List[CheckResult]:
    """Execute the configured preflight checks for an environment."""

    active_checks = list(checks) if checks is not None else list(environment.checks)
    results: List[CheckResult] = []
    for check in active_checks:
        result = _run_single_check(check, resolution)
        results.append(result)
        if not result.success and not result.optional and fail_fast:
            break
    return results


def _run_single_check(
    check: PreflightCheck, resolution: EnvironmentResolution
) -> CheckResult:
    if check.check_type == "db":
        return _run_db_check(check, resolution)
    if check.check_type == "http":
        return _run_http_check(check, resolution)
    if check.check_type == "command":
        return _run_command_check(check, resolution)
    if check.check_type == "json":
        return _run_json_check(check, resolution)
    return CheckResult(
        check=check, success=False, message=f"Unknown check type '{check.check_type}'"
    )


def _run_db_check(
    check: PreflightCheck, resolution: EnvironmentResolution
) -> CheckResult:
    buffer = io.StringIO()
    with temporary_env(resolution.values), redirect_stdout(buffer):
        rc = story_cli.cmd_db_health(argparse.Namespace())
    output = buffer.getvalue().strip()
    if not output:
        output = "Database health check executed"
    success = rc == 0
    return CheckResult(
        check=check,
        success=success,
        message=output,
        details={"returncode": rc},
    )


def _run_http_check(
    check: PreflightCheck, resolution: EnvironmentResolution
) -> CheckResult:
    params = check.params
    url_template = params.get("url")
    if not url_template:
        return CheckResult(
            check=check, success=False, message="Missing 'url' parameter"
        )

    url = _render_template(str(url_template), resolution)
    method = str(params.get("method", "GET")).upper()
    timeout = float(params.get("timeout", 5.0))
    headers = {str(k): str(v) for k, v in (params.get("headers") or {}).items()}

    request = Request(url, method=method, headers=headers)

    try:
        with urlopen(
            request, timeout=timeout
        ) as response:  # nosec B310 - controlled destinations
            status_code = int(response.status)
            body_preview = response.read(256).decode("utf-8", errors="ignore")
            success = 200 <= status_code < 400
            return CheckResult(
                check=check,
                success=success,
                message=f"HTTP {status_code} {url}",
                details={"status": status_code, "body": body_preview},
            )
    except HTTPError as exc:
        return CheckResult(
            check=check,
            success=False,
            message=f"HTTP {exc.code} {url}",
            details={"status": exc.code, "reason": str(exc.reason)},
        )
    except URLError as exc:
        return CheckResult(
            check=check,
            success=False,
            message=f"HTTP error {url}",
            details={"reason": str(exc.reason)},
        )


def _run_command_check(
    check: PreflightCheck, resolution: EnvironmentResolution
) -> CheckResult:
    params = check.params
    command = params.get("command")
    args = params.get("args")
    cwd = params.get("cwd")
    expected_rc = int(params.get("returncode", 0))
    capture_json = bool(params.get("capture_json", False))

    env = os.environ.copy()
    env.update(resolution.values)

    if command:
        # Prefer shell=False to reduce injection risk; if string, split to argv
        if isinstance(command, str):
            import shlex

            argv = shlex.split(command)
        else:
            argv = [str(x) for x in command]
        completed = subprocess.run(  # noqa: S603 - argv provided by repo config
            argv,
            shell=False,
            capture_output=True,
            text=True,
            cwd=cwd,
            env=env,
            check=False,
        )
    else:
        if not isinstance(args, (list, tuple)):
            return CheckResult(
                check=check,
                success=False,
                message="Command check requires 'command' or 'args'",
            )
        arg_list = [str(a) for a in args]
        completed = subprocess.run(  # noqa: S603 - args provided
            arg_list,
            capture_output=True,
            text=True,
            cwd=cwd,
            env=env,
            check=False,
        )

    success = completed.returncode == expected_rc
    message = (
        completed.stdout.strip()
        or completed.stderr.strip()
        or f"Exit {completed.returncode}"
    )
    details: Dict[str, object] = {
        "returncode": completed.returncode,
    }
    if capture_json and completed.stdout:
        try:
            details["json"] = json.loads(completed.stdout)
        except json.JSONDecodeError:
            pass

    return CheckResult(
        check=check,
        success=success,
        message=message,
        details=details,
    )


def _run_json_check(
    check: PreflightCheck, resolution: EnvironmentResolution
) -> CheckResult:
    params = check.params
    path_template = params.get("path")
    if not path_template:
        return CheckResult(
            check=check, success=False, message="Missing 'path' parameter"
        )
    path = os.path.expanduser(_render_template(str(path_template), resolution))
    if not os.path.exists(path):
        return CheckResult(
            check=check, success=False, message=f"File not found: {path}"
        )
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        return CheckResult(
            check=check, success=False, message=f"JSON load failed: {exc}"
        )
    keys = list(data)[:5] if isinstance(data, dict) else None
    return CheckResult(
        check=check,
        success=True,
        message=f"Loaded JSON with keys: {keys}" if keys else "Loaded JSON",
        details={"path": path},
    )


@contextmanager
def temporary_env(values: Mapping[str, str]):
    original: Dict[str, Optional[str]] = {}
    try:
        for key, value in values.items():
            original[key] = os.environ.get(key)
            os.environ[key] = value
        yield
    finally:
        for key, previous in original.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


def _render_template(template: str, resolution: EnvironmentResolution) -> str:
    context = dict(os.environ)
    context.update(resolution.values)
    try:
        return Template(template).safe_substitute(context)
    except Exception:
        return template
