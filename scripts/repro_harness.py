#!/usr/bin/env python3
"""
Repro Harness — Re-run a demo scenario from saved artifacts.

Given a prior run directory (e.g., dist/run-20250913-113000), this tool will:
  - Load args from args.snapshot.json (if present)
  - Load config from config.snapshot.yaml/json
  - Invoke run_demo.py with the same CLI args and config

Usage:
  python scripts/repro_harness.py --from-dir dist/run-YYYYMMDD-HHMMSS [--dry-run] [--trace-id ID]
  python scripts/repro_harness.py --from-dir NEW --compare-to OLD [--compare-only] [--fail-on-regression] [--report-json PATH]

Notes:
  - This harness targets the demo pipeline for deterministic reproduction.
  - It does not replay exact LLM provider responses; it re-executes the flow
    with the same configuration and CLI parameters.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_args_snapshot(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "args.snapshot.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    # Fallback: minimal defaults
    return {
        "config": None,
        "use_poml": False,
        "live": False,
        "strict_persona": False,
        "persona_threshold": None,
        "profile": "pilate",
        "job_id": None,
        "situation": None,
        "emphasis": "neutral",
        "world_pov": None,
        "runs": 3,
        "strict_output": False,
        "iterative_world": False,
    }


def discover_config_snapshot(run_dir: Path) -> Optional[Path]:
    # Prefer YAML snapshot, then JSON
    yml = run_dir / "config.snapshot.yaml"
    if yml.exists():
        return yml
    js = run_dir / "config.snapshot.json"
    if js.exists():
        return js
    return None


def build_demo_cmd(args_snap: Dict[str, Any], cfg_path: Optional[Path]) -> List[str]:
    cmd = [sys.executable, "src/story_engine/scripts/run_demo.py"]
    if cfg_path is not None:
        cmd.extend(["--config", str(cfg_path)])
    # Map args snapshot back to CLI flags
    if bool(args_snap.get("use_poml")):
        cmd.append("--use-poml")
    if bool(args_snap.get("live")):
        cmd.append("--live")
    if bool(args_snap.get("strict_persona")):
        cmd.append("--strict-persona")
    if args_snap.get("persona_threshold") is not None:
        cmd.extend(["--persona-threshold", str(args_snap.get("persona_threshold"))])
    profile = args_snap.get("profile")
    if profile:
        cmd.extend(["--profile", str(profile)])
    job_id = args_snap.get("job_id")
    if job_id:
        cmd.extend(["--job-id", str(job_id)])
    situation = args_snap.get("situation")
    if situation:
        cmd.extend(["--situation", str(situation)])
    emphasis = args_snap.get("emphasis")
    if emphasis:
        cmd.extend(["--emphasis", str(emphasis)])
    world_pov = args_snap.get("world_pov")
    if world_pov:
        cmd.extend(["--world-pov", str(world_pov)])
    runs = args_snap.get("runs")
    if runs is not None:
        cmd.extend(["--runs", str(runs)])
    if bool(args_snap.get("strict_output")):
        cmd.append("--strict-output")
    if bool(args_snap.get("iterative_world")):
        cmd.append("--iterative-world")
    return cmd


def parse_outdir_from_output(text: str) -> Optional[str]:
    # The demo prints: "Demo outputs written to: <path>"
    m = re.search(r"Demo outputs written to:\s*(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else None


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        return None
    return None


def compare_artifacts(old_dir: Path, new_dir: Path) -> Tuple[bool, List[str]]:
    """Compare metrics and continuity for regressions.

    Returns: (ok, issues)
      - ok is False if regressions detected
      - issues is a list of human-readable messages
    """
    issues: List[str] = []
    ok = True

    old_metrics = _load_json(old_dir / "metrics.json") or {}
    new_metrics = _load_json(new_dir / "metrics.json") or {}
    old_cont = _load_json(old_dir / "continuity_report.json") or {}
    new_cont = _load_json(new_dir / "continuity_report.json") or {}

    # Boolean stability checks
    def flip_bad(key: str, old: Any, new: Any) -> Optional[str]:
        if isinstance(old, bool) and isinstance(new, bool):
            if old and not new:
                return f"{key} flipped from True to False"
        return None

    # Numeric regression checks (new > old)
    def inc_bad(key: str, old: Any, new: Any) -> Optional[str]:
        try:
            o = float(old)
            n = float(new)
            if n > o:
                return f"{key} increased: {o} -> {n}"
        except Exception:
            return None
        return None

    # Key checks in metrics
    for key in ("schema_valid", "continuity_ok"):
        msg = flip_bad(key, old_metrics.get(key), new_metrics.get(key))
        if msg:
            ok = False
            issues.append(msg)
    for key in ("degraded_runs", "continuity_violations"):
        msg = inc_bad(key, old_metrics.get(key, 0), new_metrics.get(key, 0))
        if msg:
            ok = False
            issues.append(msg)

    # Continuity report detail (optional): compare violations length
    try:
        o_v = len(old_cont.get("violations") or [])
        n_v = len(new_cont.get("violations") or [])
        if n_v > o_v:
            ok = False
            issues.append(f"continuity_report.violations increased: {o_v} -> {n_v}")
    except Exception:
        pass

    return ok, issues


def main() -> int:
    ap = argparse.ArgumentParser(description="Re-run demo from saved artifacts")
    ap.add_argument("--from-dir", required=True, help="Path to prior run directory")
    ap.add_argument("--dry-run", action="store_true", help="Print command only")
    ap.add_argument("--trace-id", default=None, help="Override TRACE_ID for logs")
    ap.add_argument("--compare-to", default=None, help="Baseline run directory to compare against")
    ap.add_argument("--compare-only", action="store_true", help="Only compare artifacts; do not re-run")
    ap.add_argument("--fail-on-regression", action="store_true", help="Exit non-zero if regressions detected")
    ap.add_argument("--report-json", default=None, help="Write comparison/report JSON to PATH")
    args = ap.parse_args()

    run_dir = Path(args.from_dir).resolve()
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return 2

    # Optional compare-only path
    if args.compare_only:
        if not args.compare_to:
            print("--compare-only requires --compare-to OLD_DIR", file=sys.stderr)
            return 2
        ok, issues = compare_artifacts(Path(args.compare_to).resolve(), run_dir)
        if args.report_json:
            try:
                report = {"ok": ok, "issues": issues, "baseline": str(Path(args.compare_to).resolve()), "target": str(run_dir)}
                Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
                Path(args.report_json).write_text(json.dumps(report, indent=2))
            except Exception:
                pass
        if issues:
            print("Regression report:")
            for i in issues:
                print(f" - {i}")
        return 0 if (ok or not args.fail_on_regression) else 3

    # Build command from snapshot and (optionally) re-run
    args_snap = load_args_snapshot(run_dir)
    cfg_path = discover_config_snapshot(run_dir)

    cmd = build_demo_cmd(args_snap, cfg_path)
    cmd_str = " ".join(shlex.quote(c) for c in cmd)
    print(f"Repro command: {cmd_str}")
    if args.dry_run:
        # If requested, perform comparison even in dry-run (using existing NEW dir)
        if args.compare_to:
            ok, issues = compare_artifacts(Path(args.compare_to).resolve(), run_dir)
            if args.report_json:
                try:
                    report = {"ok": ok, "issues": issues, "baseline": str(Path(args.compare_to).resolve()), "target": str(run_dir)}
                    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
                    Path(args.report_json).write_text(json.dumps(report, indent=2))
                except Exception:
                    pass
            if issues:
                print("Regression report (dry-run):")
                for i in issues:
                    print(f" - {i}")
            if args.fail_on_regression and not ok:
                return 3
        return 0

    env = os.environ.copy()
    if args.trace_id:
        env["TRACE_ID"] = str(args.trace_id)

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        print(f"Repro run failed with exit code {proc.returncode}", file=sys.stderr)
        return proc.returncode

    # Attempt to discover the new output directory from stdout
    new_out = parse_outdir_from_output(proc.stdout) or "(unknown)"
    print(proc.stdout)
    print(f"Repro run outputs: {new_out}")

    # If baseline provided, compare new outputs with baseline
    if args.compare_to and new_out != "(unknown)":
        ok, issues = compare_artifacts(Path(args.compare_to).resolve(), Path(new_out))
        if args.report_json:
            try:
                report = {"ok": ok, "issues": issues, "baseline": str(Path(args.compare_to).resolve()), "target": str(Path(new_out).resolve())}
                Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
                Path(args.report_json).write_text(json.dumps(report, indent=2))
            except Exception:
                pass
        if issues:
            print("Regression report:")
            for i in issues:
                print(f" - {i}")
        if args.fail_on_regression and not ok:
            return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
