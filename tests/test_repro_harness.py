import json
import os
import subprocess
import sys
from pathlib import Path


def write_snapshot(dirpath: Path, args: dict | None = None, metrics: dict | None = None, continuity: dict | None = None, config_yaml: str | None = None) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "args.snapshot.json").write_text(json.dumps(args or {"profile": "pilate", "runs": 1}))
    (dirpath / "metrics.json").write_text(json.dumps(metrics or {"runs": 1, "schema_valid": True, "degraded_runs": 0, "continuity_ok": True, "continuity_violations": 0}))
    (dirpath / "continuity_report.json").write_text(json.dumps(continuity or {"ok": True, "violations": []}))
    (dirpath / "config.snapshot.yaml").write_text(config_yaml or "simulation:\n  max_concurrent: 1\n")


def test_repro_harness_dry_run(tmp_path: Path):
    run_dir = tmp_path / "run"
    write_snapshot(run_dir)
    proc = subprocess.run(
        [sys.executable, "scripts/repro_harness.py", "--from-dir", str(run_dir), "--dry-run"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "run_demo.py" in proc.stdout


def test_repro_harness_compare_only_regression(tmp_path: Path):
    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"
    write_snapshot(old_dir, metrics={"runs": 1, "schema_valid": True, "degraded_runs": 0, "continuity_ok": True, "continuity_violations": 0}, continuity={"ok": True, "violations": []})
    write_snapshot(new_dir, metrics={"runs": 1, "schema_valid": True, "degraded_runs": 1, "continuity_ok": True, "continuity_violations": 1}, continuity={"ok": False, "violations": ["X"]})

    proc_ok = subprocess.run(
        [sys.executable, "scripts/repro_harness.py", "--from-dir", str(new_dir), "--compare-to", str(old_dir), "--compare-only"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc_ok.returncode == 0, proc_ok.stderr
    assert "Regression report" in proc_ok.stdout

    proc_fail = subprocess.run(
        [sys.executable, "scripts/repro_harness.py", "--from-dir", str(new_dir), "--compare-to", str(old_dir), "--compare-only", "--fail-on-regression"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc_fail.returncode == 3

