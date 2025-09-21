import json
import os
import re
import subprocess
import sys
from pathlib import Path


def resolve_python_exec(env: dict[str, str]) -> str:
    # 1) Respect explicit PYTHON in env or os.environ
    python_exec = env.get("PYTHON") or os.environ.get("PYTHON")
    if python_exec:
        return python_exec
    # 2) If VENV_PATH is provided (direnv-managed), use its python if present
    venv_path = env.get("VENV_PATH") or os.environ.get("VENV_PATH")
    if venv_path:
        candidate = Path(venv_path) / "bin" / "python"
        if candidate.exists():
            return str(candidate)
    # 3) Fall back to the current interpreter (sys.executable)
    return sys.executable


def run_demo_and_get_outdir(env: dict[str, str]) -> Path:
    python_exec = resolve_python_exec(env)
    cmd = [
        python_exec,
        "-m",
        "story_engine.scripts.run_demo",
        "--runs",
        "1",
        "--emphasis",
        "neutral",
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(Path(__file__).resolve().parents[1]),
        timeout=60,
    )
    assert proc.returncode == 0, f"Demo runner failed with return code {proc.returncode}.\nStdout:\n{proc.stdout}\nStderr:\n{proc.stderr}"
    m = re.search(r"Demo outputs written to: (dist/run-[0-9\-]+)", proc.stdout)
    assert m, f"Could not parse output dir from stdout: {proc.stdout}"
    outdir = Path(m.group(1))
    assert outdir.exists(), f"Output directory does not exist: {outdir}"
    return outdir


def test_demo_runner_smoke(tmp_path: Path):
    # Ensure the package is importable from src without installing
    env = os.environ.copy()
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    # Force mock/non-live behavior
    env.pop("STORY_ENGINE_LIVE", None)
    outdir = run_demo_and_get_outdir(env)

    # Required artifacts
    required = [
        "story.json",
        "metrics.json",
        "console.md",
        "env.capture",
    ]
    for name in required:
        path = outdir / name
        assert path.exists(), f"Missing artifact: {path}"
    # Config snapshot may be YAML or JSON depending on optional PyYAML
    assert (outdir / "config.snapshot.yaml").exists() or (
        outdir / "config.snapshot.json"
    ).exists()

    # Basic JSON structure checks
    story = json.loads((outdir / "story.json").read_text())
    assert "runs" in story and isinstance(story["runs"], list)
    assert story.get("character")

    metrics = json.loads((outdir / "metrics.json").read_text())
    for key in ["runs", "schema_valid", "ngram_repetition_3"]:
        assert key in metrics
