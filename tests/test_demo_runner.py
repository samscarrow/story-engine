import json
import os
import re
import subprocess
from pathlib import Path


def run_demo_and_get_outdir(env: dict[str, str]) -> Path:
    cmd = [
        env.get("PYTHON") or os.environ.get("PYTHON") or str((Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python")),
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
    assert proc.returncode == 0, proc.stderr or proc.stdout
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
    assert (outdir / "config.snapshot.yaml").exists() or (outdir / "config.snapshot.json").exists()

    # Basic JSON structure checks
    story = json.loads((outdir / "story.json").read_text())
    assert "runs" in story and isinstance(story["runs"], list)
    assert story.get("character")

    metrics = json.loads((outdir / "metrics.json").read_text())
    for key in ["runs", "schema_valid", "ngram_repetition_3"]:
        assert key in metrics
