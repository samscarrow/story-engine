import json
import subprocess
import sys
from pathlib import Path


def test_cli_smoke(tmp_path: Path):
    # Invoke CLI to render a simple template
    data = tmp_path / "data.json"
    data.write_text(
        json.dumps(
            {
                "beat": {"name": "Setup", "purpose": "Establish", "tension": 2},
                "characters": [{"name": "Alice", "role": "protagonist"}],
                "previous_context": "",
            }
        ),
        encoding="utf-8",
    )

    # Use module path execution
    cmd = [
        sys.executable,
        "-m",
        "story_engine.poml.cli",
        "narrative/scene_crafting.poml",
        "--data",
        str(data),
        "--format",
        "text",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode in (0, 1)
    assert proc.stdout.strip()
