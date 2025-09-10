import subprocess
import sys
from pathlib import Path


def _run_check(template: str, data: Path, golden: Path) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        '-m',
        'story_engine.poml.cli',
        template,
        '--data', str(data),
        '--format', 'text',
        '--check-golden', str(golden),
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_golden_base_character():
    root = Path(__file__).parent / 'golden'
    proc = _run_check('characters/base_character.poml', root / 'base_character.input.json', root / 'base_character.text.golden')
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_golden_character_response():
    root = Path(__file__).parent / 'golden'
    proc = _run_check('simulations/character_response.poml', root / 'character_response.input.json', root / 'character_response.text.golden')
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_golden_scene_crafting():
    root = Path(__file__).parent / 'golden'
    proc = _run_check('narrative/scene_crafting.poml', root / 'scene_crafting.input.json', root / 'scene_crafting.text.golden')
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_golden_world_state_brief():
    root = Path(__file__).parent / 'golden'
    proc = _run_check('meta/world_state_brief.poml', root / 'world_state_brief.input.json', root / 'world_state_brief.text.golden')
    assert proc.returncode == 0, proc.stderr or proc.stdout

