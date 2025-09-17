import os
import subprocess
import sys
import pytest


def have_oracle_env() -> bool:
    # Only run when explicitly opted-in and the environment looks healthy.
    try:
        from story_engine.core.storage.database import oracle_env_is_healthy
        return oracle_env_is_healthy(require_opt_in=True)
    except Exception:
        return False


@pytest.mark.skipif(not have_oracle_env(), reason="Oracle env not configured")
def test_healthcheck_script_runs():
    proc = subprocess.run(
        [sys.executable, "scripts/oracle_healthcheck.py", "--pool"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
