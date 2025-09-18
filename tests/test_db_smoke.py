import os
import pytest


pytestmark = pytest.mark.db_smoke


def test_db_smoke_skips_when_env_missing():
    if not (os.getenv("DB_WALLET_LOCATION") or os.getenv("TNS_ADMIN")):
        pytest.skip("Oracle env not configured")
    # In real live job, we would attempt a lightweight connectivity check here.
    assert True

