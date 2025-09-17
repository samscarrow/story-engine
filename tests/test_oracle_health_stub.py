import os
import pytest


def have_oracle_env() -> bool:
    # Only run when explicitly opted-in and the environment looks healthy.
    try:
        from story_engine.core.storage.database import oracle_env_is_healthy
        return oracle_env_is_healthy(require_opt_in=True)
    except Exception:
        return False


@pytest.mark.skipif(not have_oracle_env(), reason="Oracle env not configured")
def test_oracle_connection_healthcheck():
    from story_engine.core.storage.database import OracleConnection

    conn = OracleConnection(
        user=os.getenv("DB_USER") or os.getenv("ORACLE_USER"),
        password=os.getenv("DB_PASSWORD") or os.getenv("ORACLE_PASSWORD"),
        dsn=os.getenv("DB_DSN") or os.getenv("ORACLE_DSN"),
        wallet_location=os.getenv("DB_WALLET_LOCATION") or os.getenv("ORACLE_WALLET_DIR"),
        use_pool=True,
        pool_min=1,
        pool_max=2,
        retry_attempts=2,
        retry_backoff_seconds=0.5,
    )
    conn.connect()
    assert conn.healthy() is True
    conn.disconnect()
