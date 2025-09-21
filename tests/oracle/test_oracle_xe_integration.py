import os
import pytest


pytestmark = [pytest.mark.oracle]


@pytest.fixture(scope="module")
def db_env() -> dict:
    return {
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "dsn": os.getenv("DB_DSN") or os.getenv("DB_CONNECT_STRING"),
    }


def _env_ready(env: dict) -> bool:
    return bool(env.get("user") and env.get("password") and env.get("dsn"))


def test_oracle_connect_select_dual(db_env):
    if not _env_ready(db_env):
        pytest.skip("DB_USER/DB_PASSWORD/DB_DSN not set; skipping oracle integration test")

    oracledb = pytest.importorskip("oracledb")

    conn = oracledb.connect(
        user=db_env["user"], password=db_env["password"], dsn=db_env["dsn"]
    )
    try:
        cur = conn.cursor()
        cur.execute("SELECT 'OK' FROM DUAL")
        val = cur.fetchone()[0]
        assert val == "OK"
    finally:
        conn.close()

