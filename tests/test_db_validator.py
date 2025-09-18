from story_engine.core.core.common.db_validator import validate_oracle_env


def test_validate_oracle_env_handles_missing_env(monkeypatch):
    monkeypatch.delenv("DB_WALLET_LOCATION", raising=False)
    monkeypatch.delenv("TNS_ADMIN", raising=False)
    result = validate_oracle_env()
    assert "errors" in result and result["errors"]

