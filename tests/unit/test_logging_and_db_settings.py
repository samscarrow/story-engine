import json
import os
import logging
from story_engine.core.core.common.logging import configure_json_logging, JsonLogFormatter
from story_engine.core.core.common.settings import get_db_settings


def test_configure_json_logging_emits_json(monkeypatch, capsys):
    # Use a dedicated logger and handler to avoid interference
    configure_json_logging(logging.INFO)
    logger = logging.getLogger('test.logging.json')
    # Ensure a StreamHandler with our formatter exists on this logger
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    logger.addHandler(handler)
    logger.propagate = True
    logger.setLevel(logging.INFO)

    logger.info('hello', extra={'correlation_id':'abc123','provider':'OpenAI'})
    for h in logging.getLogger().handlers + logger.handlers:
        try:
            h.flush()
        except Exception:
            pass
    captured = capsys.readouterr()
    stream = captured.out.strip() or captured.err.strip()
    assert stream, 'no log output captured'
    payload = json.loads(stream.splitlines()[-1])
    assert payload['message'] == 'hello'
    assert payload['correlation_id'] == 'abc123'
    assert payload['provider'] == 'OpenAI'
    assert payload['level'] == 'INFO'


def test_db_settings_fallback_to_sqlite_when_oracle_unhealthy(monkeypatch):
    monkeypatch.setenv('DB_TYPE','oracle')
    monkeypatch.delenv('DB_REQUIRE_ORACLE', raising=False)
    for key in [
        'DB_USER', 'DB_PASSWORD', 'DB_DSN', 'DB_CONNECT_STRING', 'ORACLE_DSN',
        'ORACLE_USER', 'ORACLE_PASSWORD', 'DB_WALLET_LOCATION', 'ORACLE_WALLET_DIR', 'TNS_ADMIN'
    ]:
        monkeypatch.delenv(key, raising=False)
    s = get_db_settings()
    assert s['db_type'] == 'sqlite'
    assert 'db_name' in s
