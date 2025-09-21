import types
import builtins
import time as _time

import pytest


def make_stub_oracledb(sequence):
    """Create a stub oracledb module.

    sequence: iterable of ('ok'|Exception) controlling connect outcomes.
    """
    stub = types.SimpleNamespace()

    class Error(Exception):
        pass

    stub.Error = Error

    seq = list(sequence)

    def connect(**kwargs):
        if not seq:
            return types.SimpleNamespace(cursor=lambda: types.SimpleNamespace(
                execute=lambda *a, **k: None, fetchone=lambda: (1,), close=lambda: None
            ))
        nxt = seq.pop(0)
        if nxt == 'ok':
            return types.SimpleNamespace(cursor=lambda: types.SimpleNamespace(
                execute=lambda *a, **k: None, fetchone=lambda: (1,), close=lambda: None
            ))
        raise nxt

    def create_pool(**kwargs):
        # return a pool with acquire method delegating to connect()
        return types.SimpleNamespace(acquire=lambda: connect())

    # defaults namespace used by code (fetch_lobs)
    stub.defaults = types.SimpleNamespace(fetch_lobs=True)
    stub.connect = connect
    stub.create_pool = create_pool
    return stub


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(_time, "sleep", lambda *_: None)
    yield


def test_retryable_then_success(monkeypatch):
    from story_engine.core.core.storage import database as db

    # Inject stub oracledb with retryable errors then success
    e1 = Exception("ORA-12541: TNS: no listener")
    e2 = Exception("ORA-12537: TNS: connection closed")
    stub = make_stub_oracledb([e1, e2, 'ok'])
    monkeypatch.setattr(db, "oracledb", stub, raising=False)

    oc = db.OracleConnection(
        user="u", password="p", dsn="mainbase_high", use_pool=False,
        retry_attempts=5, retry_backoff_seconds=0.01, ping_on_connect=True
    )
    oc.connect()
    assert oc.conn is not None


def test_non_retryable_raises(monkeypatch):
    from story_engine.core.core.storage import database as db

    e = Exception("ORA-00942: table or view does not exist")
    stub = make_stub_oracledb([e])
    monkeypatch.setattr(db, "oracledb", stub, raising=False)

    oc = db.OracleConnection(
        user="u", password="p", dsn="mainbase_high", use_pool=False,
        retry_attempts=2, retry_backoff_seconds=0.01, ping_on_connect=True
    )
    with pytest.raises(Exception):
        oc.connect()

