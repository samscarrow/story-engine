import json
import os

import pytest


def _last_json_line(out: str) -> dict:
    lines = [ln for ln in out.strip().splitlines() if ln.strip()]
    assert lines, "no log lines captured"
    return json.loads(lines[-1])


def test_metric_event_and_timer_shape(capfd, monkeypatch):
    # Force JSON logs to stdout
    monkeypatch.setenv("LOG_FORMAT", "json")
    monkeypatch.setenv("LOG_DEST", "stdout")
    monkeypatch.setenv("LOG_LEVEL", "INFO")

    from story_engine.core.core.common.observability import (
        init_logging_from_env,
        metric_event,
        observe_metric,
    )

    init_logging_from_env()

    # Counter-like metric
    metric_event("worker.plot.messages", value=1, type="counter", component="plot_worker")
    out1, _ = capfd.readouterr()
    obj1 = _last_json_line(out1)
    payload1 = json.loads(obj1["message"]) if isinstance(obj1.get("message"), str) else obj1["message"]
    assert payload1["event"] == "metric"
    assert payload1["metric"] == "worker.plot.messages"
    assert payload1["type"] == "counter"
    assert payload1["value"] == 1

    # Timer
    observe_metric("db.oracle.connect_ms", 42.0, dsn="localhost/XEPDB1", pooled=False)
    out2, _ = capfd.readouterr()
    obj2 = _last_json_line(out2)
    payload2 = json.loads(obj2["message"]) if isinstance(obj2.get("message"), str) else obj2["message"]
    assert payload2["event"] == "metric"
    assert payload2["metric"] == "db.oracle.connect_ms"
    assert payload2["unit"] == "ms"
    assert payload2["value"] >= 0
    assert payload2["dsn"] == "localhost/XEPDB1"

