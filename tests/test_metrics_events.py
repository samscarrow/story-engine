import json
import logging

import pytest

from llm_observability import init_logging_from_env, metric_event, observe_metric


class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:  # type: ignore[override]
        self.records.append(record)


def _last_payload(handler: _CaptureHandler) -> dict:
    assert handler.records, "no metric logs captured"
    return json.loads(handler.records[-1].getMessage())


def test_metric_event_and_timer_shape(monkeypatch):
    # Force JSON logs
    monkeypatch.setenv("LOG_FORMAT", "json")
    monkeypatch.setenv("LOG_DEST", "stdout")
    monkeypatch.setenv("LOG_LEVEL", "INFO")

    init_logging_from_env(force=True)

    metrics_logger = logging.getLogger("metrics")
    handler = _CaptureHandler()
    metrics_logger.addHandler(handler)
    metrics_logger.setLevel(logging.INFO)

    try:
        # Counter-like metric
        handler.records.clear()
        metric_event("worker.plot.messages", value=1, type="counter", component="plot_worker")
        payload1 = _last_payload(handler)
        assert payload1["event"] == "metric"
        assert payload1["metric"] == "worker.plot.messages"
        assert payload1["type"] == "counter"
        assert payload1["value"] == 1

        # Timer
        handler.records.clear()
        observe_metric("db.oracle.connect_ms", 42.0, dsn="localhost/XEPDB1", pooled=False)
        payload2 = _last_payload(handler)
        assert payload2["event"] == "metric"
        assert payload2["metric"] == "db.oracle.connect_ms"
        assert payload2["unit"] == "ms"
        assert payload2["value"] >= 0
        assert payload2["dsn"] == "localhost/XEPDB1"
    finally:
        metrics_logger.removeHandler(handler)
