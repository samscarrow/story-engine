import json
import logging
import os
from story_engine.core.core.common.observability import init_logging_from_env, get_logger


def test_json_logging_emits_valid_json(capfd, monkeypatch):
    # Force JSON logs to stdout
    monkeypatch.setenv("LOG_FORMAT", "json")
    monkeypatch.setenv("LOG_DEST", "stdout")
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.delenv("TRACE_ID", raising=False)

    init_logging_from_env()
    log = get_logger("test.logger", job_id="j-123")
    log.info("hello world")

    out, _ = capfd.readouterr()
    # Last line should be JSON
    line = out.strip().splitlines()[-1]
    data = json.loads(line)
    assert data["message"] == "hello world"
    assert data["name"] == "test.logger"
    assert data.get("job_id") == "j-123"
    assert "trace_id" in data


def test_log_exception_includes_code_and_component(capfd, monkeypatch):
    from story_engine.core.core.common.observability import log_exception, ErrorCodes

    monkeypatch.setenv("LOG_FORMAT", "json")
    monkeypatch.setenv("LOG_DEST", "stdout")
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    from story_engine.core.core.common.observability import init_logging_from_env

    init_logging_from_env()
    base_logger = logging.getLogger("test.logger.exc")
    try:
        raise RuntimeError("boom")
    except Exception as e:
        log_exception(base_logger, code=ErrorCodes.GEN_TIMEOUT, component="unit", exc=e, job_id="j-999")

    out, _ = capfd.readouterr()
    line = out.strip().splitlines()[-1]
    data = json.loads(line)
    # message contains embedded JSON with event structure
    payload = json.loads(data["message"]) if isinstance(data.get("message"), str) else data["message"]
    assert payload["event"] == "error"
    assert payload["code"] == ErrorCodes.GEN_TIMEOUT
    assert payload["component"] == "unit"


def test_text_logging_writes_plain_lines(tmp_path, monkeypatch):
    logfile = tmp_path / "plain.log"
    monkeypatch.setenv("LOG_FORMAT", "text")
    monkeypatch.setenv("LOG_DEST", "file")
    monkeypatch.setenv("LOG_FILE_PATH", str(logfile))
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    from story_engine.core.core.common.observability import init_logging_from_env, get_logger

    init_logging_from_env()
    log = get_logger("plain.logger")
    log.info("hello-text")

    s = logfile.read_text(encoding="utf-8")
    assert "hello-text" in s


def test_json_logging_to_file(tmp_path, monkeypatch):
    logfile = tmp_path / "json.log"
    monkeypatch.setenv("LOG_FORMAT", "json")
    monkeypatch.setenv("LOG_DEST", "file")
    monkeypatch.setenv("LOG_FILE_PATH", str(logfile))
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    from story_engine.core.core.common.observability import init_logging_from_env, get_logger

    init_logging_from_env()
    log = get_logger("json.logger", job_id="job-file")
    log.info("hello-json")

    line = logfile.read_text(encoding="utf-8").strip().splitlines()[-1]
    obj = json.loads(line)
    assert obj["message"] == "hello-json"
    assert obj.get("job_id") == "job-file"
