import json
import logging
from llm_observability import get_logger


def test_json_logging_emits_valid_json(capfd, monkeypatch):
    # Force JSON logs to stdout
    monkeypatch.setenv("LOG_FORMAT", "json")
    monkeypatch.setenv("LOG_DEST", "stdout")
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.delenv("TRACE_ID", raising=False)

    log = get_logger("test.logger")
    log.info("hello world", extra={"job_id": "j-123"})

    out, _ = capfd.readouterr()
    # Last line should be JSON
    line = out.strip().splitlines()[-1]
    data = json.loads(line)
    assert data["message"] == "hello world"
    assert data["name"] == "test.logger"
    assert data.get("job_id") == "j-123"


def test_text_logging_writes_plain_lines(tmp_path, monkeypatch):
    logfile = tmp_path / "plain.log"
    monkeypatch.setenv("LOG_FORMAT", "text")
    monkeypatch.setenv("LOG_DEST", "file")
    monkeypatch.setenv("LOG_FILE_PATH", str(logfile))
    monkeypatch.setenv("LOG_LEVEL", "INFO")

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

    log = get_logger("json.logger")
    log.info("hello-json", extra={"job_id": "job-file"})

    line = logfile.read_text(encoding="utf-8").strip().splitlines()[-1]
    obj = json.loads(line)
    assert obj["message"] == "hello-json"
    assert obj.get("job_id") == "job-file"