from __future__ import annotations

import time

from story_engine.core.core.messaging.interface import InMemoryBus, Message
from story_engine.core.core.contracts.plot import PlotRequest
from story_engine.core.core.messaging.testutils import (
    assert_correlation_and_causation,
    assert_type_matches_topic,
)


def test_inmemory_bus_routes_validation_errors_to_dlq_on_publish():
    bus = InMemoryBus()
    seen = {"dlq": 0, "main": 0}

    def dlq(msg: Message, _):
        seen["dlq"] += 1
        assert msg.type == "dlq.plot.request"
        assert "error" in msg.payload

    def handler(_msg: Message, _):  # should not be invoked for invalid
        seen["main"] += 1

    bus.subscribe("dlq.plot.request", dlq)
    bus.subscribe("plot.request", handler)

    # Missing required field 'constraints'
    invalid = Message(type="plot.request", payload={"job_id": "j1", "prompt": "p"})
    bus.publish("plot.request", invalid)

    # Give a tick to process synchronously
    time.sleep(0.01)
    assert seen["dlq"] == 1
    assert seen["main"] == 0


def test_inmemory_bus_normalizes_type_to_topic():
    bus = InMemoryBus()
    captured = {}

    def handler(msg: Message, _):
        captured["msg"] = msg

    bus.subscribe("plot.request", handler)

    valid = {"job_id": "j2", "prompt": "p", "constraints": {}}
    PlotRequest.validate(valid)
    bus.publish("plot.request", Message(type="wrong.type", payload=valid))

    msg = captured["msg"]
    assert_type_matches_topic("plot.request", msg)
    assert msg.headers.get("type_mismatch") is True
    assert msg.headers.get("original_type") == "wrong.type"


def test_testutils_correlation_and_causation():
    parent = Message(type="plot.request", payload={"job_id": "j3", "prompt": "p", "constraints": {}})
    child = Message(
        type="plot.done",
        payload={"job_id": "j3", "outline_id": "o1", "outline_ref": "r1"},
        correlation_id=parent.correlation_id or parent.id,
        causation_id=parent.id,
    )
    assert_correlation_and_causation(parent, child)
    assert_type_matches_topic("plot.done", child)

