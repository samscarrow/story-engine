from __future__ import annotations

from story_engine.core.messaging.interface import InMemoryBus, Message
from story_engine.core.contracts.plot import PlotRequest


def test_plot_request_to_done_inmemory_bus():
    bus = InMemoryBus()
    done_messages = []

    def handle_plot_request(msg: Message, pub):
        req = PlotRequest.validate(msg.payload)
        # Respond with a minimal done message
        done = Message(
            type="plot.done",
            payload={
                "job_id": req.job_id,
                "outline_id": "outline-1",
                "outline_ref": "outline:outline-1",
            },
            correlation_id=req.job_id,
            causation_id=msg.id,
        )
        pub.publish("plot.done", done)

    def handle_plot_done(msg: Message, _pub):
        done_messages.append(msg)

    bus.subscribe("plot.request", handle_plot_request)
    bus.subscribe("plot.done", handle_plot_done)

    req = Message(type="plot.request", payload={"job_id": "j1", "prompt": "hello", "constraints": {}})
    bus.publish("plot.request", req)

    assert len(done_messages) == 1
    assert done_messages[0].payload["job_id"] == "j1"
    assert done_messages[0].payload["outline_ref"].startswith("outline:")
