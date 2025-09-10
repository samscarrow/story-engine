from __future__ import annotations

from typing import Optional

from .interface import Message


def assert_type_matches_topic(topic: str, msg: Message) -> None:
    """Assert that the message.type equals the topic.

    Raises AssertionError with a helpful message if they differ.
    """
    if msg.type != topic:
        original = msg.headers.get("original_type") if isinstance(msg.headers, dict) else None
        raise AssertionError(
            f"Message.type '{msg.type}' does not match topic '{topic}'. "
            f"original_type={original!r} mismatch={msg.headers.get('type_mismatch') if isinstance(msg.headers, dict) else None}"
        )


def assert_correlation_and_causation(
    parent: Optional[Message], child: Message, *, require_causation: bool = True
) -> None:
    """Assert correlation/causation invariants for derived messages.

    - correlation_id should match parent's correlation_id when parent is present
    - causation_id should equal parent's id when require_causation=True
    """
    if parent is None:
        if not child.correlation_id:
            raise AssertionError("child message missing correlation_id")
        return

    expected_corr = parent.correlation_id or parent.id
    if child.correlation_id != expected_corr:
        raise AssertionError(
            f"correlation_id mismatch: child={child.correlation_id!r} expected={expected_corr!r}"
        )
    if require_causation and child.causation_id != parent.id:
        raise AssertionError(
            f"causation_id mismatch: child={child.causation_id!r} parent.id={parent.id!r}"
        )
