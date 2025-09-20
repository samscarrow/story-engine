"""
Response normalization utilities for OpenAI-style chat/embeddings responses.

Goals:
- Always produce meaningful assistant text when upstreams vary in fields.
- Preserve auxiliary fields (reasoning, usage, headers) for observability.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def _first_non_empty(*vals: Optional[str]) -> str:
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

def _coerce_text(value: Any) -> str:
    """Convert structured content (str or list of segments) into plain text."""

    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        candidate = value.get("text") or value.get("content") or value.get("value")
        if isinstance(candidate, str):
            return candidate.strip()
        return ""
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                candidate = item.get("text") or item.get("content") or item.get("value")
                if isinstance(candidate, str):
                    parts.append(candidate)
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts).strip()
    return ""


def _extract_reasoning(message: Dict[str, Any]) -> str:
    options = [
        message.get("reasoning_content"),
        message.get("reasoning") or message.get("thoughts"),
    ]
    for opt in options:
        reason = _coerce_text(opt)
        if reason:
            return reason
    return ""





def extract_text_and_reasoning(message: Dict[str, Any]) -> tuple[str, str]:
    """Return (text, reasoning) pairs from an OpenAI message structure."""

    if not isinstance(message, dict):
        return "", ""
    content = _coerce_text(message.get("content"))
    reasoning = _extract_reasoning(message)
    return content, reasoning

def normalize_openai_chat(
    data: Dict[str, Any], headers: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Normalize an OpenAI-compatible chat completion response.

    Returns a dict with:
    - text: best-effort assistant content (never whitespace-only; may be empty if truly absent)
    - reasoning: extracted reasoning_content when present
    - meta: usage/model/headers for observability
    """
    headers = dict(headers or {})
    choices = data.get("choices") or []

    text: str = ""
    reasoning: str = ""

    # Prefer the first choice with content; fall back to reasoning_content; then legacy 'text'
    for ch in choices:
        if not isinstance(ch, dict):
            continue
        msg = ch.get("message") or {}
        content, reason = extract_text_and_reasoning(msg)
        if content:
            text = content
            reasoning = reason
            break
        if not text and reason:
            text = reason
            reasoning = reason
            break
        # Legacy non-chat completions sometimes use top-level 'text'
        t = _coerce_text(ch.get("text"))
        if t:
            text = t
            break

    # As a last resort, scan all choices for any non-empty field
    if not text:
        for ch in choices:
            if not isinstance(ch, dict):
                continue
            msg = ch.get("message") or {}
            content, reason = extract_text_and_reasoning(msg)
            cand = _first_non_empty(content, reason)
            if cand:
                text = cand
                reasoning = reason
                break
            fallback = _coerce_text(ch.get("text"))
            if fallback:
                text = fallback
                break

    # Effective model preference: header over body
    effective_model = headers.get("x-selected-model") or data.get("model")

    meta = {
        "usage": data.get("usage") or {},
        "model": data.get("model"),
        "effective_model": effective_model,
        "headers": {k.lower(): v for k, v in headers.items()},
    }

    return {"text": text or "", "reasoning": reasoning or "", "meta": meta}
