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


def normalize_openai_chat(data: Dict[str, Any], headers: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
        if isinstance(msg, dict):
            c = msg.get("content")
            r = msg.get("reasoning_content") or msg.get("reasoning")
            if isinstance(c, str) and c.strip():
                text = c.strip()
                reasoning = _first_non_empty(r)
                break
            if not text:
                # Stash reasoning as candidate text if content is empty
                rc = _first_non_empty(r)
                if rc:
                    text = rc
                    reasoning = rc
                    break
        # Legacy non-chat completions sometimes use top-level 'text'
        t = ch.get("text")
        if isinstance(t, str) and t.strip():
            text = t.strip()
            break

    # As a last resort, scan all choices for any non-empty field
    if not text:
        for ch in choices:
            if not isinstance(ch, dict):
                continue
            msg = ch.get("message") or {}
            if isinstance(msg, dict):
                c = msg.get("content")
                r = msg.get("reasoning_content") or msg.get("reasoning")
                cand = _first_non_empty(c, r)
                if cand:
                    text = cand
                    reasoning = _first_non_empty(r)
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

