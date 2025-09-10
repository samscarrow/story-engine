"""
Model filtering utilities for provider model lists.

These helpers normalize heterogeneous `/v1/models` responses and filter to
text-generation-capable models, excluding embeddings and speech (TTS/STT).

They are safe to use across the project wherever model lists are consumed.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


EXCLUDE_NAME_TOKENS = (
    # embeddings
    "embed",
    "embedding",
    "embeddings",
    # text-to-speech / audio
    "tts",
    "text-to-speech",
    "parler",
    "xtts",
    "elevenlabs",
    "audio",
    # speech-to-text
    "stt",
    "speech-to-text",
    "whisper",
    "asr",
)

SMALL_NAME_TOKENS = (
    "1b",
    "1.5b",
    "2b",
    "3b",
    "3.5b",
    "4b",
    "tiny",
    "mini",
    "small",
    "lite",
    "nano",
)


def _lower_str(x: Any) -> str:
    try:
        return str(x).lower()
    except Exception:
        return ""


def _get_model_id(m: Dict[str, Any]) -> str:
    return (
        m.get("id")
        or m.get("name")
        or m.get("model")
        or m.get("slug")
        or ""
    )


def _has_any_token(s: str, tokens: Iterable[str]) -> bool:
    return any(tok in s for tok in tokens)


def _listify(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return list(x)
    return [x]


def is_text_generation_model(model: Dict[str, Any]) -> bool:
    """Return True if the model appears to support text generation.

    Heuristics prioritize explicit metadata when available and fall back to
    conservative name-based filtering.
    """
    mid = _get_model_id(model)
    s = _lower_str(mid)

    # Fast exclude by common name tokens
    if _has_any_token(s, EXCLUDE_NAME_TOKENS):
        return False

    # Capabilities field (boolean flags or list of strings)
    caps = model.get("capabilities") or model.get("capability")
    if isinstance(caps, dict):
        # Known keys across vendors: embeddings/audio/speech/code/vision
        for k in ("embedding", "embeddings", "audio", "speech", "stt", "tts"):
            v = caps.get(k)
            if isinstance(v, bool) and v:
                # If a model is explicitly marked as embedding/audio-only, exclude.
                # We don't exclude vision, code, or json flags here.
                if k in ("embedding", "embeddings", "audio", "speech", "stt", "tts"):
                    return False
    elif isinstance(caps, (list, tuple, set)):
        caps_l = {_lower_str(c) for c in caps}
        if any(c in caps_l for c in ("embedding", "embeddings", "audio", "speech", "stt", "tts")):
            return False

    # Modalities field (string or list) e.g., ["text","vision"] or ["audio"]
    modalities = model.get("modalities") or model.get("modality")
    if modalities is not None:
        mods = {_lower_str(m) for m in _listify(modalities)}
        # If it has audio-only, exclude; if it includes text, allow
        if mods and "text" in mods:
            # still ensure it isn't explicitly speech-only (rarely both present)
            if not ("audio" in mods or "speech" in mods):
                return True
        if mods and ("audio" in mods or "speech" in mods):
            return False

    # Type/task/pipeline tags used by various ecosystems
    for key in ("type", "task", "pipeline_tag", "category"):
        val = _lower_str(model.get(key))
        if val:
            if any(tok in val for tok in ("embed", "embedding", "tts", "stt", "audio", "asr")):
                return False
            if any(tok in val for tok in ("text", "chat", "generation")):
                return True

    # If no explicit positive evidence but no exclusion tokens either, assume text-capable.
    return True


def filter_models(
    models: Iterable[Dict[str, Any]],
    *,
    require_text: bool = True,
    exclude_embeddings: bool = True,
    exclude_audio: bool = True,
    prefer_small: bool = True,
) -> List[Dict[str, Any]]:
    """Filter and optionally re-order model dicts.

    - require_text: keep only text-generation models (via is_text_generation_model)
    - exclude_embeddings/audio: kept for API clarity (redundant when require_text=True)
    - prefer_small: stable-stable sort to push smaller models earlier using name hints
    """
    models_list = list(models or [])
    if require_text:
        models_list = [m for m in models_list if is_text_generation_model(m)]

    if not prefer_small or not models_list:
        return models_list

    def small_score(m: Dict[str, Any]) -> int:
        s = _lower_str(_get_model_id(m))
        # Lower score means earlier in sort
        return 0 if _has_any_token(s, SMALL_NAME_TOKENS) else 1

    # Stable sort keeps original provider order within buckets
    return sorted(models_list, key=small_score)


def choose_first_id(models: Iterable[Dict[str, Any]]) -> Optional[str]:
    """Return the first model identifier from a list of model dicts."""
    for m in models:
        mid = _get_model_id(m)
        if mid:
            return str(mid)
    return None

