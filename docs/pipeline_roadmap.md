Story Engine Pipeline – Enhancements Roadmap
===========================================

This document captures targeted upgrades to improve reliability (schema-true JSON), efficiency, and developer ergonomics across the two-stage POML pipeline and the orchestrated fallback path.

Objectives
----------
- Enforce JSON schema adherence with deterministic structuring.
- Reduce latency and cost via caching, prompt reuse, and bounded concurrency.
- Improve robustness with retries, repair, and context budgeting.
- Make behavior configurable via `config.yaml` and observable via enriched metadata.

High-Priority (Phase 1)
-----------------------
- JSON schema + validation/repair
  - Define Pydantic models for `PlotStructure`, `Scene`, `Dialogue`.
  - After Stage 2, parse → validate; on failure, attempt a single repair pass and re-validate.
  - Return typed objects to callers; keep raw text in meta for audits.
- Provider native structured output (when supported)
  - Plumb provider params (e.g., response_format=json_schema, function/tool calls) through the orchestrator.
  - Fall back to prompt-only structuring + repair when not available.
- Structuring prompt guardrails (POML)
  - Add: “Only output JSON. No prose. Start with { and end with }.”
  - Add stop sequences to block commentary; set temperature=0.0.
- POML engine reuse
  - Instantiate a single POMLEngine per adapter; memoize static system prompts (Stage 2 templates).

Reliability & Robustness (Phase 2)
----------------------------------
- Resilient generate wrapper
  - Central helper that retries once on empty/invalid output with lower temperature, shorter prompt, or JSON-clamped system.
  - Log attempt metadata; bubble up structured error if still failing.
- Token/context budgeting
  - Estimate tokens for prompt + system + max_tokens; truncate long inputs (e.g., previous context) with a summarizer when near limits.
- Deterministic structuring
  - Force temperature=0.0, fixed top_p, and seed when provider supports it.

Throughput & Efficiency (Phase 3)
---------------------------------
- Bounded concurrency for scenes
  - Fan-out Stage 1 scene freeform with asyncio.Semaphore(n); keep Stage 2 sequential per scene.
- Caching improvements
  - Canonical key: (provider, sha256(prompt), json.dumps(params, sort_keys=True)).
  - Optional on-disk cache (SQLite/JSONL) behind the in-memory TTL cache.
- Config strategy
  - Centralize defaults under narrative.defaults.{freeform,structuring}.
  - Component → provider routing under narrative.providers.{plot,scene,dialogue,evaluation,enhancement}.

Observability
-------------
- Enrich last_generation_meta and per-call metadata
  - Include: component, stage, attempt, prompt_hash, params_snapshot.
  - Optional: write failing outputs to a rotating logs/invalid_json/ for triage.

Testing & DX
------------
- Template render tests
  - Render all POML templates with minimal fixtures to ensure no missing variables and JSON guardrails are present.
- Contract tests for schemas
  - Feed adversarial freeform text into structuring; assert strict model validation.
- Fast pipeline smoke
  - Keep a mocked orchestrator to exercise end-to-end without network.

Implementation Sketches
-----------------------

1) Schema validation with repair (core/schema/models.py)
```
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Any
import json

class PlotBeat(BaseModel):
    name: str
    purpose: str
    tension: int = Field(ge=1, le=10)

class PlotStructure(BaseModel):
    structure_type: str
    beats: List[PlotBeat]

def validate_json_payload(text: str, model):
    def _loads(s: str) -> Any:
        return json.loads(s.strip())
    try:
        return model.model_validate(_loads(text))
    except (ValidationError, json.JSONDecodeError):
        try:
            from json_repair import repair_json  # optional dep
            fixed = repair_json(text)
            return model.model_validate(_loads(fixed))
        except Exception as e:
            raise ValueError(f"Schema validation failed: {e}")
```

2) Resilient generate wrapper
```
async def robust_generate(orchestrator, *, prompt, system=None, attempts=2, **params):
    last_exc = None
    for i in range(attempts):
        try:
            resp = await orchestrator.generate(prompt=prompt, system=system, **params)
            if getattr(resp, 'text', '').strip():
                return resp
        except Exception as e:
            last_exc = e
        # adjust params on retry
        params = {**params, 'temperature': 0.1, 'max_tokens': max(256, params.get('max_tokens', 512))}
        prompt = prompt[:4000]
    if last_exc:
        raise last_exc
    raise RuntimeError('Empty response after retries')
```

3) Canonical cache key (ResponseCache)
```
import hashlib, json

def make_key(provider: str, prompt: str, params: dict) -> str:
    p = provider or 'active'
    ph = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
    pj = json.dumps(params or {}, sort_keys=True, separators=(',', ':'))
    pjh = hashlib.sha256(pj.encode('utf-8')).hexdigest()
    return f"v1:{p}:{ph}:{pjh}"
```

4) Config additions (example snippet)
```
narrative:
  providers:
    plot: lmstudio
    scene: lmstudio
    dialogue: lmstudio
    evaluation: lmstudio
    enhancement: lmstudio
  defaults:
    freeform:
      temperature: 0.8
      max_tokens: 1500
      timeout: 180
    structuring:
      temperature: 0.0
      max_tokens: 4000
      timeout: 180
```

5) POML prompt hardening (example system text)
```
You are a formatter that outputs only JSON. No prose, no markdown.
Begin your response with '{' and end with '}'. If you are unsure, return {}.
```

Acceptance Checklist
--------------------
- Structured outputs validate against Pydantic models with zero false positives.
- Dialogue generation no longer returns empty text without at least one retry/repair attempt.
- Cache hit-rate increases and keys are stable across order of params.
- Scene generation parallelism configurable and safe under memory/CPU limits.
- Logs include component/stage/attempt with prompt hash for correlation.

Rollout Plan
------------
1. Implement schema + prompt guardrails under a feature flag.
2. Ship cache key change with auto-fallback to old keys for one release.
3. Introduce robust generate wrapper and provider-level JSON support.
4. Add bounded concurrency and context budgeting.
5. Turn on flags by default after bake-in and metrics review.

