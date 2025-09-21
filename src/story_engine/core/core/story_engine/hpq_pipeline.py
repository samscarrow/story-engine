"""
High-Performance Quality (HPQ) Narrative Pipeline

Goals:
- Minimize average latency while improving output quality
- Use fast model(s) for planning and candidates, selectively escalate to 24B+
- Provide gating, reranking, and optional canary escalation controls

Design:
  Stage 0  Triage/Budgeting (lightweight; env-driven caps)
  Stage 1  Context & Situation via persona scene_architect (fast)
  Stage 2  Candidate Generation xN (fast)
  Stage 3  Automated Checks (quality_evaluation) → numeric score
  Stage 4  Rerank + Gate (threshold/hysteresis)
  Stage 5  Escalate + Finalize (24B+) or Keep Best Fast
  Stage 6  Postprocess + Metrics
"""

from __future__ import annotations

import asyncio
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import logging
from story_engine.core.core.cache.response_cache import ResponseCache
from llm_observability import get_logger, log_exception, observe_metric, inc_metric, ErrorCodes
from story_engine.core.core.orchestration.unified_llm_orchestrator import (
    UnifiedLLMOrchestrator,
    LLMPersona,
)
from story_engine.core.common.config import load_config


_obs = get_logger("hpq")
logger = logging.getLogger(__name__)


@dataclass
class HPQOptions:
    candidates: int = 3
    threshold_avg: float = 7.5  # gate to escalate to HQ (lower bound)
    threshold_high: float = 8.3  # hysteresis: above this, prefer fast keep
    max_tokens_fast: int = 600
    max_tokens_hq: int = 800
    temperature_fast: float = 0.7
    temperature_hq: float = 0.6
    canary_pct: float = 0.0  # 0.0..1.0 additional random escalation
    budget_ms: int = 0  # per-call budget hint to providers (0 = disabled)
    concurrency: int = 2  # bounded parallelism for candidate generation
    use_structured_scoring: bool = False  # switch to structured JSON scoring (2-pass)


def _truthy_env(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


class HPQPipeline:
    def __init__(self, orchestrator: Optional[UnifiedLLMOrchestrator] = None, opts: Optional[HPQOptions] = None) -> None:
        # Prefer unified orchestrator for POML personas
        self.unified = orchestrator or UnifiedLLMOrchestrator.from_env_and_config()
        self.opts = opts or HPQOptions()
        # TTL cache to avoid redoing identical expensive calls inside a run
        ttl = int(os.getenv("HPQ_CACHE_TTL", "1800") or 1800)
        self.cache = ResponseCache(ttl_seconds=ttl)
        # Optional Redis cache setup (best-effort)
        self._redis = None
        try:
            import redis.asyncio as _redis

            host = os.getenv("REDIS_HOST", "localhost")
            port = int(os.getenv("REDIS_PORT", "6379") or 6379)
            self._redis = _redis.Redis(host=host, port=port, decode_responses=True)
        except Exception:
            self._redis = None
        # Local chat-capable model cache: model_id -> (bool, expires_at)
        self._chat_capable: Dict[str, Tuple[bool, float]] = {}

        # Apply config.yaml overrides (best-effort)
        try:
            cfg = load_config("config.yaml") or {}
            hpq = cfg.get("hpq", {}) if isinstance(cfg, dict) else {}
            # Concurrency
            if "concurrency" in hpq and isinstance(hpq["concurrency"], int):
                self.opts.concurrency = hpq["concurrency"]
            # Structured scoring
            if "structured_scoring" in hpq:
                self.opts.use_structured_scoring = bool(hpq.get("structured_scoring"))
            # Canary
            if "canary_pct" in hpq:
                try:
                    self.opts.canary_pct = float(hpq.get("canary_pct", self.opts.canary_pct))
                except Exception:
                    pass
            # Thresholds
            if "threshold_low" in hpq:
                try:
                    self.opts.threshold_avg = float(hpq.get("threshold_low", self.opts.threshold_avg))
                except Exception:
                    pass
            if "threshold_high" in hpq:
                try:
                    self.opts.threshold_high = float(hpq.get("threshold_high", self.opts.threshold_high))
                except Exception:
                    pass
        except Exception:
            pass

    async def _list_models(self, prefer_small: bool) -> List[Dict[str, Any]]:
        try:
            return await self.unified.orchestrator.list_models_filtered(prefer_small=prefer_small)
        except Exception as e:
            logger.error(f"Error in pipeline: {e}")
            return []

    async def _select_fast_model(self) -> Optional[str]:
        # Priority: env override → small filtered list → None (auto)
        env_fast = os.getenv("LM_FAST_MODEL")
        if env_fast:
            return env_fast
        models = await self._list_models(prefer_small=True)
        return models[0].get("id") if models else None

    async def _select_hq_model(self) -> Optional[str]:
        # Priority: env override → heuristic pick of larger models → None (auto)
        env_hq = os.getenv("LM_HQ_MODEL")
        if env_hq:
            return env_hq
        candidates = await self._list_models(prefer_small=False)
        if not candidates:
            return None
        # Prefer obvious large variants
        preferred = [
            r"70b", r"72b", r"65b", r"40b", r"32b", r"24b", r"8x7b",
        ]
        ids = [c.get("id", "") for c in candidates]
        for pat in preferred:
            for mid in ids:
                if re.search(pat, mid, flags=re.IGNORECASE):
                    return mid
        # Fallback to last item (often bigger) or first if single
        return ids[-1] if ids else None

    async def _cache_get(self, key: str) -> Optional[str]:
        # Try Redis first
        if self._redis is not None:
            try:
                val = await self._redis.get(key)
                return val
            except Exception:
                pass
        # Fallback to in-memory cache interface
        return self.cache.get(key)  # type: ignore[return-value]

    async def _cache_set(self, key: str, value: str, ttl: int = 900) -> None:
        if self._redis is not None:
            try:
                await self._redis.set(key, value, ex=ttl)
                return
            except Exception:
                pass
        self.cache.set(key, value)

    async def _is_chat_capable(self, model_id: str) -> bool:
        """Probe whether a model id can handle chat completions.

        Caches positive/negative results for a short TTL to prevent repeated probes.
        """
        import time as _t

        now = _t.time()
        cached = self._chat_capable.get(model_id)
        if cached and cached[1] > now:
            return cached[0]

        # Also check Redis flag if present
        try:
            key = f"hpq:chat_capable:{model_id}"
            v = await self._cache_get(key)
            if v in ("1", "0"):
                ok = v == "1"
                self._chat_capable[model_id] = (ok, now + 600)
                return ok
        except Exception:
            pass

        ok = False
        try:
            # Tiny probe with strict budget
            resp = await self.unified.orchestrator.generate(
                prompt="ping",
                temperature=0.0,
                max_tokens=8,
                allow_fallback=False,
                model=model_id,
                budget_ms=min(self.opts.budget_ms or 8000, 8000) or 8000,
            )
            ok = bool(getattr(resp, "text", ""))
        except Exception:
            ok = False

        self._chat_capable[model_id] = (ok, now + 600)
        try:
            await self._cache_set(key, "1" if ok else "0", ttl=600)
        except Exception:
            pass
        return ok

    async def _select_valid_hq_model(self) -> Optional[str]:
        """Pick an HQ model and ensure it is chat-capable. Fallback across candidates."""
        preferred: List[str] = []
        env_hq = os.getenv("LM_HQ_MODEL")
        if env_hq:
            preferred.append(env_hq)
        # Add heuristic picks from available models
        avail = await self._list_models(prefer_small=False)
        ids = [m.get("id", "") for m in avail if isinstance(m, dict)]
        pats = [r"70b", r"72b", r"65b", r"40b", r"32b", r"24b", r"27b", r"8x7b"]
        for pat in pats:
            for mid in ids:
                if mid not in preferred and re.search(pat, mid, flags=re.IGNORECASE):
                    preferred.append(mid)

        for mid in preferred:
            if not mid:
                continue
            if await self._is_chat_capable(mid):
                return mid
        return None

    def _score_quality_lines(self, lines: List[str]) -> Tuple[float, Dict[str, int]]:
        # Expect lines like: "Narrative Coherence: 7/10 - ..."
        scores: Dict[str, int] = {}
        for line in lines:
            try:
                name, rest = line.split(":", 1)
                m = re.search(r"(\d+)/10", rest)
                if m:
                    val = int(m.group(1))
                    scores[name.strip()] = val
            except Exception:
                continue
        avg = sum(scores.values()) / len(scores) if scores else 0.0
        return avg, scores

    async def _evaluate_quality(self, story: str) -> Tuple[float, Dict[str, int]]:
        metrics = [
            "Narrative Coherence",
            "Character Development",
            "Pacing",
            "Emotional Impact",
            "Dialogue Quality",
            "Setting/Atmosphere",
            "Theme Integration",
            "Overall Engagement",
        ]
        # Choose structured scoring when enabled (two-pass), else text 8-line eval
        if self.opts.use_structured_scoring or _truthy_env("HPQ_STRUCTURED_SCORING", False):
            # Pass 1: freeform evaluation
            freeform = await self.unified.orchestrator.generate(
                prompt=self.unified.poml.render(
                    "narrative/quality_evaluation_freeform.poml", {"story": story}
                ),
                temperature=0.2,
                max_tokens=350,
                allow_fallback=True,
                budget_ms=self.opts.budget_ms,
            )
            ff_text = getattr(freeform, "text", "")
            # Pass 2: structure into JSON with numeric scores
            structured = await self.unified.orchestrator.generate(
                prompt=ff_text,
                temperature=0.0,
                max_tokens=300,
                allow_fallback=True,
                budget_ms=self.opts.budget_ms,
                # Hint: model choice not critical; smaller is fine
            )
            raw = getattr(structured, "text", "")
            try:
                import json as _json

                data = _json.loads(raw)
                scores_dict = data.get("scores", {}) if isinstance(data, dict) else {}
                # Normalize keys
                mapped = {}
                for k, v in scores_dict.items():
                    try:
                        mapped[str(k)] = int(v)
                    except Exception:
                        continue
                avg = sum(mapped.values()) / len(mapped) if mapped else 0.0
                return avg, mapped
            except Exception:
                # Fallback to text scorer if JSON parse fails
                pass

        # Text 8-line fallback
        resp = await self.unified.orchestrator.generate(
            prompt=self.unified.poml.render("narrative/quality_evaluation.poml", {"story": story, "metrics": metrics}),
            temperature=0.2,
            max_tokens=200,
            allow_fallback=True,
            budget_ms=self.opts.budget_ms,
        )
        text = getattr(resp, "text", "")
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        return self._score_quality_lines(lines[:8])

    async def _generate_candidate(self, situation: str, characters: List[Dict[str, Any]], *, model: Optional[str]) -> str:
        # Simple direct instruction prompt; keep output in a single narrative block.
        char_names = ", ".join([c.get("name", c.get("id", "")) for c in characters])
        prompt = (
            "Write a vivid cinematic scene based on the situation below.\n"
            "- Maintain continuity and consistent character voice.\n"
            "- Use concrete details and subtext; avoid exposition.\n"
            "- Write in a high historical drama tone (HBO 'Rome' energy).\n\n"
            f"Situation: {situation}\n"
            f"Characters: {char_names}\n\n"
            "Output: the scene text only (no headings)."
        )
        resp = await self.unified.orchestrator.generate(
            prompt=prompt,
            temperature=self.opts.temperature_fast,
            max_tokens=self.opts.max_tokens_fast,
            allow_fallback=True,
            model=model,
            budget_ms=self.opts.budget_ms,
        )
        return getattr(resp, "text", "")

    async def _finalize_hq(self, best_candidate: str, *, model: Optional[str]) -> str:
        system = (
            "You are a masterful line editor and ghostwriter for high historical drama."
        )
        prompt = (
            "Refine the following scene for maximal coherence, rhythm, and dramatic intensity while preserving content.\n"
            "- Keep voice consistent; cut filler; tighten pacing.\n"
            "- Use concrete sensory details and subtext.\n"
            "- Return only the improved scene text.\n\n"
            f"Scene:\n{best_candidate}"
        )
        resp = await self.unified.orchestrator.generate(
            prompt=prompt,
            system=system,
            temperature=self.opts.temperature_hq,
            max_tokens=self.opts.max_tokens_hq,
            allow_fallback=False,
            model=model,
            budget_ms=self.opts.budget_ms,
        )
        return getattr(resp, "text", "")

    async def craft_scene_hpq(
        self,
        beat: Dict[str, Any],
        characters: List[Dict[str, Any]],
        previous_context: str = "",
    ) -> Dict[str, Any]:
        """HPQ flow for a single scene: plan → candidates → check → finalize."""
        # Resolve models
        fast_model = await self._select_fast_model()
        hq_model = await self._select_hq_model()

        # Stage 1: Situation via persona (fast)
        scene_arch = await self.unified.generate_with_persona(
            LLMPersona.SCENE_ARCHITECT,
            {"beat": beat, "characters": characters, "previous_context": previous_context},
            temperature=0.6,
            max_tokens=300,
            allow_fallback=True,
        )
        situation = getattr(scene_arch, "text", "")

        # Stage 2/3: Candidates (bounded parallelism)
        n = max(1, int(self.opts.candidates))
        candidates: List[str] = [""] * n
        sem = asyncio.Semaphore(max(1, int(self.opts.concurrency)))

        async def _one(i: int):
            async with sem:
                try:
                    t0 = asyncio.get_event_loop().time()
                    text = await self._generate_candidate(situation, characters, model=fast_model)
                    elapsed_ms = int((asyncio.get_event_loop().time() - t0) * 1000)
                    try:
                        observe_metric("hpq.candidate_ms", elapsed_ms, idx=i)
                    except Exception:
                        pass
                    candidates[i] = text or ""
                except Exception as e:
                    log_exception(_obs, code=ErrorCodes.GEN_TIMEOUT, component="hpq.candidate", exc=e, idx=i)

        await asyncio.gather(*[_one(i) for i in range(n)])

        if not candidates:
            return {"situation": situation, "final": "", "candidates": [], "scores": [], "model_hq": hq_model, "model_fast": fast_model}

        # Stage 4: Checks + rerank
        scored: List[Tuple[float, int]] = []  # (avg, index)
        score_breakdown: List[Dict[str, int]] = []
        for idx, cand in enumerate(candidates):
            try:
                t0 = asyncio.get_event_loop().time()
                avg, details = await self._evaluate_quality(cand)
                scored.append((avg, idx))
                score_breakdown.append(details)
                try:
                    observe_metric("hpq.evaluate_ms", int((asyncio.get_event_loop().time() - t0) * 1000), idx=idx)
                except Exception:
                    pass
            except Exception as e:
                log_exception(_obs, code=ErrorCodes.GEN_PARSE_ERROR, component="hpq.evaluate", exc=e, idx=idx)
                scored.append((0.0, idx))
                score_breakdown.append({})

        scored.sort(key=lambda t: t[0], reverse=True)
        best_avg, best_idx = scored[0]
        best_candidate = candidates[best_idx]

        # Stage 5: Gating + optional canary
        force_24b = _truthy_env("HPQ_FORCE_24B", False)
        do_canary = False
        if self.opts.canary_pct > 0.0:
            try:
                do_canary = random.random() < float(self.opts.canary_pct)
            except Exception:
                do_canary = False
        # Hysteresis window: if clearly good, skip; if clearly low, escalate; else rely on canary/force
        escalate = force_24b or do_canary or (best_avg < float(self.opts.threshold_avg))
        if not force_24b and not do_canary and best_avg >= float(self.opts.threshold_high):
            escalate = False
        inc_metric("hpq.escalate", 1 if escalate else 0, reason=("force" if force_24b else ("canary" if do_canary else ("score" if best_avg < self.opts.threshold_avg else "skip"))))

        # Stage 6: Finalize (HQ when needed)
        final: str
        if escalate:
            # Validate HQ model capability (fallback across candidates)
            valid_hq = hq_model or await self._select_valid_hq_model()
            with_model = valid_hq or fast_model
        else:
            with_model = fast_model
        try:
            if escalate and with_model and with_model != fast_model:
                t0 = asyncio.get_event_loop().time()
                final = await self._finalize_hq(best_candidate, model=with_model)
                try:
                    observe_metric("hpq.finalize_ms", int((asyncio.get_event_loop().time() - t0) * 1000), model=with_model)
                except Exception:
                    pass
            else:
                final = best_candidate
        except Exception as e:
            log_exception(_obs, code=ErrorCodes.AI_LB_UNAVAILABLE, component="hpq.finalize", exc=e)
            final = best_candidate

        try:
            observe_metric("hpq.best_avg", best_avg, beat=beat.get("name"))
        except Exception:
            pass

        return {
            "situation": situation,
            "candidates": candidates,
            "scores": [s for s in score_breakdown],
            "best_avg": best_avg,
            "selected_idx": best_idx,
            "final": final,
            "model_fast": fast_model,
            "model_hq": hq_model,
            "used_model": with_model,
        }
