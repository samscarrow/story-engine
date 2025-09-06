from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple

from core.character_engine.character_simulation_engine_v2 import (
    SimulationEngine,
    CharacterState,
    EmotionalState,
    CharacterMemory,
    RetryHandler,
)
from core.common.config import load_config
from core.orchestration.orchestrator_loader import create_orchestrator_from_yaml


class MetaNarrativePipeline:
    """Run multi-simulation → review → synthesis → screenplay drafting via POML personas."""

    def __init__(self, use_poml: bool = True, orchestrator: Any = None, target_metrics: Optional[List[str]] = None, weights: Optional[Dict[str, float]] = None, character_flags: Optional[Dict[str, Dict[str, Any]]] = None, use_iterative_persona: Optional[bool] = None):
        cfg = load_config("config.yaml")
        self.config = cfg
        self.orchestrator = orchestrator or create_orchestrator_from_yaml("config.yaml")
        self.engine = SimulationEngine(
            llm_provider=None,
            retry_handler=RetryHandler(max_retries=2, base_delay=0.6),
            config=cfg,
            orchestrator=self.orchestrator,
            use_poml=use_poml,
        )

        # POML adapter
        from poml.lib.poml_integration import StoryEnginePOMLAdapter
        self.poml = StoryEnginePOMLAdapter(runtime_flags=character_flags)
        # Preferences for biasing review/evaluation
        self.target_metrics: List[str] = list(target_metrics or [])
        self.criteria_weights: Dict[str, float] = dict(weights or {})
        # Iterative persona loop is on by default when strict persona is enabled unless explicitly disabled
        self.use_iterative_persona: bool = bool(use_iterative_persona if use_iterative_persona is not None else cfg.get('features', {}).get('strict_persona_mode', False))

    async def simulate(self, character: CharacterState, situations: List[str], runs_per: int = 3) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for s in situations:
            # Apply reviewer parameters if available
            rp = getattr(self, '_reviewer_params', {}) or {}
            if self.use_iterative_persona:
                # Use runs_per as iteration count for iterative persona refinement
                iters = int(runs_per or 3)
                window = int(rp.get('window', 3))
                final = await self.engine.run_iterative_simulation(
                    character,
                    s,
                    emphasis=(rp.get('emphases') or ["neutral"])[:1][0],
                    iterations=iters,
                    window=window,
                    max_tokens=rp.get('max_tokens'),
                )
                # Append only the final attempt to keep the shape compatible for throughline review
                if isinstance(final, dict) and final.get('final'):
                    results.append(final['final'])
                else:
                    # fallback in case of unexpected format
                    results.append(final)
            else:
                sims = await self.engine.run_multiple_simulations(
                    character,
                    s,
                    num_runs=runs_per,
                    emphases=rp.get('emphases'),
                    fixed_temperature=rp.get('temperature'),
                    max_tokens=rp.get('max_tokens')
                )
                results.extend(sims)
        return results

    async def review_throughlines(self, character: Dict[str, Any], situations: List[str], simulations: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt = self.poml.get_review_throughlines_prompt(
            character,
            situations,
            simulations,
            target_criteria=self.target_metrics or None,
            weights=self.criteria_weights or None,
        )
        # Larger budget for reviewer to consider many simulations and include scoring fields
        resp = await self.orchestrator.generate(
            prompt,
            allow_fallback=True,
            temperature=0.3,
            max_tokens=2200,
        )
        text = getattr(resp, 'text', '') or ''
        try:
            data = json.loads(text)
        except Exception:
            # fallback: try to extract JSON block
            import re
            m = re.search(r"\{[\s\S]*\}", text)
            data = json.loads(m.group(0)) if m else {"throughlines": []}
        # Ensure criteria_scores/weighted_score are present if preferences were provided
        return self._backfill_weighted_scores(data)

    def _backfill_weighted_scores(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            ths = data.get('throughlines') or []
            for t in ths:
                cs = t.get('criteria_scores') or {}
                # Ensure all target metrics exist
                for m in self.target_metrics or []:
                    cs.setdefault(m, 0)
                t['criteria_scores'] = cs
                # Compute weighted score if missing or zero
                ws = t.get('weighted_score')
                if ws in (None, 0, 0.0, '') and (self.criteria_weights or {}):
                    total = 0.0
                    for k, w in (self.criteria_weights or {}).items():
                        try:
                            total += float(cs.get(k, 0) or 0) * float(w)
                        except Exception:
                            continue
                    t['weighted_score'] = round(total, 2)
        except Exception:
            pass
        return data

    def apply_reviewer_params(self, params: Dict[str, Any]) -> None:
        """Apply reviewer-recommended parameters to future simulations."""
        self._reviewer_params = params or {}

    async def synthesize_meta(self, character: Dict[str, Any], throughline: Dict[str, Any]) -> str:
        prompt = self.poml.get_throughline_synthesis_prompt(character, throughline)
        resp = await self.orchestrator.generate(prompt, allow_fallback=True, temperature=0.6, max_tokens=900)
        return getattr(resp, 'text', '') or ''

    async def draft_screenplay(self, meta_outline: str, style: str = "HBO Rome") -> str:
        prompt = self.poml.get_screenplay_draft_prompt(meta_outline, style=style)
        resp = await self.orchestrator.generate(prompt, allow_fallback=True, temperature=0.8, max_tokens=1100)
        return getattr(resp, 'text', '') or ''

    async def evaluate_meta(self, meta_outline: str) -> Dict[str, Any]:
        # Reuse story evaluation to assess outline potential
        from poml.lib.poml_integration import StoryEnginePOMLAdapter
        adapter = self.poml
        metrics = self.target_metrics or [
            "Narrative Coherence",
            "Character Motivation",
            "Conflict Density",
            "Psychological Depth",
            "Thematic Clarity",
            "Originality",
            "Dramatic Momentum",
            "Overall Potential",
        ]
        prompt = adapter.get_quality_evaluation_prompt(meta_outline[:2000], metrics)
        resp = await self.orchestrator.generate(prompt, allow_fallback=True, temperature=0.3, max_tokens=500)
        text = getattr(resp, 'text', '') or ''
        # Parse into scores
        scores: Dict[str, float] = {}
        for line in text.splitlines():
            if ':' in line and '/' in line:
                try:
                    metric, rest = line.split(':', 1)
                    num = rest.strip().split('/')[0]
                    scores[metric.strip()] = float(num.strip())
                except Exception:
                    continue
        return {"evaluation_text": text, "scores": scores}

    async def enhance_meta(self, meta_outline: str, evaluation: Dict[str, Any]) -> str:
        # Enhance with focus on psychological backdrops, motivations, and conflicts
        enhanced = await self.orchestrator.generate(
            self.poml.get_enhancement_prompt(
                meta_outline,
                evaluation.get('evaluation_text', ''),
                focus="psychological backdrops, motivations, conflicts"
            ),
            allow_fallback=True,
            temperature=0.6,
            max_tokens=1000
        )
        return getattr(enhanced, 'text', '') or ''

    def select_best_throughline(self, throughlines: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Pick best throughline using arc strength + evidence richness + conflict coverage."""
        if not throughlines:
            return None
        # Prefer weighted score if reviewer provided it
        def score(t: Dict[str, Any]) -> float:
            if 'weighted_score' in t:
                try:
                    return float(t.get('weighted_score') or 0.0)
                except Exception:
                    pass
            # Otherwise compute ad-hoc, optionally factoring criteria_scores with weights
            base = float(t.get('arc_strength', 0) or 0)
            ev_count = len(t.get('evidence', []) or [])
            conflicts = len(set((t.get('conflicts') or [])))
            s = base + 2.0 * ev_count + 1.0 * conflicts
            cs = t.get('criteria_scores') or {}
            if cs and getattr(self, 'criteria_weights', None):
                try:
                    for k, w in self.criteria_weights.items():
                        s += float(w) * float(cs.get(k, 0) or 0)
                except Exception:
                    pass
            return s
        return max(throughlines, key=score)

    @staticmethod
    def character_from_dict(d: Dict[str, Any]) -> CharacterState:
        return CharacterState(
            id=d.get('id', 'char'),
            name=d.get('name', 'Character'),
            backstory=d.get('backstory', {}),
            traits=d.get('traits', []),
            values=d.get('values', []),
            fears=d.get('fears', []),
            desires=d.get('desires', []),
            emotional_state=EmotionalState(**d.get('emotional_state', {})) if d.get('emotional_state') else EmotionalState(),
            memory=CharacterMemory(recent_events=d.get('memory', {}).get('recent_events', [])),
            current_goal=d.get('current_goal'),
            internal_conflict=d.get('internal_conflict'),
        )
