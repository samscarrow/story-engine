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

    def __init__(self, use_poml: bool = True, orchestrator: Any = None):
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
        self.poml = StoryEnginePOMLAdapter()

    async def simulate(self, character: CharacterState, situations: List[str], runs_per: int = 3) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for s in situations:
            # Apply reviewer parameters if available
            rp = getattr(self, '_reviewer_params', {}) or {}
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
        prompt = self.poml.get_review_throughlines_prompt(character, situations, simulations)
        # Large budget for reviewer to consider many simulations
        resp = await self.orchestrator.generate(prompt, allow_fallback=True, temperature=0.3, max_tokens=1800)
        text = getattr(resp, 'text', '') or ''
        try:
            data = json.loads(text)
        except Exception:
            # fallback: try to extract JSON block
            import re
            m = re.search(r"\{[\s\S]*\}", text)
            data = json.loads(m.group(0)) if m else {"throughlines": []}
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
        prompt = adapter.get_quality_evaluation_prompt(meta_outline[:2000], [
            "Narrative Coherence",
            "Character Motivation",
            "Conflict Density",
            "Psychological Depth",
            "Thematic Clarity",
            "Originality",
            "Dramatic Momentum",
            "Overall Potential",
        ])
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

    @staticmethod
    def select_best_throughline(throughlines: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Pick best throughline using arc strength + evidence richness + conflict coverage."""
        if not throughlines:
            return None
        def score(t: Dict[str, Any]) -> float:
            strength = float(t.get('arc_strength', 0) or 0)
            ev = t.get('evidence', []) or []
            conflicts = set((t.get('conflicts') or []))
            return strength + 2.0 * len(ev) + 1.0 * len(conflicts)
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
