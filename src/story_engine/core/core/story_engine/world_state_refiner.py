from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from story_engine.core.story_engine.world_state import WorldState
from story_engine.core.story_engine.world_state_manager import WorldStateManager


class WorldStateRefiner:
    """Refine and normalize world state via LLM (LMStudio) using POML prompts.

    Produces a patch that is applied conservatively (upserts-first) to avoid destructive edits.
    """

    def __init__(self, orchestrator: Any, poml_adapter: Any, manager: Optional[WorldStateManager] = None):
        self.orchestrator = orchestrator
        self.poml = poml_adapter
        self.manager = manager or WorldStateManager()

    async def refine(
        self,
        focus_characters: Optional[List[str]] = None,
        location: Optional[str] = None,
        last_n_events: int = 8,
        workflow_name: Optional[str] = None,
        mode: str = "json_patch",
        system_prompt: Optional[str] = None,
        output_poml_path: Optional[str] = None,
    ) -> WorldState:
        """Load latest world, request a refinement patch, apply, and persist.

        Returns the refined WorldState.
        """
        # Load current
        world: WorldState = self.manager.load_latest()
        ws_dict = world.to_dict()

        if mode == "poml_enhance":
            export_poml = self.poml.get_world_state_export_poml(ws_dict)
            resp = await self.orchestrator.generate(
                export_poml,
                system=(system_prompt or ""),
                allow_fallback=True,
                temperature=0.2,
                max_tokens=2000,
            )
            enhanced = getattr(resp, 'text', '') or ''
            # Optionally save the enhanced POML to a file
            if output_poml_path:
                try:
                    with open(output_poml_path, 'w', encoding='utf-8') as f:
                        f.write(enhanced)
                except Exception:
                    pass
            # In POML mode, we don't attempt to parse POML back; just persist original world for continuity
            # Optionally, we could still save a small meta record noting enhanced POML path
            meta = {"refined": True, "mode": "poml_enhance"}
            if output_poml_path:
                meta["poml_output"] = output_poml_path
            if workflow_name:
                old = self.manager.workflow_name
                try:
                    self.manager.workflow_name = workflow_name
                    self.manager.save(world, meta=meta)
                finally:
                    self.manager.workflow_name = old
            else:
                self.manager.save(world, meta=meta)
            return world
        else:
            # Default: JSON patch refinement
            prompt = self.poml.get_world_state_refinement_prompt(
                ws_dict,
                focus_characters=focus_characters or [],
                location=location or "",
                last_n_events=last_n_events,
            )
            resp = await self.orchestrator.generate(
                prompt,
                allow_fallback=True,
                temperature=0.3,
                max_tokens=1200,
            )
            text = getattr(resp, 'text', '') or ''
            patch: Dict[str, Any] = {}
            try:
                patch = json.loads(text)
            except Exception:
                import re
                m = re.search(r"\{[\s\S]*\}", text)
                if m:
                    try:
                        patch = json.loads(m.group(0))
                    except Exception:
                        patch = {}
            refined = self._apply_patch(world, patch)
            if workflow_name:
                old = self.manager.workflow_name
                try:
                    self.manager.workflow_name = workflow_name
                    self.manager.save(refined, meta={"refined": True, "mode": "json_patch"})
                finally:
                    self.manager.workflow_name = old
            else:
                self.manager.save(refined, meta={"refined": True, "mode": "json_patch"})
            return refined

    def _apply_patch(self, world: WorldState, patch: Dict[str, Any]) -> WorldState:
        """Apply upserts and safe deletes from a patch onto a copy of the world state.

        Patch schema (best-effort):
        {
          "upserts": {
            "facts": { key: {value, public, visible_to, rumor, confidence} | value },
            "relationships": {"A->B": {...}},
            "availability": {...},
            "locations": {...},
            "props": {...},
            "timeline": [ {time, desc, public, visible_to, subject, actors, participants} ]
          },
          "deletes": { "facts": [..], "relationships": [..] },
          "notes": "..."
        }
        """
        if not isinstance(patch, dict):
            return world
        base = world.to_dict()
        up = patch.get('upserts') or {}
        de = patch.get('deletes') or {}

        # Facts
        facts = base.get('facts') or {}
        for k, v in (up.get('facts') or {}).items():
            facts[str(k)] = v
        for k in (de.get('facts') or []):
            facts.pop(str(k), None)
        base['facts'] = facts

        # Relationships
        rels = base.get('relationships') or {}
        for k, v in (up.get('relationships') or {}).items():
            rels[str(k)] = v
        for k in (de.get('relationships') or []):
            rels.pop(str(k), None)
        base['relationships'] = rels

        # Availability/locations/props (overwrite keys)
        for section in ['availability', 'locations', 'props']:
            src = base.get(section) or {}
            for k, v in (up.get(section) or {}).items():
                src[str(k)] = v
            base[section] = src

        # Timeline: append-only (safe). If items include an "id" matching an existing, replace it.
        tl: List[Dict[str, Any]] = list(base.get('timeline') or [])
        for ev in (up.get('timeline') or []):
            if isinstance(ev, dict) and 'id' in ev:
                replaced = False
                for i, cur in enumerate(tl):
                    if isinstance(cur, dict) and cur.get('id') == ev.get('id'):
                        tl[i] = ev
                        replaced = True
                        break
                if not replaced:
                    tl.append(ev)
            else:
                tl.append(ev)
        base['timeline'] = tl

        return WorldState.from_dict(base)

