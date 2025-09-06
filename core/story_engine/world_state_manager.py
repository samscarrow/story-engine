from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from core.story_engine.world_state import WorldState
from core.storage import get_database_connection, DatabaseConnection


class WorldStateManager:
    """DB-backed world state manager with targeted briefs."""

    def __init__(self, storage: Optional[DatabaseConnection] = None, workflow_name: str = 'world_state'):
        self.workflow_name = workflow_name
        self.storage = storage or self._build_db()

    def _build_db(self) -> DatabaseConnection:
        db_user = os.getenv('DB_USER')
        if db_user:
            return get_database_connection(
                'postgresql',
                db_name=os.getenv('DB_NAME', 'story_db'),
                user=db_user,
                password=os.getenv('DB_PASSWORD'),
                host=os.getenv('DB_HOST', '127.0.0.1'),
                port=int(os.getenv('DB_PORT', '5432')),
                sslmode=os.getenv('DB_SSLMODE'),
                sslrootcert=os.getenv('DB_SSLROOTCERT'),
                sslcert=os.getenv('DB_SSLCERT'),
                sslkey=os.getenv('DB_SSLKEY'),
            )
        return get_database_connection('sqlite', db_name='workflow_outputs.db')

    def load_latest(self) -> WorldState:
        self.storage.connect()
        try:
            rows = self.storage.get_outputs(self.workflow_name) or []
        finally:
            self.storage.disconnect()
        # Take the last row if available
        if not rows:
            return WorldState()
        data = rows[-1]
        ws_dict = data.get('world_state') if isinstance(data, dict) else data
        if isinstance(ws_dict, dict):
            return WorldState.from_dict(ws_dict)
        return WorldState()

    def save(self, world_state: WorldState, meta: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            'world_state': world_state.to_dict(),
            'meta': meta or {},
        }
        self.storage.connect()
        try:
            self.storage.store_output(self.workflow_name, payload)
        finally:
            self.storage.disconnect()

    def targeted_subset(
        self,
        world_state: WorldState,
        characters: Optional[List[str]] = None,
        location: Optional[str] = None,
        last_n_events: int = 5,
    ) -> Dict[str, Any]:
        """Produce a filtered world dict focusing on characters/location and recent timeline."""
        ws = world_state.to_dict()
        out: Dict[str, Any] = {
            'facts': dict(ws.get('facts') or {}),
            'relationships': {},
            'timeline': list(ws.get('timeline') or [])[-last_n_events:],
            'availability': {},
            'locations': {},
            'props': dict(ws.get('props') or {}),
        }
        chars = set([c.lower() for c in (characters or [])])
        # Filter relationships touching target characters
        rels = ws.get('relationships') or {}
        if chars:
            for k, v in rels.items():
                try:
                    src, dst = k.split('->', 1)
                except ValueError:
                    continue
                if src.lower() in chars or dst.lower() in chars:
                    out['relationships'][k] = v
        else:
            out['relationships'] = rels
        # Filter availability for target characters
        av = ws.get('availability') or {}
        out['availability'] = {k: v for k, v in av.items() if not chars or k.lower() in chars}
        # Location focus if provided
        locs = ws.get('locations') or {}
        if location:
            if location in locs:
                out['locations'][location] = locs[location]
        else:
            out['locations'] = locs
        return out

    def pov_subset(
        self,
        world_state: WorldState,
        character_id: str,
        location: Optional[str] = None,
        last_n_events: int = 5,
    ) -> Dict[str, Any]:
        """Character-limited subset. Heuristics:
        - Include facts marked public=True
        - Include facts with visible_to containing character_id
        - Include relationships where character participates
        - Include availability for character_id + directly related entities
        - Collect uncertain items from facts with rumor=True or confidence<0.6
        """
        ws = world_state.to_dict()
        out = {
            'facts': {},
            'relationships': {},
            'availability': {},
            'locations': {},
            'timeline': list(ws.get('timeline') or [])[-last_n_events:],
            'props': {},
            'uncertain': [],
            'assumptions': [],
        }
        cid = (character_id or '').lower()
        # Facts
        for k, v in (ws.get('facts') or {}).items():
            try:
                if isinstance(v, dict):
                    public = bool(v.get('public', False))
                    visible = [x.lower() for x in (v.get('visible_to') or [])]
                    rumor = bool(v.get('rumor', False))
                    conf = float(v.get('confidence', 1.0))
                    if public or cid in visible:
                        out['facts'][k] = v.get('value', v)
                        if rumor or conf < 0.6:
                            out['uncertain'].append(f"{k}")
                else:
                    # If simple values, treat as public
                    out['facts'][k] = v
            except Exception:
                continue
        # Relationships touching character
        for rel_key, rel_val in (ws.get('relationships') or {}).items():
            try:
                src, dst = rel_key.split('->', 1)
                if src.lower() == cid or dst.lower() == cid:
                    out['relationships'][rel_key] = rel_val
            except Exception:
                continue
        # Availability
        av = ws.get('availability') or {}
        if cid in (x.lower() for x in av.keys()):
            # exact key may differ; include all and let prompt focus
            out['availability'] = av
        else:
            # include only the character if present
            for k, v in av.items():
                if k.lower() == cid:
                    out['availability'][k] = v
        # Locations
        locs = ws.get('locations') or {}
        if location and location in locs:
            out['locations'][location] = locs[location]
        return out
