from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from story_engine.core.story_engine.world_state import WorldState
from story_engine.core.storage import get_database_connection, DatabaseConnection


class WorldStateManager:
    """DB-backed world state manager with targeted briefs."""

    def __init__(
        self,
        storage: Optional[DatabaseConnection] = None,
        workflow_name: str = "world_state",
    ):
        self.workflow_name = workflow_name
        self.storage = storage or self._build_db()

    def _build_db(self) -> DatabaseConnection:
        # Centralized env parsing with optional .env(.oracle) support
        try:
            from story_engine.core.core.common.settings import get_db_settings

            s = get_db_settings()
        except Exception:
            # Fallback to legacy behavior
            s = {
                "db_type": (os.getenv("DB_TYPE") or "postgresql").lower(),
                "user": os.getenv("DB_USER"),
            }
        db_type = s.get("db_type")
        if db_type == "oracle" and s.get("user"):
            return get_database_connection(
                "oracle",
                user=s.get("user"),
                password=s.get("password"),
                dsn=s.get("dsn"),
                wallet_location=s.get("wallet_location"),
                wallet_password=s.get("wallet_password"),
                use_pool=bool(s.get("use_pool", True)),
                pool_min=int(s.get("pool_min", 1)),
                pool_max=int(s.get("pool_max", 4)),
                pool_increment=int(s.get("pool_increment", 1)),
                pool_timeout=int(s.get("pool_timeout", 60)),
                wait_timeout=s.get("wait_timeout"),
                retry_attempts=int(s.get("retry_attempts", 3)),
                retry_backoff_seconds=float(s.get("retry_backoff_seconds", 1.0)),
                ping_on_connect=bool(s.get("ping_on_connect", True)),
            )
        if db_type == "postgresql" and s.get("user"):
            return get_database_connection(
                "postgresql",
                db_name=s.get("db_name", "story_db"),
                user=s.get("user"),
                password=s.get("password"),
                host=s.get("host", "127.0.0.1"),
                port=int(s.get("port", 5432)),
                sslmode=s.get("sslmode"),
                sslrootcert=s.get("sslrootcert"),
                sslcert=s.get("sslcert"),
                sslkey=s.get("sslkey"),
            )
        return get_database_connection(
            "sqlite", db_name=s.get("db_name", "workflow_outputs.db")
        )

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
        ws_dict = data.get("world_state") if isinstance(data, dict) else data
        if isinstance(ws_dict, dict):
            return WorldState.from_dict(ws_dict)
        return WorldState()

    def save(
        self, world_state: WorldState, meta: Optional[Dict[str, Any]] = None
    ) -> None:
        payload = {
            "world_state": world_state.to_dict(),
            "meta": meta or {},
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
            "facts": dict(ws.get("facts") or {}),
            "relationships": {},
            "timeline": list(ws.get("timeline") or [])[-last_n_events:],
            "availability": {},
            "locations": {},
            "props": dict(ws.get("props") or {}),
        }
        chars = set([c.lower() for c in (characters or [])])
        # Filter relationships touching target characters
        rels = ws.get("relationships") or {}
        if chars:
            for k, v in rels.items():
                try:
                    src, dst = k.split("->", 1)
                except ValueError:
                    continue
                if src.lower() in chars or dst.lower() in chars:
                    out["relationships"][k] = v
        else:
            out["relationships"] = rels
        # Filter availability for target characters
        av = ws.get("availability") or {}
        out["availability"] = {
            k: v for k, v in av.items() if not chars or k.lower() in chars
        }
        # Location focus if provided
        locs = ws.get("locations") or {}
        if location:
            if location in locs:
                out["locations"][location] = locs[location]
        else:
            out["locations"] = locs
        return out

    def pov_subset(
        self,
        world_state: WorldState,
        character_id: str,
        location: Optional[str] = None,
        last_n_events: int = 5,
        persona: Optional[Dict[str, Any]] = None,
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
            "facts": {},
            "relationships": {},
            "availability": {},
            "locations": {},
            "timeline": [],
            "props": {},
            "uncertain": [],
            "assumptions": [],
        }
        cid = (character_id or "").lower()

        def _lower_list(x: Any) -> List[str]:
            try:
                return [str(i).lower() for i in (x or [])]
            except Exception:
                return []

        # Facts
        for k, v in (ws.get("facts") or {}).items():
            try:
                if isinstance(v, dict):
                    public = bool(v.get("public", False))
                    visible = [x.lower() for x in (v.get("visible_to") or [])]
                    rumor = bool(v.get("rumor", False))
                    conf = float(v.get("confidence", 1.0))
                    if public or cid in visible:
                        out["facts"][k] = v.get("value", v)
                        if rumor or conf < 0.6:
                            out["uncertain"].append(f"{k}")
                else:
                    # If simple values, treat as public
                    out["facts"][k] = v
            except Exception:
                continue
        # Relationships touching character
        for rel_key, rel_val in (ws.get("relationships") or {}).items():
            try:
                src, dst = rel_key.split("->", 1)
                if src.lower() == cid or dst.lower() == cid:
                    out["relationships"][rel_key] = rel_val
            except Exception:
                continue
        # Availability — only POV character and directly related entities
        av = ws.get("availability") or {}
        related: set = set()
        for k in out["relationships"].keys():
            try:
                src, dst = k.split("->", 1)
                if src.lower() != cid:
                    related.add(src)
                if dst.lower() != cid:
                    related.add(dst)
            except Exception:
                continue
        for k, v in av.items():
            kl = k.lower()
            if kl == cid or kl in (x.lower() for x in related):
                out["availability"][k] = v
        # Locations
        locs = ws.get("locations") or {}
        if location and location in locs:
            out["locations"][location] = locs[location]

        # Timeline gating: include only events plausibly visible to the POV
        # Heuristics:
        # - Include if event.public is True
        # - Include if POV cid is in event.visible_to
        # - Include if cid appears as subject/actor/participants (best-effort)
        # - If no gating keys, include by default (assume public)
        tl = list(ws.get("timeline") or [])[-last_n_events:]
        for ev in tl:
            try:
                if not isinstance(ev, dict):
                    out["timeline"].append(ev)
                    continue
                public = bool(ev.get("public", False))
                visible = _lower_list(ev.get("visible_to"))
                subject = str(ev.get("subject", "")).lower()
                actors = _lower_list(ev.get("actors"))
                participants = _lower_list(ev.get("participants"))
                if (
                    public
                    or cid in visible
                    or cid == subject
                    or cid in actors
                    or cid in participants
                ):
                    out["timeline"].append(ev)
                else:
                    # If no explicit gating keys provided, treat as public
                    if not any(
                        k in ev
                        for k in (
                            "public",
                            "visible_to",
                            "subject",
                            "actors",
                            "participants",
                        )
                    ):
                        out["timeline"].append(ev)
            except Exception:
                continue

        # Persona-aware adjustments (optional)
        if persona and isinstance(persona, dict):
            try:
                blind = set([str(x) for x in (persona.get("blindspots") or [])])
                for b in blind:
                    if b in out["facts"]:
                        out["facts"].pop(b, None)
                overrides = persona.get("belief_overrides") or {}
                if isinstance(overrides, dict):
                    for k, v in overrides.items():
                        # Replace or insert as assumption if not present
                        if k in out["facts"]:
                            out["facts"][k] = v
                        else:
                            out["assumptions"].append(f"Believes {k} = {v}")
                addl_assumptions = persona.get("assumptions") or []
                if isinstance(addl_assumptions, list):
                    out["assumptions"].extend([str(x) for x in addl_assumptions])
            except Exception:
                pass

        return out
