from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional


def _env_truthy(val: Optional[str]) -> bool:
    if val is None:
        return False
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


class GenerationDBLogger:
    """Minimal Oracle logger for generations table.

    Safe to import when oracledb is not installed; only activates when
    DB_LOG_GENERATIONS is truthy and connection env vars are present.
    """

    def __init__(self) -> None:
        self.enabled = _env_truthy(os.environ.get("DB_LOG_GENERATIONS"))
        self._oracledb = None
        if not self.enabled:
            return
        # Basic env configuration
        self._dsn = os.environ.get("ORACLE_DSN")
        self._user = os.environ.get("ORACLE_USER")
        self._password = os.environ.get("ORACLE_PASSWORD")
        if not (self._dsn and self._user and self._password):
            # Missing credentials → disable
            self.enabled = False
            return
        try:
            import oracledb  # type: ignore

            self._oracledb = oracledb
        except Exception:
            # Library unavailable → disable silently
            self.enabled = False

    def _connect(self):
        assert self._oracledb is not None
        return self._oracledb.connect(
            user=self._user, password=self._password, dsn=self._dsn
        )

    def _ensure_provider(self, cur, name: str, ptype: str, endpoint: str) -> int:
        pid = cur.var(int)
        cur.execute(
            """
            MERGE INTO providers p
            USING (SELECT :name AS name FROM dual) s
            ON (p.name = s.name)
            WHEN MATCHED THEN UPDATE SET p.type = :ptype, p.endpoint = :endpoint
            WHEN NOT MATCHED THEN INSERT (name, type, endpoint) VALUES (:name, :ptype, :endpoint)
            RETURNING p.provider_id INTO :pid
            """,
            name=name,
            ptype=ptype,
            endpoint=endpoint,
            pid=pid,
        )
        return int(pid.getvalue())

    def _lookup_model_id(
        self, cur, provider_id: int, model_key: Optional[str]
    ) -> Optional[int]:
        if not model_key:
            return None
        cur.execute(
            "SELECT model_id FROM models WHERE provider_id = :pid AND model_key = :mk",
            pid=provider_id,
            mk=model_key,
        )
        row = cur.fetchone()
        return int(row[0]) if row else None

    def log_generation(
        self,
        *,
        provider_name: str,
        provider_type: str,
        provider_endpoint: str,
        prompt: str,
        system: Optional[str],
        request_params: Dict[str, Any],
        response_text: Optional[str],
        response_json: Optional[Dict[str, Any]],
        status: str,
        latency_ms: Optional[float],
        model_key: Optional[str],
    ) -> None:
        if not self.enabled:
            return
        try:
            conn = self._connect()
            try:
                cur = conn.cursor()
                provider_id = self._ensure_provider(
                    cur, provider_name, provider_type, provider_endpoint
                )
                model_id = self._lookup_model_id(cur, provider_id, model_key)

                # Build request/response JSON payloads
                req_json = {
                    "prompt": prompt,
                    **({"system": system} if system else {}),
                    "params": request_params or {},
                }
                started = time.time()
                finished = started
                if latency_ms is not None:
                    finished = started + (latency_ms / 1000.0)

                cur.execute(
                    """
                    INSERT INTO generations (
                      prompt_id, persona_id, provider_id, model_id, model_key,
                      request_json, response_text, response_json,
                      usage_prompt, usage_completion, cost_usd, latency_ms, status,
                      started_at, finished_at
                    ) VALUES (
                      NULL, NULL, :provider_id, :model_id, :model_key,
                      :request_json, :response_text, :response_json,
                      NULL, NULL, NULL, :latency_ms, :status,
                      TO_TIMESTAMP_TZ(:started, 'YYYY-MM-DD"T"HH24:MI:SS.FF TZH:TZM'),
                      TO_TIMESTAMP_TZ(:finished, 'YYYY-MM-DD"T"HH24:MI:SS.FF TZH:TZM')
                    )
                    """,
                    provider_id=provider_id,
                    model_id=model_id,
                    model_key=model_key,
                    request_json=json.dumps(req_json),
                    response_text=response_text,
                    response_json=json.dumps(response_json) if response_json else None,
                    latency_ms=float(latency_ms) if latency_ms is not None else None,
                    status=status,
                    started=time.strftime(
                        "%Y-%m-%dT%H:%M:%S.%f%z", time.localtime(started)
                    ),
                    finished=time.strftime(
                        "%Y-%m-%dT%H:%M:%S.%f%z", time.localtime(finished)
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception:
            # Swallow logging errors to avoid impacting generation
            return

    def log_event(
        self,
        *,
        kind: str,
        provider_name: str,
        provider_type: str,
        provider_endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        model_key: Optional[str] = None,
        run_id: Optional[int] = None,
    ) -> None:
        if not self.enabled:
            return
        try:
            conn = self._connect()
            try:
                cur = conn.cursor()
                provider_id = self._ensure_provider(
                    cur, provider_name, provider_type, provider_endpoint
                )
                model_id = self._lookup_model_id(cur, provider_id, model_key)
                cur.execute(
                    """
                    INSERT INTO metrics_events (
                      kind, run_id, provider_id, model_id, data, at
                    ) VALUES (
                      :kind, :run_id, :provider_id, :model_id, :data,
                      SYSTIMESTAMP
                    )
                    """,
                    kind=kind,
                    run_id=run_id,
                    provider_id=provider_id,
                    model_id=model_id,
                    data=json.dumps(data) if data else None,
                )
                conn.commit()
            finally:
                conn.close()
        except Exception:
            return
