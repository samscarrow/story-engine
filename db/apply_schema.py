from __future__ import annotations

import os
import sys


SQLPLUS_DELIMITER = "/"


def read_statements(path: str):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Split on lines that contain only a single "/" (SQL*Plus block terminator)
    blocks = []
    current: list[str] = []
    for raw in lines:
        # Preserve lines as-is; we'll clean comments later
        if raw.strip() == SQLPLUS_DELIMITER:
            if current:
                blocks.append("".join(current))
                current = []
        else:
            current.append(raw)
    if current:
        blocks.append("".join(current))

    statements: list[str] = []
    for block in blocks:
        # Drop pure comment lines
        lines_no_comments = [
            line for line in block.splitlines() if not line.strip().startswith("--")
        ]
        cleaned_block = "\n".join(lines_no_comments)
        upper = cleaned_block.upper()

        # If the block contains a trigger, split out any preface DDL, then keep the trigger intact
        trig_markers = ("CREATE OR REPLACE TRIGGER", "CREATE TRIGGER")
        trig_pos = -1
        for marker in trig_markers:
            pos = upper.find(marker)
            if pos != -1:
                trig_pos = pos
                break

        if trig_pos != -1:
            pre = cleaned_block[:trig_pos].strip()
            trigger_stmt = cleaned_block[trig_pos:].strip()
            # First handle any preface DDL split by semicolons
            if pre:
                for part in pre.split(";"):
                    s = part.strip()
                    if s:
                        statements.append(s)
            # Then add trigger as a single statement
            if trigger_stmt:
                statements.append(trigger_stmt)
            continue

        # Regular DDL/DML: split on semicolons
        for part in cleaned_block.split(";"):
            s = part.strip()
            if s:
                statements.append(s)

    return statements


def main():
    # Ensure env vars are available even when not using direnv
    def _ensure_oracle_env() -> None:
        try:
            sys.path.append("src")
            from story_engine.core.common.dotenv_loader import load_dotenv_keys  # type: ignore

            load_dotenv_keys()
            load_dotenv_keys(path=".env.oracle")
        except Exception:
            pass

        if not os.environ.get("ORACLE_DSN"):
            if os.environ.get("ORACLE_TNS_ALIAS"):
                os.environ["ORACLE_DSN"] = os.environ["ORACLE_TNS_ALIAS"]
            elif os.environ.get("DB_DSN"):
                os.environ["ORACLE_DSN"] = os.environ["DB_DSN"]
            elif os.environ.get("ORACLE_CONNECT_STRING"):
                os.environ["ORACLE_DSN"] = os.environ["ORACLE_CONNECT_STRING"]

        if not os.environ.get("ORACLE_USER") and os.environ.get("DB_USER"):
            os.environ["ORACLE_USER"] = os.environ["DB_USER"]
        if not os.environ.get("ORACLE_PASSWORD") and os.environ.get("DB_PASSWORD"):
            os.environ["ORACLE_PASSWORD"] = os.environ["DB_PASSWORD"]

    _ensure_oracle_env()
    try:
        import oracledb  # type: ignore
    except Exception:
        print(
            "oracledb is required. Install with: uv add oracledb (or pip)",
            file=sys.stderr,
        )
        sys.exit(2)

    dsn = os.environ.get("ORACLE_DSN")
    user = os.environ.get("ORACLE_USER")
    password = os.environ.get("ORACLE_PASSWORD")
    if not all([dsn, user, password]):
        print("Set ORACLE_DSN, ORACLE_USER, ORACLE_PASSWORD in env", file=sys.stderr)
        sys.exit(2)

    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")

    # Optional: clean existing objects to avoid mismatched legacy definitions
    def _cleanup(cur):
        tables = [
            "METRICS_EVENTS",
            "CACHE_ENTRIES",
            "WORLD_RELATIONS",
            "WORLD_FACTS",
            "WORLD_ENTITIES",
            "DIALOGUE",
            "SCENES",
            "EVALUATIONS",
            "GENERATIONS",
            "PROMPTS",
            "PROMPT_SETS",
            "TEMPLATES",
            "PERSONAS",
            "MODELS",
            "PROVIDERS",
        ]
        sequences = [
            "METRICS_EVENTS_SEQ",
            "WORLD_RELATIONS_SEQ",
            "WORLD_FACTS_SEQ",
            "WORLD_ENTITIES_SEQ",
            "DIALOGUE_SEQ",
            "SCENES_SEQ",
            "EVALUATIONS_SEQ",
            "GENERATIONS_SEQ",
            "PROMPTS_SEQ",
            "PROMPT_SETS_SEQ",
            "TEMPLATES_SEQ",
            "PERSONAS_SEQ",
            "MODELS_SEQ",
            "PROVIDERS_SEQ",
        ]
        triggers = [
            "METRICS_EVENTS_BI",
            "WORLD_RELATIONS_BI",
            "WORLD_FACTS_BI",
            "WORLD_ENTITIES_BI",
            "DIALOGUE_BI",
            "SCENES_BI",
            "EVALUATIONS_BI",
            "GENERATIONS_BI",
            "PROMPTS_BI",
            "PROMPT_SETS_BI",
            "TEMPLATES_BI",
            "PERSONAS_BI",
            "MODELS_BI",
            "PROVIDERS_BI",
            "TRG_ENTITIES_UPD",
            "TRG_RELATIONS_UPD",
        ]

        # Drop indexes by querying USER_INDEXES for our known prefixes
        cur.execute(
            "SELECT index_name FROM user_indexes WHERE index_name LIKE 'UX_%' OR index_name LIKE 'IX_%'"
        )
        idxs = [r[0] for r in cur]
        for idx in idxs:
            try:
                cur.execute(f"DROP INDEX {idx}")
            except Exception:
                pass

        for trg in triggers:
            try:
                cur.execute(f"DROP TRIGGER {trg}")
            except Exception:
                pass
        for seq in sequences:
            try:
                cur.execute(f"DROP SEQUENCE {seq}")
            except Exception:
                pass
        for tbl in tables:
            try:
                cur.execute(f"DROP TABLE {tbl} CASCADE CONSTRAINTS")
            except Exception:
                pass

    stmts = read_statements(schema_path)
    print(f"Applying schema with {len(stmts)} statements...")

    conn = oracledb.connect(user=user, password=password, dsn=dsn)
    try:
        cur = conn.cursor()
        # Clean up any legacy/partial objects to ensure a consistent rebuild
        _cleanup(cur)

        for i, s in enumerate(stmts, 1):
            try:
                cur.execute(s)
            except Exception as e:
                # Do not silently accept mismatches; surface the exact failing DDL
                print(f"Statement {i} failed: {e}\n---\n{s}\n---", file=sys.stderr)
                raise
        conn.commit()
        print("Schema applied.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
