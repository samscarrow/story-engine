import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make sure package imports work if run from a checkout
try:
    from story_engine.core.core.common.observability import (
        init_logging_from_env,
        get_logger,
    )
    from story_engine.core.core.common.settings import get_db_settings
    from story_engine.core.core.storage.database import get_database_connection
    from story_engine.core.core.orchestration.llm_orchestrator import (
        LLMOrchestrator,
        LLMConfig,
        ModelProvider,
    )
except Exception:  # pragma: no cover - fall back to relative imports
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from story_engine.core.core.common.observability import (
        init_logging_from_env,
        get_logger,
    )
    from story_engine.core.core.common.settings import get_db_settings
    from story_engine.core.core.storage.database import get_database_connection
    from story_engine.core.core.orchestration.llm_orchestrator import (
        LLMOrchestrator,
        LLMConfig,
        ModelProvider,
    )


def _parse_tags(pairs: List[str]) -> Dict[str, str]:
    tags: Dict[str, str] = {}
    for p in pairs or []:
        if "=" in p:
            k, v = p.split("=", 1)
            if k:
                tags[k.strip()] = v.strip()
    return tags


def _read_inputs(prompt: Optional[str], input_path: Optional[str]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if input_path:
        p = Path(input_path)
        if not p.exists():
            raise FileNotFoundError(f"input not found: {p}")
        if p.suffix.lower() in {".json", ".jsonl", ".ndjson"}:
            # Read JSON array or NDJSON
            text = p.read_text(encoding="utf-8")
            try:
                data = json.loads(text)
                if isinstance(data, list):
                    for x in data:
                        items.append({"prompt": str(x.get("prompt") if isinstance(x, dict) else x)})
                else:
                    items.append({"prompt": str(data)})
            except Exception:
                # NDJSON fallback
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        obj = {"prompt": line}
                    if isinstance(obj, dict):
                        items.append({"prompt": str(obj.get("prompt") or line)})
        else:
            # Plain text: one prompt per line
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    items.append({"prompt": line})
    if prompt:
        items.append({"prompt": prompt})
    return items


async def _check_lmstudio_endpoint(url: str) -> Tuple[bool, str]:
    try:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(url.rstrip("/") + "/v1/models", timeout=5) as r:
                return (r.status == 200, f"status={r.status}")
    except Exception as e:
        return (False, str(e))


async def _generate_one(
    orch: LLMOrchestrator,
    provider_name: str,
    item: Dict[str, Any],
    model: Optional[str],
    system_prompt: Optional[str],
) -> Dict[str, Any]:
    prompt = str(item.get("prompt") or "").strip()
    resp = await orch.generate(
        prompt,
        system=system_prompt,
        provider_name=provider_name,
    )
    text = (resp.text or "").strip()
    return {
        "ok": bool(text),
        "prompt": prompt,
        "response": {"text": text},
        "metadata": {
            "provider_name": provider_name,
            "model": model,
        },
    }


async def cmd_run(args: argparse.Namespace) -> int:
    init_logging_from_env()
    log = get_logger(__name__, component="cli", workflow="run")

    # Inputs
    items = _read_inputs(args.prompt, args.input)
    if not items and args.runs > 0:
        # Default single generic prompt if nothing provided
        items = [{"prompt": "Write one vivid sentence about sunrise in Jerusalem."}]

    # Repeat for --runs if explicit
    if args.runs > 1:
        base = items.copy()
        items = []
        for _ in range(args.runs):
            items.extend(base)

    # Provider setup (LM Studio default)
    provider = (args.provider or "lmstudio").lower()
    endpoint = args.endpoint or os.getenv("LM_ENDPOINT") or os.getenv("LMSTUDIO_URL") or "http://localhost:1234"
    model = args.model or os.getenv("LM_MODEL") or "auto"

    orch = LLMOrchestrator(fail_on_all_providers=True)
    if provider == "lmstudio":
        orch.register_provider(
            "lmstudio",
            LLMConfig(
                provider=ModelProvider.LMSTUDIO,
                endpoint=endpoint,
                model=model,
                temperature=float(args.temperature),
                max_tokens=int(args.max_tokens),
                require_explicit_success=True,
            ),
        )
        # Quick health check
        ok, info = await _check_lmstudio_endpoint(endpoint)
        if not ok:
            log.error("LM Studio not reachable", extra={"endpoint": endpoint, "info": info})
            if args.require_healthy:
                print(f"LM Studio not reachable at {endpoint}: {info}")
                return 2
    else:
        print(f"Unsupported provider: {provider}")
        return 2

    # DB connection (health-aware settings)
    s = get_db_settings()
    db_type = s.get("db_type", "sqlite")
    if db_type == "oracle":
        db = get_database_connection(
            db_type="oracle",
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
            retry_attempts=int(s.get("retry_attempts", 2)),
            retry_backoff_seconds=float(s.get("retry_backoff_seconds", 0.5)),
            ping_on_connect=bool(s.get("ping_on_connect", True)),
        )
    elif db_type == "postgresql":
        db = get_database_connection(
            db_type="postgresql",
            db_name=s.get("db_name", "story_db"),
            user=s.get("user", "story"),
            password=s.get("password"),
            host=s.get("host", "localhost"),
            port=int(s.get("port", 5432)),
            sslmode=s.get("sslmode"),
            sslrootcert=s.get("sslrootcert"),
            sslcert=s.get("sslcert"),
            sslkey=s.get("sslkey"),
        )
    else:
        db_name = s.get("db_name", os.getenv("SQLITE_DB", "workflow_outputs.db"))
        db = get_database_connection("sqlite", db_name=db_name)

    # Open DB unless dry-run
    if not args.dry_run:
        try:
            db.connect()
        except Exception as e:
            log.error("db.connect failed", extra={"error": str(e)})
            if args.require_healthy:
                print(f"DB connect failed: {e}")
                return 2

    # Run tasks with bounded concurrency
    sem = asyncio.Semaphore(max(1, int(args.parallel)))
    results: List[Dict[str, Any]] = []

    async def worker(item: Dict[str, Any]) -> None:
        async with sem:
            try:
                r = await _generate_one(orch, provider, item, model=model, system_prompt=args.system)
                results.append(r)
                if not args.dry_run:
                    payload = {
                        "ok": r.get("ok", False),
                        "input": {"prompt": r.get("prompt")},
                        "output": r.get("response"),
                        "provider": provider,
                        "model": model,
                        "tags": _parse_tags(args.tag or []),
                    }
                    db.store_output(args.workflow, payload)
            except Exception as e:
                log.error("generation failed", extra={"error": str(e)})
                if args.fail_fast:
                    raise

    await asyncio.gather(*(worker(it) for it in items))

    # Close DB
    try:
        if not args.dry_run:
            db.disconnect()
    except Exception:
        pass

    # Summary
    ok_count = sum(1 for r in results if r.get("ok"))
    print(json.dumps({
        "workflow": args.workflow,
        "requested": len(items),
        "ok": ok_count,
        "failed": len(items) - ok_count,
        "db_type": db_type,
    }, indent=2))
    return 0 if ok_count > 0 else 1


def cmd_db_health(_: argparse.Namespace) -> int:
    init_logging_from_env()
    s = get_db_settings()
    db_type = s.get("db_type", "sqlite")
    try:
        if db_type == "oracle":
            db = get_database_connection(
                db_type="oracle",
                user=s.get("user"),
                password=s.get("password"),
                dsn=s.get("dsn"),
                wallet_location=s.get("wallet_location"),
                wallet_password=s.get("wallet_password"),
                use_pool=True,
                pool_min=1,
                pool_max=2,
                retry_attempts=1,
            )
        elif db_type == "postgresql":
            db = get_database_connection(
                db_type="postgresql",
                db_name=s.get("db_name", "story_db"),
                user=s.get("user", "story"),
                password=s.get("password"),
                host=s.get("host", "localhost"),
                port=int(s.get("port", 5432)),
            )
        else:
            db = get_database_connection("sqlite", db_name=s.get("db_name", "workflow_outputs.db"))
        db.connect()
        db.disconnect()
        print(json.dumps({"db_type": db_type, "healthy": True}))
        return 0
    except Exception as e:
        print(json.dumps({"db_type": db_type, "healthy": False, "error": str(e)}))
        return 1


def cmd_db_export(args: argparse.Namespace) -> int:
    init_logging_from_env()
    s = get_db_settings()
    db_type = s.get("db_type", "sqlite")
    if db_type == "oracle":
        db = get_database_connection(
            db_type="oracle",
            user=s.get("user"),
            password=s.get("password"),
            dsn=s.get("dsn"),
            wallet_location=s.get("wallet_location"),
        )
    elif db_type == "postgresql":
        db = get_database_connection(
            db_type="postgresql",
            db_name=s.get("db_name", "story_db"),
            user=s.get("user", "story"),
            password=s.get("password"),
            host=s.get("host", "localhost"),
            port=int(s.get("port", 5432)),
        )
    else:
        db = get_database_connection("sqlite", db_name=s.get("db_name", "workflow_outputs.db"))
    db.connect()
    try:
        rows = db.get_outputs(args.workflow)
        out = "\n".join(json.dumps(r) for r in rows[: (args.limit or len(rows))])
        if args.output:
            Path(args.output).write_text(out)
        else:
            print(out)
    finally:
        db.disconnect()
    return 0


def cmd_config_show(_: argparse.Namespace) -> int:
    s = get_db_settings()
    eff = {
        "db_type": s.get("db_type"),
        "db_user": s.get("user"),
        "db_host": s.get("host"),
        "db_name": s.get("db_name"),
        "lm_endpoint": os.getenv("LM_ENDPOINT") or os.getenv("LMSTUDIO_URL"),
        "lm_model": os.getenv("LM_MODEL"),
    }
    print(json.dumps(eff, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="story", description="Story Engine CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # run
    pr = sub.add_parser("run", help="Run prompts through the engine and persist outputs")
    pr.add_argument("--workflow", default="cli_run", help="Workflow name for persistence")
    pr.add_argument("--provider", default="lmstudio", help="Model provider (default: lmstudio)")
    pr.add_argument("--endpoint", default=None, help="Provider endpoint (defaults to LM_ENDPOINT)")
    pr.add_argument("--model", default=None, help="Model id/name")
    pr.add_argument("--prompt", default=None, help="Inline prompt text")
    pr.add_argument("--input", default=None, help="File with prompts (txt,json,jsonl)")
    pr.add_argument("--runs", type=int, default=1, help="Repeat runs per prompt")
    pr.add_argument("--parallel", type=int, default=1, help="Parallelism level")
    pr.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    pr.add_argument("--max-tokens", type=int, default=128, help="Max tokens per generation")
    pr.add_argument("--system", default="You are a helpful, concise assistant.")
    pr.add_argument("--tag", action="append", help="Attach tag key=value (repeat)")
    pr.add_argument("--dry-run", action="store_true", help="Skip DB persistence")
    pr.add_argument("--fail-fast", action="store_true", help="Abort on first generation failure")
    pr.add_argument("--require-healthy", action="store_true", help="Fail if LM or DB unhealthy")

    # db health
    phealth = sub.add_parser("db-health", help="Check DB connectivity")

    # db export
    pexp = sub.add_parser("db-export", help="Export workflow outputs as NDJSON")
    pexp.add_argument("--workflow", required=True)
    pexp.add_argument("--output", default=None, help="Path to write (stdout if omitted)")
    pexp.add_argument("--limit", type=int, default=None)

    # config show
    pshow = sub.add_parser("config-show", help="Show effective runtime config")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "run":
        return asyncio.run(cmd_run(args))
    if args.cmd == "db-health":
        return cmd_db_health(args)
    if args.cmd == "db-export":
        return cmd_db_export(args)
    if args.cmd == "config-show":
        return cmd_config_show(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

