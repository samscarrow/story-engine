from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from llm_observability import get_logger, observe_metric, log_exception, ErrorCodes
from .interfaces import EngineContext, Plan, Step, StepKind, EngineResult
from .results import StepResult, StepStatus


@dataclass
class OrchestratorOptions:
    max_concurrency: int = 4
    default_timeout_sec: float = 60.0


class EngineOrchestrator:
    """Executes Plan DAGs with retries, timeouts, and idempotency.

    This orchestrator is intentionally minimal: it resolves dependencies and
    runs steps whose prerequisites are complete, respecting a concurrency cap.
    """

    def __init__(self, options: Optional[OrchestratorOptions] = None, logger: Optional[logging.Logger] = None):
        self.options = options or OrchestratorOptions()
        self._log = logger or get_logger("engine.orchestrator")

    async def run(self, plan: Plan, ctx: EngineContext) -> EngineResult:
        self._log.info(
            "engine.plan.start",
            extra={"roots": plan.roots, "steps": list(plan.steps.keys()), **(plan.metadata or {})},
        )

        completed: Dict[str, Any] = {}
        errors: Dict[str, str] = {}

        # Simple readiness: a step is ready when all dependencies are in completed
        pending = set(plan.steps.keys())
        running: set[str] = set()
        sem = asyncio.Semaphore(self.options.max_concurrency)

        async def _run_step(step: Step):
            nonlocal completed, errors
            async with sem:
                # Idempotency short-circuit
                try:
                    if step.idempotency_key and ctx.jobs and ctx.jobs.was_processed(step.idempotency_key):
                        self._log.info("engine.step.skip_idempotent", extra={"step": step.key})
                        completed[step.key] = {"skipped": True}
                        try:
                            ctx.results[step.key] = completed[step.key]
                            ctx.meta[step.key] = StepResult(
                                status=StepStatus.SKIPPED,
                                elapsed_ms=0.0,
                                summary={"idempotent": True},
                            )
                        except Exception:
                            pass
                        return
                except Exception:
                    pass

                attempts = max(1, step.retry.max_attempts)
                base_delay = max(0.0, step.retry.base_delay_sec)

                last_err: Optional[Exception] = None
                for attempt in range(1, attempts + 1):
                    try:
                        timeout = step.timeout_sec or self.options.default_timeout_sec
                        # best-effort timing
                        try:
                            from time import perf_counter as _pc
                            _t0 = _pc()
                        except Exception:
                            _t0 = None  # type: ignore[assignment]
                        with_obs = {"step": step.key, "kind": step.kind.value, **(step.metadata or {})}
                        self._log.info("engine.step.start", extra={**with_obs, "attempt": attempt})

                        async def _do() -> Any:
                            if step.kind == StepKind.AI_REQUEST:
                                # Allow dynamic AI steps to prepare params or call AI directly
                                if step.func is not None:
                                    return await _maybe_await(step.func(ctx, step.params))
                                if not ctx.ai:
                                    raise RuntimeError("AI client not configured in EngineContext")
                                resp = await ctx.ai.generate(**step.params)
                                text = getattr(resp, "text", None)
                                return resp if text is None else {"text": text, "raw": resp}
                            elif step.kind in {StepKind.TRANSFORM, StepKind.PERSIST, StepKind.FETCH}:
                                if not step.func:
                                    raise RuntimeError(f"Step {step.key} missing callable func")
                                return await _maybe_await(step.func(ctx, step.params))
                            elif step.kind == StepKind.BRANCH:
                                # Branch step can compute and attach decision metadata
                                if step.func:
                                    return await _maybe_await(step.func(ctx, step.params))
                                return {"branch": True}
                            else:
                                raise RuntimeError(f"Unsupported step kind: {step.kind}")

                        result = await asyncio.wait_for(_do(), timeout=timeout)
                        completed[step.key] = result
                        # Expose result to subsequent steps via context
                        try:
                            ctx.results[step.key] = result
                        except Exception:
                            pass
                        # duration metric
                        elapsed_ms = 0.0
                        try:
                            if _t0 is not None:
                                from time import perf_counter as _pc
                                elapsed_ms = (_pc() - _t0) * 1000.0
                        except Exception:
                            pass
                        observe_metric("engine.step.ms", elapsed_ms, step=step.key, kind=step.kind.value)
                        # Mark idempotent steps as processed
                        try:
                            if step.idempotency_key and ctx.jobs:
                                ctx.jobs.mark_processed(step.idempotency_key)
                        except Exception:
                            pass
                        self._log.info("engine.step.ok", extra={**with_obs, "elapsed_ms": int(elapsed_ms)})
                        try:
                            ctx.meta[step.key] = StepResult(status=StepStatus.SUCCESS, elapsed_ms=elapsed_ms)
                        except Exception:
                            pass
                        return
                    except Exception as e:  # noqa: PERF203 - different exception paths
                        last_err = e
                        log_exception(
                            self._log,
                            code=(
                                ErrorCodes.GEN_TIMEOUT
                                if "timeout" in str(e).lower()
                                else ErrorCodes.CONFIG_INVALID
                            ),
                            component="engine.step",
                            exc=e,
                            step=step.key,
                        )
                        if attempt < attempts:
                            # compute backoff with jitter
                            delay = base_delay * (2 ** (attempt - 1))
                            try:
                                import random
                                delay *= (1.0 - step.retry.jitter) + (2 * step.retry.jitter * random.random())
                            except Exception:
                                pass
                            await asyncio.sleep(delay)
                            continue
                        break
                # Exhausted attempts
                err = str(last_err or RuntimeError("step_failed"))
                errors[step.key] = err
                try:
                    ctx.meta[step.key] = StepResult(status=StepStatus.FAILED, elapsed_ms=0.0, error=err)
                except Exception:
                    pass

        # Helper: maybe await sync callables
        async def _maybe_await(value: Any) -> Any:
            if asyncio.iscoroutine(value):
                return await value
            return value

        # Execution loop: run-ready until no progress or all done
        tasks: Dict[str, asyncio.Task] = {}
        while pending:
            made_progress = False
            ready = [k for k in list(pending) if all(d in completed for d in plan.steps[k].depends_on)]

            # If nothing ready but tasks are running, wait for one to finish
            if not ready and tasks:
                await asyncio.wait(tasks.values(), return_when=asyncio.FIRST_COMPLETED)
                # collect finished
                for k, t in list(tasks.items()):
                    if t.done():
                        t.result()  # propagate exceptions
                        pending.discard(k)
                        running.discard(k)
                        tasks.pop(k, None)
                        made_progress = True
                continue

            for key in ready:
                if key in running:
                    continue
                running.add(key)
                tasks[key] = asyncio.create_task(_run_step(plan.steps[key]))
                made_progress = True

            if not made_progress:
                # deadlock or all running; wait for completion
                if tasks:
                    await asyncio.wait(tasks.values(), return_when=asyncio.FIRST_COMPLETED)
                    for k, t in list(tasks.items()):
                        if t.done():
                            t.result()
                            pending.discard(k)
                            running.discard(k)
                            tasks.pop(k, None)
                else:
                    # No tasks to wait on; break to avoid infinite loop
                    break

        # Drain any remaining tasks
        if tasks:
            await asyncio.gather(*tasks.values())
            pending -= set(tasks.keys())

        ok = len(errors) == 0
        result = EngineResult(success=ok, step_results=completed, errors=errors)
        self._log.info("engine.plan.done", extra={"ok": ok, "errors": list(errors.keys())})
        return result
