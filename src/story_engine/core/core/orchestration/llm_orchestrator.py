"""
LLM Orchestration Layer - Strict Version
No silent failures, no mock fallbacks, explicit error handling
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import aiohttp
import json
import os
from datetime import datetime
import logging
import traceback
from time import monotonic

from .model_filters import filter_models
from .response_normalizer import normalize_openai_chat, extract_text_and_reasoning, _coerce_text
from .kobold_normalizer import normalize_kobold
from llm_observability import (
    get_logger,
    GenerationDBLogger,
    log_exception,
    observe_metric,
    metric_event,
    inc_metric,
    ErrorCodes,
)

# Logger (configuration is handled by entrypoints)
logger = logging.getLogger(__name__)
_obs = get_logger("orchestrator")


class ModelProvider(Enum):
    """Supported LLM providers"""

    LMSTUDIO = "lmstudio"
    KOBOLDCPP = "koboldcpp"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LLAMACPP = "llamacpp"
    OLLAMA = "ollama"
    CUSTOM = "custom"


class GenerationError(Exception):
    """Specific exception for generation failures"""

    def __init__(self, provider: str, original_error: Exception, details: Dict = None):
        self.provider = provider
        self.original_error = original_error
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()
        super().__init__(f"Generation failed on {provider}: {original_error}")


@dataclass
class ProviderFailure:
    """Record of a provider failure"""

    provider_name: str
    error_type: str
    error_message: str
    timestamp: str
    request_prompt: str
    traceback: str = ""

    def to_dict(self) -> Dict:
        return {
            "provider": self.provider_name,
            "error_type": self.error_type,
            "error": self.error_message,
            "timestamp": self.timestamp,
            "prompt_preview": (
                self.request_prompt[:200] + "..."
                if len(self.request_prompt) > 200
                else self.request_prompt
            ),
            "traceback": self.traceback,
        }


@dataclass
class LLMConfig:
    """Universal configuration for LLM providers"""

    provider: ModelProvider
    endpoint: str
    model: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.1
    stop_sequences: List[str] = None
    timeout: int = 60
    extra_params: Dict[str, Any] = None
    require_explicit_success: bool = True  # Must get valid response

    def __post_init__(self):
        if self.stop_sequences is None:
            self.stop_sequences = []
        if self.extra_params is None:
            self.extra_params = {}


@dataclass
class LLMResponse:
    """Standardized response format"""

    text: str
    provider: ModelProvider
    provider_name: str
    model: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    raw_response: Optional[Dict] = None
    timestamp: str = None
    generation_time_ms: Optional[float] = None
    failures_before_success: List[ProviderFailure] = field(default_factory=list)

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

        # Validate response is not empty or mock
        if not self.text or self.text.strip() == "":
            raise ValueError("Empty response text")

        # Check for mock/placeholder patterns
        mock_patterns = [
            "[mock response]",
            "mock llm",
            "placeholder text",
            "lorem ipsum",
            "test response",
        ]

        text_lower = self.text.lower()
        for pattern in mock_patterns:
            if pattern in text_lower:
                raise ValueError(f"Mock/placeholder response detected: {pattern}")


@dataclass
class LMStudioCapabilities:
    """Describes feature support advertised (or inferred) from an LM Studio endpoint."""

    supports_reasoning: bool
    supports_response_format: bool
    supports_tools: bool
    requires_model: bool
    models: List[Dict[str, Any]] = field(default_factory=list)
    obtained_at: float = 0.0


@dataclass
class LMStudioRequestPlan:
    """Request parameters chosen after capability probing."""

    payload: Dict[str, Any]
    headers: Dict[str, str]
    attempts: int
    base_delay: float
    expect_reasoning: bool
    stream: bool


def _model_supports_reasoning(model: Dict[str, Any]) -> bool:
    """Heuristically determine if the model advertises reasoning/thinking output."""

    caps = model.get("capabilities") or {}
    if isinstance(caps, dict):
        if caps.get("reasoning") is True:
            return True
        features = caps.get("features")
        if isinstance(features, (list, tuple)) and any(
            isinstance(item, str) and "reason" in item.lower() for item in features
        ):
            return True
    tags = model.get("tags")
    if isinstance(tags, (list, tuple)) and any(
        isinstance(tag, str) and "reason" in tag.lower() for tag in tags
    ):
        return True
    model_id = str(model.get("id", "")).lower()
    return any(keyword in model_id for keyword in ("reason", "think", "r1", "deepseek"))


def _model_supports_tools(model: Dict[str, Any]) -> bool:
    caps = model.get("capabilities") or {}
    if isinstance(caps, dict) and caps.get("tools"):
        return True
    tags = model.get("tags")
    if isinstance(tags, (list, tuple)) and any(
        isinstance(tag, str) and "tool" in tag.lower() for tag in tags
    ):
        return True
    return False


def _model_id_prefers_reasoning(model_id: Optional[str]) -> bool:
    if not model_id:
        return False
    name = str(model_id).lower()
    return any(token in name for token in ("reason", "think", "r1", "deepseek"))


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.failure_count = 0
        self.last_failure: Optional[ProviderFailure] = None

    @abstractmethod
    async def generate(
        self, prompt: str, system: Optional[str] = None, **kwargs
    ) -> LLMResponse:
        """Generate text from prompt - must return real response or raise exception"""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check provider health - return detailed status"""
        pass

    @abstractmethod
    def format_prompt(self, prompt: str, system: str = None) -> str:
        """Format prompt for specific provider"""
        pass

    def record_failure(self, error: Exception, prompt: str) -> ProviderFailure:
        """Record a failure for debugging"""
        self.failure_count += 1
        failure = ProviderFailure(
            provider_name=self.__class__.__name__,
            error_type=type(error).__name__,
            error_message=str(error),
            timestamp=datetime.now().isoformat(),
            request_prompt=prompt,
            traceback=traceback.format_exc(),
        )
        self.last_failure = failure
        return failure


class LMStudioProvider(LLMProvider):
    """LMStudio API provider"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._cb_failures: int = 0
        self._cb_open_until: float = 0.0
        self._capabilities: Optional[LMStudioCapabilities] = None
        try:
            self._capabilities_ttl = float(os.getenv("LM_CAPABILITIES_TTL", "300") or 300.0)
        except Exception:
            self._capabilities_ttl = 300.0

    async def _consume_stream(
        self,
        response: aiohttp.ClientResponse,
        expect_reasoning: bool,
        *,
        model: str,
    ) -> tuple[str, str, Dict[str, Any], list[Dict[str, Any]]]:
        """Consume an OpenAI-style streaming response and return text, reasoning, usage, and raw events."""

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage: Dict[str, Any] = {}
        events: list[Dict[str, Any]] = []

        buffer = ""
        done = False

        def _process_event(block: str) -> bool:
            nonlocal usage
            data_lines: list[str] = []
            for line in block.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith(":"):
                    continue
                if stripped.startswith("data:"):
                    data_lines.append(stripped[5:].lstrip())
            if not data_lines:
                return False

            payload_text = "\n".join(data_lines).strip()
            if not payload_text:
                return False
            if payload_text == "[DONE]":
                return True

            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError:
                return False

            events.append(payload)
            for choice in payload.get("choices", []):
                delta = choice.get("delta") or {}
                content = _coerce_text(delta.get("content"))
                if content:
                    text_parts.append(content)
                reason_delta = delta.get("reasoning")
                reason_text = _coerce_text(reason_delta)
                if reason_text:
                    reasoning_parts.append(reason_text)
                finish_reason = choice.get("finish_reason")
                if finish_reason and finish_reason != "stop" and expect_reasoning:
                    reasoning_parts.append(f"[finish_reason:{finish_reason}]")

            payload_usage = payload.get("usage")
            if isinstance(payload_usage, dict):
                usage = payload_usage or usage

            return False

        async for raw in response.content:
            try:
                chunk = raw.decode("utf-8")
            except Exception:
                continue
            buffer += chunk

            while True:
                idx_rn = buffer.find("\r\n\r\n")
                idx_n = buffer.find("\n\n")
                if idx_rn == -1 and idx_n == -1:
                    break
                if idx_rn != -1 and (idx_n == -1 or idx_rn < idx_n):
                    event_block = buffer[:idx_rn]
                    buffer = buffer[idx_rn + 4 :]
                else:
                    event_block = buffer[:idx_n]
                    buffer = buffer[idx_n + 2 :]

                if _process_event(event_block):
                    done = True
                    break
            if done:
                break

        if not done and buffer.strip():
            _process_event(buffer)

        text = "".join(text_parts).strip()
        reasoning = " ".join(reasoning_parts).strip()
        return text, reasoning, usage, events

    def _capabilities_fresh(self) -> bool:
        return (
            self._capabilities is not None
            and self._capabilities.obtained_at > 0
            and (monotonic() - self._capabilities.obtained_at) < self._capabilities_ttl
        )

    def _use_ssl(self) -> bool:
        try:
            return str(self.config.endpoint).lower().startswith("https")
        except Exception:
            return False

    async def _get_capabilities(self, force_refresh: bool = False) -> LMStudioCapabilities:
        if not force_refresh and self._capabilities_fresh():
            return self._capabilities  # type: ignore[return-value]

        prefer_reasoning = _env_truthy(os.getenv("LM_PREFER_REASONING"))
        default_caps = LMStudioCapabilities(
            supports_reasoning=prefer_reasoning,
            supports_response_format=prefer_reasoning,
            supports_tools=False,
            requires_model=False,
            models=[],
            obtained_at=monotonic(),
        )

        if not _env_truthy(os.getenv("LM_SKIP_CAPABILITIES_PROBE")):
            url = f"{self.config.endpoint}/v1/models"
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=5, ssl=self._use_ssl()) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            models = []
                            if isinstance(data, dict):
                                models = data.get("data", []) or []
                            supports_reasoning = prefer_reasoning or any(
                                isinstance(item, dict) and _model_supports_reasoning(item)
                                for item in models
                            )
                            supports_tools = any(
                                isinstance(item, dict) and _model_supports_tools(item)
                                for item in models
                            )
                            requires_model = bool(models)
                            default_caps = LMStudioCapabilities(
                                supports_reasoning=supports_reasoning,
                                supports_response_format=supports_reasoning
                                or prefer_reasoning
                                or _env_truthy(os.getenv("LM_ASSUME_RESPONSE_FORMAT")),
                                supports_tools=supports_tools,
                                requires_model=requires_model,
                                models=models,
                                obtained_at=monotonic(),
                            )
                        else:
                            # Keep defaults but refresh timestamp so we don't hammer endpoint
                            default_caps.obtained_at = monotonic()
            except Exception:
                default_caps.obtained_at = monotonic()

        # Allow manual overrides even when fetch succeeds
        if _env_truthy(os.getenv("LM_FORCE_REASONING")):
            default_caps.supports_reasoning = True
            default_caps.supports_response_format = True
        if _env_truthy(os.getenv("LM_DISABLE_REASONING")):
            default_caps.supports_reasoning = False

        self._capabilities = default_caps
        return default_caps

    def _select_model(self, caps: LMStudioCapabilities, requested: Optional[str]) -> str:
        if requested:
            return requested
        if self.config.model:
            return self.config.model
        env_model = os.getenv("LM_MODEL")
        if env_model:
            return env_model
        for item in caps.models:
            if isinstance(item, dict):
                model_id = item.get("id")
                if model_id:
                    return str(model_id)
        # fall back to auto routing
        return "auto"

    async def _build_request_plan(
        self,
        prompt: str,
        system: Optional[str],
        caps: LMStudioCapabilities,
        kwargs: Dict[str, Any],
    ) -> LMStudioRequestPlan:
        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Determine model
        model_override = kwargs.get("model")
        selected_model = self._select_model(caps, model_override)

        stream_requested = bool(kwargs.get("stream") or kwargs.get("streaming"))
        payload: Dict[str, Any] = {
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "stream": stream_requested,
        }
        if selected_model:
            payload["model"] = selected_model

        # Carry through explicit response_format if caller requested it
        if "response_format" in kwargs and kwargs.get("response_format"):
            payload["response_format"] = kwargs["response_format"]

        headers = {"Content-Type": "application/json"}

        try:
            env_attempts = int(os.getenv("LM_RETRY_ATTEMPTS", "1") or 1)
        except Exception:
            env_attempts = 1
        try:
            base_delay = float(os.getenv("LM_RETRY_BASE_DELAY", "0.2") or 0.2)
        except Exception:
            base_delay = 0.2

        attempts = max(env_attempts, 2 if "response_format" in payload else 1)

        expect_reasoning = caps.supports_reasoning or _model_id_prefers_reasoning(selected_model)
        if _env_truthy(os.getenv("LM_DISABLE_REASONING")):
            expect_reasoning = False

        if not stream_requested:
            auto_stream = expect_reasoning and caps.supports_reasoning and _env_truthy(os.getenv("LM_STREAM_REASONING"))
            if auto_stream:
                stream_requested = True
        payload["stream"] = stream_requested

        return LMStudioRequestPlan(
            payload=payload,
            headers=headers,
            attempts=attempts,
            base_delay=base_delay,
            expect_reasoning=expect_reasoning,
            stream=stream_requested,
        )

    async def generate(
        self, prompt: str, system: Optional[str] = None, **kwargs
    ) -> LLMResponse:
        """Generate using LMStudio chat completions API"""
        # Simple circuit breaker
        now = monotonic()
        if self._cb_open_until and now < self._cb_open_until:
            # Allow a half-open probe if scheduled
            if getattr(self, "_cb_half_open_at", 0.0) and now >= self._cb_half_open_at:
                pass  # allow single probe
            else:
                msg = f"circuit open until {self._cb_open_until:.2f}"
                log_exception(_obs, code=ErrorCodes.AI_LB_UNAVAILABLE, component="lmstudio", exc=RuntimeError(msg))
                raise GenerationError("lmstudio", RuntimeError("circuit_open"), {"until": self._cb_open_until})

        start_time = asyncio.get_event_loop().time()
        caps = await self._get_capabilities()
        plan = await self._build_request_plan(prompt, system, caps, kwargs)

        model_for_log = plan.payload.get("model") or kwargs.get("model") or self.config.model or os.getenv("LM_MODEL", "auto")
        _obs.info(
            "llm.request",
            extra={
                "provider": "lmstudio",
                "endpoint": self.config.endpoint,
                "model": model_for_log,
                "expect_reasoning": plan.expect_reasoning,
            },
        )
        url = f"{self.config.endpoint}/v1/chat/completions"
        # Optional sticky session for ai-lb routing
        session_id = kwargs.get("session_id")

        payload_base = dict(plan.payload)
        payload_without_response_format = (
            {k: v for k, v in payload_base.items() if k != "response_format"}
            if "response_format" in payload_base
            else payload_base
        )
        payload_has_response_format = payload_without_response_format is not payload_base
        expect_reasoning = plan.expect_reasoning

        try:
            async with aiohttp.ClientSession() as session:
                # attempts: env override or response_format fallback
                attempts = plan.attempts
                base_delay = plan.base_delay


                _evt_logger = GenerationDBLogger()
                # optional time budget in ms
                budget_ms = kwargs.get("budget_ms") or int(os.getenv("LM_REQUEST_BUDGET_MS", "0") or 0)
                tb_start = asyncio.get_event_loop().time()
                for i in range(attempts):
                    drop_response_format = i > 0 and payload_has_response_format
                    current_payload = (
                        payload_without_response_format if drop_response_format else payload_base
                    )
                    if drop_response_format:
                        logger.warning(
                            "Retrying LMStudio generation without response_format (text mode)"
                        )
                        expect_reasoning = False

                    headers = dict(plan.headers)
                    # LB hints (best-effort, optional)
                    try:
                        import os as _os
                        if _os.environ.get("LM_PREFER_SMALL", "").strip().lower() in {"1", "true", "yes", "on"}:
                            headers["x-prefer-small"] = "1"
                    except Exception:
                        pass
                    if isinstance(session_id, str) and session_id.strip():
                        headers["x-session-id"] = session_id.strip()

                    # compute remaining timeout if budget present
                    total_timeout = self.config.timeout
                    if budget_ms and budget_ms > 0:
                        elapsed_ms = int((asyncio.get_event_loop().time() - tb_start) * 1000)
                        remaining_ms = max(0, budget_ms - elapsed_ms)
                        if remaining_ms <= 0:
                            raise asyncio.TimeoutError("time_budget_exhausted")
                        total_timeout = min(total_timeout, remaining_ms / 1000.0)

                    try:
                        async with session.post(
                            url,
                            headers=headers,
                            json=current_payload,
                            timeout=aiohttp.ClientTimeout(total=total_timeout),
                            ssl=self._use_ssl(),
                        ) as response:
                            if current_payload.get("stream"):
                                if response.status != 200:
                                    error_text = await response.text()
                                    raise GenerationError(
                                        "lmstudio",
                                        Exception(f"HTTP {response.status}: {error_text}"),
                                        {"status": response.status, "response": error_text},
                                    )
                                stream_text, stream_reasoning, usage, events = await self._consume_stream(
                                    response, expect_reasoning, model=model_for_log
                                )
                                if not stream_text and stream_reasoning and not expect_reasoning:
                                    stream_text = stream_reasoning
                                result = LLMResponse(
                                    text=stream_text,
                                    provider=ModelProvider.LMSTUDIO,
                                    provider_name="lmstudio",
                                    model=model_for_log,
                                    usage=usage or {},
                                    raw_response={
                                        "stream": events,
                                        "headers": {k.lower(): v for k, v in response.headers.items()},
                                        "normalized": {"reasoning": stream_reasoning},
                                    },
                                    generation_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000,
                                )
                                # Metrics code removed as it is no longer needed.
                                try:
                                    _obs.info(
                                        "llm.response",
                                        extra={
                                            "provider": "lmstudio",
                                            "endpoint": self.config.endpoint,
                                            "model": result.model,
                                            "elapsed_ms": int(result.generation_time_ms or 0),
                                            "len": len(result.text or ""),
                                            "ok": True,
                                            "attempt": int(i + 1),
                                            "reasoning_expected": expect_reasoning,
                                            "reasoning_present": bool(stream_reasoning),
                                        },
                                    )
                                except Exception:
                                    pass
                                if self._cb_open_until:
                                    _obs.info("circuit.reset", extra={"provider": "lmstudio"})
                                self._cb_open_until = 0.0
                                self._cb_failures = 0
                                return result

                            if response.status != 200:
                                error_text = await response.text()
                                if response.status in {400, 404} and ("model_not_found" in error_text or "No models loaded" in error_text):
                                    try:
                                        _obs.warning(
                                            "lmstudio.model_not_loaded",
                                            extra={
                                                "endpoint": self.config.endpoint,
                                                "hint": "Load a model in LM Studio or set LM_MODEL to a valid model id (e.g., one from /v1/models).",
                                            },
                                        )
                                    except Exception:
                                        pass
                                if (
                                    i == 0
                                    and response.status == 400
                                    and "response_format.type" in error_text
                                    and "response_format" in payload_base
                                ):
                                    try:
                                        _evt_logger.log_event(
                                            kind="retry",
                                            provider_name="lmstudio",
                                            provider_type="lmstudio",
                                            provider_endpoint=self.config.endpoint,
                                            data={
                                                "reason": "response_format_rejected",
                                                "status": response.status,
                                                "error": error_text,
                                            },
                                            model_key=self.config.model,
                                        )
                                    except Exception:
                                        pass
                                if i < attempts - 1:
                                    try:
                                        import random as _rnd
                                        jitter = 0.8 + 0.4 * _rnd.random()
                                        await asyncio.sleep(min(base_delay * (2 ** i) * jitter, 2.0))
                                    except Exception:
                                        pass
                                continue
                                if 500 <= response.status < 600 and i < attempts - 1:
                                    try:
                                        import random as _rnd
                                        jitter = 0.8 + 0.4 * _rnd.random()
                                        await asyncio.sleep(min(base_delay * (2 ** i) * jitter, 2.5))
                                    except Exception:
                                        pass
                                    continue
                                raise GenerationError(
                                    "lmstudio",
                                    Exception(f"HTTP {response.status}: {error_text}"),
                                    {"status": response.status, "response": error_text},
                                )

                            data = await response.json()
                            headers_map = {
                                k.lower(): v for k, v in response.headers.items()
                            }
                            norm = normalize_openai_chat(data, headers=headers_map)

                            result = LLMResponse(
                                text=norm.get("text", ""),
                                provider=ModelProvider.LMSTUDIO,
                                provider_name="lmstudio",
                                model=norm.get("meta", {}).get("effective_model") or data.get("model") or current_payload.get("model") or self.config.model,
                                usage=norm.get("meta", {}).get("usage") or data.get("usage"),
                                raw_response={
                                    "data": data,
                                    "headers": headers_map,
                                    "normalized": {"reasoning": norm.get("reasoning", "")},
                                },
                                generation_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000,
                            )
                            try:
                                # from ..common.observability import observe_metric, metric_event, inc_metric
                                # observe_metric("llm.lmstudio.gen_ms", result.generation_time_ms or 0.0, endpoint=self.config.endpoint, model=result.model)
                                # metric_event("llm.lmstudio.attempts", value=float(i + 1), endpoint=self.config.endpoint, model=result.model)
                                # if i > 0:
                                #     inc_metric("llm.lmstudio.retries", n=i, endpoint=self.config.endpoint, model=result.model)
                                pass
                            except Exception:
                                pass
                            try:
                                usage = result.usage or {}
                                _obs.info(
                                    "llm.response",
                                    extra={
                                        "provider": "lmstudio",
                                        "endpoint": self.config.endpoint,
                                        "model": result.model,
                                        "elapsed_ms": int(result.generation_time_ms or 0),
                                        "len": len(result.text or ""),
                                        "ok": True,
                                        "attempt": int(i + 1),
                                        "reasoning_expected": expect_reasoning,
                                        "prompt_tokens": usage.get("prompt_tokens"),
                                        "completion_tokens": usage.get("completion_tokens"),
                                        "total_tokens": usage.get("total_tokens"),
                                    },
                                )
                            except Exception:
                                pass
                            # Success resets circuit
                            if self._cb_open_until:
                                _obs.info("circuit.reset", extra={"provider": "lmstudio"})
                            self._cb_open_until = 0.0
                            self._cb_failures = 0
                            return result
                    except (aiohttp.ClientError, asyncio.TimeoutError) as ce:
                        # Transport or timeout: backoff if attempts remain
                        if i < attempts - 1:
                            try:
                                import random as _rnd
                                jitter = 0.8 + 0.4 * _rnd.random()
                                await asyncio.sleep(min(base_delay * (2 ** i) * jitter, 2.5))
                            except Exception:
                                pass
                            continue
                        raise GenerationError("lmstudio", ce, {"error": str(ce)})
                # If loop completes without returning, raise last error
                raise GenerationError(
                    "lmstudio",
                    Exception("All attempts failed for LMStudio generation"),
                    {"status": 400},
                )

        except GenerationError as e:
            log_exception(
                _obs,
                code=ErrorCodes.GEN_TIMEOUT if "timeout" in str(e).lower() else ErrorCodes.AI_LB_UNAVAILABLE,
                component="lmstudio",
                exc=e,
                endpoint=self.config.endpoint,
            )
            # Update circuit breaker after generation failure
            try:
                self._cb_failures += 1
                threshold = int(os.getenv("LLM_CB_THRESHOLD", "3") or 3)
                window = float(os.getenv("LLM_CB_WINDOW_SEC", "15") or 15)
                if self._cb_failures >= threshold:
                    self._cb_open_until = monotonic() + window
                    self._cb_failures = 0
                    _obs.info("circuit.open", extra={"provider": "lmstudio", "window_sec": window})
            except Exception:
                pass
            raise
        except Exception as e:
            failure = self.record_failure(e, prompt)
            log_exception(
                _obs,
                code=ErrorCodes.AI_LB_UNAVAILABLE,
                component="lmstudio",
                exc=e,
                endpoint=self.config.endpoint,
            )
            # Update circuit breaker after failure
            try:
                self._cb_failures += 1
                threshold = int(os.getenv("LLM_CB_THRESHOLD", "3") or 3)
                window = float(os.getenv("LLM_CB_WINDOW_SEC", "15") or 15)
                if self._cb_failures >= threshold:
                    self._cb_open_until = monotonic() + window
                    self._cb_failures = 0
                    _obs.info("circuit.open", extra={"provider": "lmstudio", "window_sec": window})
            except Exception:
                pass
            raise GenerationError("lmstudio", e, {"failure": failure.to_dict()})

    async def health_check(self) -> Dict[str, Any]:
        """Check LMStudio availability with details"""
        try:
            url = f"{self.config.endpoint}/v1/models"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5, ssl=self._use_ssl()) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "healthy": True,
                            "models": data.get("data", []),
                            "endpoint": self.config.endpoint,
                        }
                    else:
                        return {
                            "healthy": False,
                            "error": f"HTTP {response.status}",
                            "endpoint": self.config.endpoint,
                        }
        except Exception as e:
            return {"healthy": False, "error": str(e), "endpoint": self.config.endpoint}

    def format_prompt(self, prompt: str, system: str = None) -> str:
        """Format for chat completion"""
        if system:
            return f"{system}\n\n{prompt}"
        return prompt


class KoboldCppProvider(LLMProvider):
    """KoboldCpp API provider"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._cb_failures: int = 0
        self._cb_open_until: float = 0.0

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate using KoboldCpp API"""
        # Circuit breaker: fast-fail when open
        try:
            from time import monotonic as _mono
            now = _mono()
        except Exception:
            now = 0.0
        if getattr(self, "_cb_open_until", 0.0) and now < self._cb_open_until:
            log_exception(
                _obs,
                code=ErrorCodes.AI_LB_UNAVAILABLE,
                component="koboldcpp",
                exc=RuntimeError(f"circuit open until {self._cb_open_until:.2f}"),
                endpoint=self.config.endpoint,
            )
            raise GenerationError("koboldcpp", RuntimeError("circuit_open"), {"until": self._cb_open_until})

        start_time = asyncio.get_event_loop().time()
        url = f"{self.config.endpoint}/api/v1/generate"

        payload = {
            "prompt": prompt,
            "max_context_length": kwargs.get("max_context", 4096),
            "max_length": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "top_k": kwargs.get("top_k", self.config.top_k),
            "rep_pen": kwargs.get("rep_pen", self.config.repetition_penalty),
            "rep_pen_range": kwargs.get("rep_pen_range", 320),
            "sampler_order": kwargs.get("sampler_order", [6, 0, 1, 3, 4, 2, 5]),
            "stop_sequence": self.config.stop_sequences,
            "trim_stop": True,
        }

        payload.update(self.config.extra_params)

        try:
            async with aiohttp.ClientSession() as session:
                # Request event
                try:
                    _obs.info(
                        "llm.request",
                        extra={
                            "provider": "koboldcpp",
                            "endpoint": self.config.endpoint,
                            "model": self.config.model,
                        },
                    )
                except Exception:
                    pass

                attempts = int(os.getenv("KOBOLD_RETRY_ATTEMPTS", "1") or 1)
                base_delay = float(os.getenv("KOBOLD_RETRY_BASE_DELAY", "0.2") or 0.2)
                for attempt in range(max(1, attempts)):
                    try:
                        async with session.post(
                            url,
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                        ) as response:
                            if response.status != 200:
                                error_text = await response.text()
                                # Retry on 5xx
                                if 500 <= response.status < 600 and attempt < attempts - 1:
                                    await asyncio.sleep(base_delay * (2 ** attempt))
                                    continue
                                raise GenerationError(
                                    "koboldcpp",
                                    Exception(f"HTTP {response.status}: {error_text}"),
                                    {"status": response.status, "response": error_text},
                                )

                            data = await response.json()

                            # Validate KoboldCpp response
                            if "results" not in data or len(data["results"]) == 0:
                                raise GenerationError(
                                    "koboldcpp",
                                    ValueError("No results in response"),
                                    {"response": data},
                                )

                            text = data["results"][0].get("text", "")

                            meta = normalize_kobold(data, self.config.model)
                            result = LLMResponse(
                                text=text,
                                provider=ModelProvider.KOBOLDCPP,
                                provider_name="koboldcpp",
                                model=self.config.model,
                                raw_response={"data": data, "normalized": meta},
                                generation_time_ms=(
                                    asyncio.get_event_loop().time() - start_time
                                )
                                * 1000,
                            )
                            try:
                                # from ..common.observability import observe_metric
                                # observe_metric(
                                #     "llm.koboldcpp.gen_ms",
                                #     result.generation_time_ms or 0.0,
                                #     endpoint=self.config.endpoint,
                                #     model=self.config.model,
                                # )
                                pass
                            except Exception:
                                pass
                            return result
                    except (aiohttp.ClientError, asyncio.TimeoutError) as ce:
                        if attempt < attempts - 1:
                            await asyncio.sleep(base_delay * (2 ** attempt))
                            continue
                        raise GenerationError("koboldcpp", ce, {"error": str(ce)})

        except GenerationError:
            # update/open circuit on repeated failures
            try:
                self._cb_failures += 1
                threshold = int(os.getenv("LLM_CB_THRESHOLD", "3") or 3)
                window = float(os.getenv("LLM_CB_WINDOW_SEC", "15") or 15)
                from time import monotonic as _mono
                if self._cb_failures >= threshold:
                    self._cb_open_until = _mono() + window
                    self._cb_failures = 0
                    _obs.info("circuit.open", extra={"provider": "koboldcpp", "window_sec": window})
            except Exception:
                pass
            raise
        except Exception as e:
            failure = self.record_failure(e, prompt)
            log_exception(
                _obs,
                code=ErrorCodes.AI_LB_UNAVAILABLE,
                component="koboldcpp",
                exc=e,
                endpoint=self.config.endpoint,
            )
            raise GenerationError("koboldcpp", e, {"failure": failure.to_dict()})

    async def health_check(self) -> Dict[str, Any]:
        """Check KoboldCpp availability"""
        try:
            url = f"{self.config.endpoint}/api/v1/model"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        result = {
                            "healthy": True,
                            "model": data.get("result"),
                            "endpoint": self.config.endpoint,
                        }
                        # Optional deeper probe behind flag
                        if os.getenv("KOBOLD_HEALTH_GEN", "").strip().lower() in {"1", "true", "yes"}:
                            try:
                                tiny = await self.generate("ping", max_tokens=8, temperature=0.0)
                                result["gen_probe"] = bool(getattr(tiny, "text", ""))
                            except Exception:
                                result["gen_probe"] = False
                        return result
                    else:
                        return {
                            "healthy": False,
                            "error": f"HTTP {response.status}",
                            "endpoint": self.config.endpoint,
                        }
        except Exception as e:
            return {"healthy": False, "error": str(e), "endpoint": self.config.endpoint}

    def format_prompt(self, prompt: str, system: str = None) -> str:
        """Format prompt for KoboldCpp"""
        if system:
            return f"### System: {system}\n### User: {prompt}\n### Assistant:"
        return prompt


def _env_truthy(val: Optional[str]) -> bool:
    if val is None:
        return False
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


class LLMOrchestrator:
    """Orchestrator with strict failure handling - no silent fallbacks"""

    def __init__(self, fail_on_all_providers: bool = True):
        self.providers: Dict[str, LLMProvider] = {}
        self.active_provider: Optional[str] = None
        self.fail_on_all_providers = fail_on_all_providers
        self.generation_history: List[Dict] = []
        self._inflight: Dict[str, asyncio.Task] = {}
        # Preference for small models when selecting from /v1/models
        self.prefer_small_models: bool = _env_truthy(os.environ.get("LM_PREFER_SMALL"))
        # Optional DB logger (no-op unless DB_LOG_GENERATIONS is truthy and env is configured)
        db_uri = os.environ.get("DB_URI")
        if db_uri:
            self._db_logger = GenerationDBLogger(db_uri)
        else:
            self._db_logger = None

    def register_provider(self, name: str, config: LLMConfig) -> None:
        """Register a new LLM provider"""
        if config.provider == ModelProvider.LMSTUDIO:
            provider = LMStudioProvider(config)
        elif config.provider == ModelProvider.KOBOLDCPP:
            provider = KoboldCppProvider(config)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")

        self.providers[name] = provider

        if self.active_provider is None:
            self.active_provider = name

        logger.info(f"Registered provider: {name} ({config.provider.value})")

    def set_active(self, name: str) -> None:
        """Set the active provider"""
        if name not in self.providers:
            raise ValueError(f"Provider {name} not registered")
        self.active_provider = name
        logger.info(f"Active provider set to: {name}")

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        provider_name: str = None,
        allow_fallback: bool = False,
        fallback_providers: List[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """Single-endpoint generation optimized for ai-lb.

        This pared-down generate routes to the active provider only. Health
        check failures are logged as warnings (ai-lb is authoritative for
        node health). No in-process fallback is attempted.
        """
        target = provider_name or self.active_provider
        if not target or target not in self.providers:
            raise ValueError("No active provider set or provider not registered")

        provider = self.providers[target]

        # Lightweight reachability check (warn only)
        health = await provider.health_check()
        if not health.get("healthy", False):
            logger.warning(f"Active provider appears unhealthy: {health}")

        start = asyncio.get_event_loop().time()
        # Single-flight key (provider, system snippet, prompt snippet, model, core params)
        model_key = kwargs.get("model") or getattr(provider.config, "model", None)
        sf_key = "|".join(
            [
                target or "",
                (system or "")[:64],
                (prompt or "")[:128],
                str(model_key or ""),
                str(kwargs.get("temperature", "")),
                str(kwargs.get("max_tokens", "")),
            ]
        )
        try:
            if sf_key in self._inflight:
                response = await self._inflight[sf_key]
            else:
                task = asyncio.create_task(provider.generate(prompt, system=system, **kwargs))
                self._inflight[sf_key] = task
                response = await task
        except Exception as e:
            # Log failure path to DB if enabled, then re-raise
            latency_ms = (asyncio.get_event_loop().time() - start) * 1000
            if self._db_logger:
                event_data = {
                    "provider_name": target,
                    "provider_type": provider.config.provider.value,
                    "provider_endpoint": provider.config.endpoint,
                    "prompt": prompt,
                    "system": system,
                    "request_params": {
                        k: v
                        for k, v in kwargs.items()
                        if k in ("temperature", "max_tokens", "top_p", "top_k")
                    },
                    "response_text": None,
                    "response_json": {"error": str(e)},
                    "status": "error",
                    "latency_ms": latency_ms,
                    "model_key": getattr(provider.config, "model", None),
                }
                self._db_logger.log_event("generations", event_data)

                # Also record a metrics event if fallbacks were requested
                if kwargs.get("allow_fallback") or kwargs.get("fallback_providers"):
                    fallback_event = {
                        "kind": "fallback_considered",
                        "provider_name": target,
                        "provider_type": provider.config.provider.value,
                        "provider_endpoint": provider.config.endpoint,
                        "data": {
                            "allow_fallback": bool(kwargs.get("allow_fallback")),
                            "fallback_providers": kwargs.get("fallback_providers"),
                            "error": str(e),
                        },
                        "model_key": getattr(provider.config, "model", None),
                    }
                    self._db_logger.log_event("events", fallback_event)
            raise
        finally:
            self._inflight.pop(sf_key, None)

        # Successful generation — record history and optionally to DB
        latency_ms = (asyncio.get_event_loop().time() - start) * 1000

        # Record minimal history
        self.generation_history.append(
            {
                "timestamp": response.timestamp,
                "provider": target,
                "prompt_preview": prompt[:100],
                "success": True,
                "generation_time_ms": response.generation_time_ms,
            }
        )

        # DB log (best-effort)
        if self._db_logger:
            event_data = {
                "provider_name": target,
                "provider_type": provider.config.provider.value,
                "provider_endpoint": provider.config.endpoint,
                "prompt": prompt,
                "system": system,
                "request_params": {
                    k: v
                    for k, v in kwargs.items()
                    if k in ("temperature", "max_tokens", "top_p", "top_k")
                },
                "response_text": getattr(response, "text", None),
                "response_json": getattr(response, "raw_response", None),
                "status": "ok",
                "latency_ms": latency_ms,
                "model_key": getattr(response, "model", None)
                or getattr(provider.config, "model", None),
            }
            self._db_logger.log_event("generations", event_data)

        return response

    async def health_check_all(self) -> Dict[str, Dict]:
        """Detailed health check of all providers"""
        results = {}
        for name, provider in self.providers.items():
            results[name] = await provider.health_check()
        return results

    async def list_models_filtered(
        self, prefer_small: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """Aggregate and filter models across providers.

        - Excludes embeddings and speech (TTS/STT) models
        - Optionally prefers small models (<=~4B hints) when ordering
        """
        health = await self.health_check_all()
        all_models: List[Dict[str, Any]] = []
        for info in health.values():
            ms = info.get("models") or []
            if isinstance(ms, list):
                all_models.extend([m for m in ms if isinstance(m, dict)])
        prefer = (
            self.prefer_small_models if prefer_small is None else bool(prefer_small)
        )
        return filter_models(all_models, require_text=True, prefer_small=prefer)

    def get_generation_stats(self) -> Dict:
        """Get statistics about generation history"""
        if not self.generation_history:
            return {"total": 0, "success": 0, "failure": 0}

        success = sum(1 for h in self.generation_history if h.get("success", False))
        failure = len(self.generation_history) - success

        avg_time = None
        if success > 0:
            times = [
                h["generation_time_ms"]
                for h in self.generation_history
                if h.get("success") and h.get("generation_time_ms")
            ]
            if times:
                avg_time = sum(times) / len(times)

        return {
            "total": len(self.generation_history),
            "success": success,
            "failure": failure,
            "success_rate": (
                (success / len(self.generation_history) * 100)
                if self.generation_history
                else 0
            ),
            "avg_generation_time_ms": avg_time,
            "providers_used": list(
                set(
                    h.get("provider", "unknown")
                    for h in self.generation_history
                    if h.get("success")
                )
            ),
        }

    @classmethod
    def from_config_file(
        cls, config_path: str, strict: bool = True
    ) -> "LLMOrchestrator":
        """Create orchestrator from configuration file"""
        with open(config_path, "r") as f:
            config = json.load(f)

        orchestrator = cls(fail_on_all_providers=strict)

        for provider_config in config.get("providers", []):
            name = provider_config.pop("name")
            provider_type = ModelProvider(provider_config.pop("provider"))
            provider_config["require_explicit_success"] = True
            llm_config = LLMConfig(provider=provider_type, **provider_config)
            orchestrator.register_provider(name, llm_config)

        if "active" in config:
            orchestrator.set_active(config["active"])

        return orchestrator


# Test the strict orchestrator
async def test_orchestrator():
    """Test strict orchestration with no silent failures"""

    print("🔒 TESTING LLM ORCHESTRATOR")
    print("=" * 70)
    print("No mock responses, no silent failures")
    print()

    orchestrator = LLMOrchestrator(fail_on_all_providers=True)

    # Register only real providers
    orchestrator.register_provider(
        "kobold",
        LLMConfig(
            provider=ModelProvider.KOBOLDCPP,
            endpoint="http://localhost:5001",
            temperature=0.5,
            max_tokens=200,
            require_explicit_success=True,
        ),
    )

    orchestrator.register_provider(
        "lmstudio",
        LLMConfig(
            provider=ModelProvider.LMSTUDIO,
            endpoint="http://localhost:1234",
            model="gemma-2-27b",
            temperature=0.7,
            max_tokens=200,
            require_explicit_success=True,
        ),
    )

    # Health check
    print("📊 Provider Health Status:")
    health = await orchestrator.health_check_all()
    for name, status in health.items():
        icon = "✅" if status.get("healthy") else "❌"
        print(f"  {icon} {name}: {status}")
    print()

    # Test generation
    test_prompts = [
        "Write one sentence about the ocean.",
        "What is 2+2?",
        "Name three colors.",
    ]

    for prompt in test_prompts:
        print(f"📝 Prompt: {prompt}")
        try:
            # Try with explicit provider
            response = await orchestrator.generate(
                prompt,
                provider_name="kobold",
                allow_fallback=True,
                fallback_providers=["lmstudio"],
            )
            print(f"✅ Response: {response.text[:100]}")
            if response.failures_before_success:
                print(f"   (After {len(response.failures_before_success)} failures)")
        except Exception as e:
            print(f"❌ Failed: {e}")
        print()

    # Show statistics
    stats = orchestrator.get_generation_stats()
    print("📈 Generation Statistics:")
    print(f"  Total attempts: {stats['total']}")
    print(f"  Success: {stats['success']} ({stats['success_rate']:.1f}%)")
    print(f"  Failures: {stats['failure']}")
    if stats["avg_generation_time_ms"]:
        print(f"  Avg time: {stats['avg_generation_time_ms']:.1f}ms")

    print("\n✨ Orchestration test complete!")


if __name__ == "__main__":
    asyncio.run(test_orchestrator())

# Backwards-compat alias expected by examples/tests
StrictLLMOrchestrator = LLMOrchestrator
