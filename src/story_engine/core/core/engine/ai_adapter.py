from __future__ import annotations

from typing import Any, Optional

from ..orchestration.standardized_llm_interface import (
    StandardizedLLMInterface,
    StandardizedQuery,
    QueryType,
)


class StandardizedAIAdapter:
    """Adapter exposing a `.generate(...)` method over StandardizedLLMInterface.

    This allows EngineContext.ai to rely on the standardized pathway while
    keeping the simple `generate(prompt=..., **kwargs)` signature used across
    the project.
    """

    def __init__(
        self,
        interface: StandardizedLLMInterface,
        default_query: QueryType = QueryType.NARRATIVE_ANALYSIS,
    ) -> None:
        self.interface = interface
        self.default_query = default_query

    async def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        provider_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        data = {"prompt": prompt}
        if system:
            data["system"] = system
        if kwargs:
            data.update(kwargs)
        q = StandardizedQuery(
            query_type=self.default_query,
            data=data,
            temperature_override=temperature,
            max_tokens_override=max_tokens,
            provider_preference=provider_name,
        )
        return await self.interface.query(q)
