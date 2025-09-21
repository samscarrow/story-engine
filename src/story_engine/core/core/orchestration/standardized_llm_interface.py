"""
Standardized LLM Interface
Provides a unified API for all Story Engine components to interact with LLMs
"""

import asyncio
import time as import_time
import logging
from typing import Dict, Any, Optional, List, Protocol, runtime_checkable
from dataclasses import dataclass
from enum import Enum

from .unified_llm_orchestrator import UnifiedLLMOrchestrator, LLMPersona
from .llm_orchestrator import LLMResponse

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Standard types of LLM queries used across the Story Engine"""

    # Character queries
    CHARACTER_RESPONSE = "character_response"
    MULTI_CHARACTER_INTERACTION = "multi_character_interaction"
    GROUP_DYNAMICS = "group_dynamics"

    # Narrative queries
    SCENE_CREATION = "scene_creation"
    DIALOGUE_GENERATION = "dialogue_generation"
    STORY_DEVELOPMENT = "story_development"

    # Analysis queries
    NARRATIVE_ANALYSIS = "narrative_analysis"
    QUALITY_ASSESSMENT = "quality_assessment"
    CONTINUITY_CHECK = "continuity_check"

    # Meta queries
    WORLD_BUILDING = "world_building"
    STORY_ENHANCEMENT = "story_enhancement"
    PLAUSIBILITY_ASSESSMENT = "plausibility_assessment"


@dataclass
class StandardizedQuery:
    """Standardized query structure for all LLM interactions"""

    query_type: QueryType
    data: Dict[str, Any]
    persona_override: Optional[LLMPersona] = None
    temperature_override: Optional[float] = None
    max_tokens_override: Optional[int] = None
    provider_preference: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@runtime_checkable
class LLMInterface(Protocol):
    """Protocol defining the standardized LLM interface"""

    async def query(self, standardized_query: StandardizedQuery) -> LLMResponse:
        """Execute a standardized query"""
        ...

    async def batch_query(self, queries: List[StandardizedQuery]) -> List[LLMResponse]:
        """Execute multiple queries concurrently"""
        ...


class StandardizedLLMInterface:
    """
    Implementation of the standardized LLM interface using persona-based orchestration
    """

    # Mapping of query types to their corresponding personas
    QUERY_TO_PERSONA_MAP = {
        QueryType.CHARACTER_RESPONSE: LLMPersona.CHARACTER_SIMULATOR,
        QueryType.MULTI_CHARACTER_INTERACTION: LLMPersona.GROUP_FACILITATOR,
        QueryType.GROUP_DYNAMICS: LLMPersona.FACTION_MEDIATOR,
        QueryType.SCENE_CREATION: LLMPersona.SCENE_ARCHITECT,
        QueryType.DIALOGUE_GENERATION: LLMPersona.DIALOGUE_COACH,
        QueryType.STORY_DEVELOPMENT: LLMPersona.STORY_DESIGNER,
        QueryType.NARRATIVE_ANALYSIS: LLMPersona.NARRATIVE_ANALYST,
        QueryType.QUALITY_ASSESSMENT: LLMPersona.QUALITY_ASSESSOR,
        QueryType.CONTINUITY_CHECK: LLMPersona.CONTINUITY_CHECKER,
        QueryType.WORLD_BUILDING: LLMPersona.WORLD_BUILDER,
        QueryType.STORY_ENHANCEMENT: LLMPersona.STORY_ENHANCER,
        QueryType.PLAUSIBILITY_ASSESSMENT: LLMPersona.PLAUSIBILITY_JUDGE,
    }

    def __init__(self, unified_orchestrator: UnifiedLLMOrchestrator):
        """
        Initialize with a unified orchestrator

        Args:
            unified_orchestrator: The UnifiedLLMOrchestrator instance
        """
        self.orchestrator = unified_orchestrator
        self.query_history: List[Dict[str, Any]] = []

        # Performance tracking
        self.performance_metrics = {
            "total_queries": 0,
            "query_type_counts": {qt.value: 0 for qt in QueryType},
            "average_response_times": {},
            "success_rates": {},
        }

        logger.info("StandardizedLLMInterface initialized")

    async def query(self, standardized_query: StandardizedQuery) -> LLMResponse:
        """
        Execute a standardized query using the appropriate persona

        Args:
            standardized_query: The standardized query to execute

        Returns:
            LLMResponse from the appropriate persona
        """
        import time

        start_time = time.time()

        # Update metrics
        self.performance_metrics["total_queries"] += 1
        query_type_str = standardized_query.query_type.value
        self.performance_metrics["query_type_counts"][query_type_str] += 1

        try:
            # Determine the persona to use
            persona = (
                standardized_query.persona_override
                or self.QUERY_TO_PERSONA_MAP[standardized_query.query_type]
            )

            # Prepare parameters
            temperature = standardized_query.temperature_override
            max_tokens = standardized_query.max_tokens_override
            provider_name = standardized_query.provider_preference

            # Execute the query
            response = await self.orchestrator.generate_with_persona(
                persona=persona,
                data=standardized_query.data,
                temperature=temperature,
                max_tokens=max_tokens,
                provider_name=provider_name,
                allow_fallback=True,
            )

            # Track performance
            response_time = time.time() - start_time
            self._update_performance_metrics(
                query_type_str, response_time, success=True
            )

            # Add to history
            self._add_to_history(standardized_query, response, response_time)

            return response

        except Exception as e:
            response_time = time.time() - start_time
            self._update_performance_metrics(
                query_type_str, response_time, success=False
            )
            logger.error(f"Error in standardized llm interface: {e}")
            raise

    async def batch_query(
        self, queries: List[StandardizedQuery], max_concurrent: int = 5
    ) -> List[LLMResponse]:
        """
        Execute multiple standardized queries concurrently

        Args:
            queries: List of standardized queries to execute
            max_concurrent: Maximum number of concurrent queries

        Returns:
            List of LLMResponse objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def execute_single_query(query: StandardizedQuery) -> LLMResponse:
            async with semaphore:
                return await self.query(query)

        # Execute all queries concurrently
        tasks = [execute_single_query(query) for query in queries]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def _update_performance_metrics(
        self, query_type: str, response_time: float, success: bool
    ):
        """Update performance metrics"""

        # Update average response time
        if query_type not in self.performance_metrics["average_response_times"]:
            self.performance_metrics["average_response_times"][query_type] = []

        self.performance_metrics["average_response_times"][query_type].append(
            response_time
        )

        # Keep only last 100 response times for moving average
        if len(self.performance_metrics["average_response_times"][query_type]) > 100:
            self.performance_metrics["average_response_times"][query_type] = (
                self.performance_metrics["average_response_times"][query_type][-100:]
            )

        # Update success rates
        if query_type not in self.performance_metrics["success_rates"]:
            self.performance_metrics["success_rates"][query_type] = {
                "total": 0,
                "successes": 0,
            }

        self.performance_metrics["success_rates"][query_type]["total"] += 1
        if success:
            self.performance_metrics["success_rates"][query_type]["successes"] += 1

    def _add_to_history(
        self, query: StandardizedQuery, response: LLMResponse, response_time: float
    ):
        """Add query to history for debugging and analysis"""
        history_entry = {
            "timestamp": import_time.time(),
            "query_type": query.query_type.value,
            "persona_used": getattr(response, "metadata", {}).get("persona", "unknown"),
            "response_time": response_time,
            "success": hasattr(response, "text") or hasattr(response, "content"),
            "data_keys": list(query.data.keys()) if query.data else [],
            "metadata": query.metadata,
        }

        self.query_history.append(history_entry)

        # Keep only last 1000 entries
        if len(self.query_history) > 1000:
            self.query_history = self.query_history[-1000:]

    # Convenience methods for common query patterns

    def create_character_query(
        self,
        character: Dict[str, Any],
        situation: str,
        emphasis: str = "neutral",
        **kwargs,
    ) -> StandardizedQuery:
        """Create a standardized character response query"""
        return StandardizedQuery(
            query_type=QueryType.CHARACTER_RESPONSE,
            data={"character": character, "situation": situation, "emphasis": emphasis},
            **kwargs,
        )

    def create_scene_query(
        self,
        beat: Dict[str, Any],
        characters: List[Dict[str, Any]],
        previous_context: str = "",
        **kwargs,
    ) -> StandardizedQuery:
        """Create a standardized scene creation query"""
        return StandardizedQuery(
            query_type=QueryType.SCENE_CREATION,
            data={
                "beat": beat,
                "characters": characters,
                "previous_context": previous_context,
            },
            **kwargs,
        )

    def create_analysis_query(
        self,
        content: str,
        analysis_type: str = "comprehensive",
        criteria: List[str] = None,
        **kwargs,
    ) -> StandardizedQuery:
        """Create a standardized narrative analysis query"""
        return StandardizedQuery(
            query_type=QueryType.NARRATIVE_ANALYSIS,
            data={
                "content": content,
                "analysis_type": analysis_type,
                "criteria": criteria
                or ["coherence", "pacing", "character_development"],
            },
            **kwargs,
        )

    def create_group_query(
        self,
        characters: List[Dict[str, Any]],
        situation: str,
        group_dynamics: Optional[Dict] = None,
        **kwargs,
    ) -> StandardizedQuery:
        """Create a standardized group interaction query"""
        return StandardizedQuery(
            query_type=QueryType.MULTI_CHARACTER_INTERACTION,
            data={
                "characters": characters,
                "situation": situation,
                "group_dynamics": group_dynamics or {},
            },
            **kwargs,
        )

    def create_dialogue_query(
        self,
        character: Dict[str, Any],
        scene: Dict[str, Any],
        dialogue_context: Dict[str, Any],
        **kwargs,
    ) -> StandardizedQuery:
        """Create a standardized dialogue generation query"""
        return StandardizedQuery(
            query_type=QueryType.DIALOGUE_GENERATION,
            data={"character": character, "scene": scene, "context": dialogue_context},
            **kwargs,
        )

    # Metrics and monitoring

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        # Calculate average response times
        avg_times = {}
        for query_type, times in self.performance_metrics[
            "average_response_times"
        ].items():
            if times:
                avg_times[query_type] = sum(times) / len(times)

        # Calculate success rates
        success_rates = {}
        for query_type, stats in self.performance_metrics["success_rates"].items():
            if stats["total"] > 0:
                success_rates[query_type] = stats["successes"] / stats["total"]

        return {
            "total_queries": self.performance_metrics["total_queries"],
            "query_type_distribution": dict(
                self.performance_metrics["query_type_counts"]
            ),
            "average_response_times": avg_times,
            "success_rates": success_rates,
            "history_size": len(self.query_history),
        }

    def get_recent_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent query history"""
        return self.query_history[-limit:] if self.query_history else []

    def reset_metrics(self):
        """Reset all performance metrics"""
        self.performance_metrics = {
            "total_queries": 0,
            "query_type_counts": {qt.value: 0 for qt in QueryType},
            "average_response_times": {},
            "success_rates": {},
        }
        self.query_history.clear()
        logger.info("Performance metrics reset")


# Migration helpers for existing engines


class LegacyLLMAdapter:
    """
    Adapter to help existing engines migrate to the standardized interface
    """

    def __init__(self, standardized_interface: StandardizedLLMInterface):
        self.interface = standardized_interface

    async def generate_response(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        context_type: str = "general",
        **kwargs,
    ) -> LLMResponse:
        """
        Legacy method that maps to standardized interface
        Tries to infer query type from context or falls back to analysis
        """

        # Try to infer query type from context_type or use analysis as default
        query_type_map = {
            "character": QueryType.CHARACTER_RESPONSE,
            "scene": QueryType.SCENE_CREATION,
            "dialogue": QueryType.DIALOGUE_GENERATION,
            "group": QueryType.MULTI_CHARACTER_INTERACTION,
            "analysis": QueryType.NARRATIVE_ANALYSIS,
            "general": QueryType.NARRATIVE_ANALYSIS,  # fallback
        }

        query_type = query_type_map.get(context_type, QueryType.NARRATIVE_ANALYSIS)

        query = StandardizedQuery(
            query_type=query_type,
            data={"prompt": prompt, **kwargs},
            temperature_override=temperature,
            max_tokens_override=max_tokens,
        )

        return await self.interface.query(query)

    async def call_llm(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.8
    ) -> Optional[Dict]:
        """Legacy call_llm interface"""
        query = StandardizedQuery(
            query_type=QueryType.NARRATIVE_ANALYSIS,
            data={"system_prompt": system_prompt, "user_prompt": user_prompt},
            temperature_override=temperature,
        )

        response = await self.interface.query(query)

        # Convert to expected format
        if hasattr(response, "text") and response.text:
            try:
                import json

                return json.loads(response.text)
            except (json.JSONDecodeError, ValueError):
                return {"content": response.text}

        return None


# Factory function
def create_standardized_interface(
    unified_orchestrator: UnifiedLLMOrchestrator,
) -> StandardizedLLMInterface:
    """Create a StandardizedLLMInterface instance"""
    return StandardizedLLMInterface(unified_orchestrator)


# Import fix retained at top as `import_time`
