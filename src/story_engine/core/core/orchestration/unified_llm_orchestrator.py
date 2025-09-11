"""
Unified LLM Orchestrator with Persona-Based POML Templates
Standardizes all LLM queries across the Story Engine project
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass

from story_engine.poml.lib.poml_integration import POMLEngine, StoryEnginePOMLAdapter
from .llm_orchestrator import LLMOrchestrator, LLMResponse

logger = logging.getLogger(__name__)

class LLMPersona(Enum):
    """Defined personas for different types of LLM interactions"""
    
    # Character-focused personas
    CHARACTER_SIMULATOR = "character_simulator"      # Individual character responses
    GROUP_FACILITATOR = "group_facilitator"          # Multi-character interactions
    FACTION_MEDIATOR = "faction_mediator"            # Complex group dynamics
    
    # Narrative-focused personas
    SCENE_ARCHITECT = "scene_architect"              # Scene creation and design
    DIALOGUE_COACH = "dialogue_coach"                # Character dialogue generation
    STORY_DESIGNER = "story_designer"                # Plot and arc development
    
    # Meta-narrative personas
    NARRATIVE_ANALYST = "narrative_analyst"          # Story analysis and evaluation
    QUALITY_ASSESSOR = "quality_assessor"            # Content quality evaluation
    WORLD_BUILDER = "world_builder"                  # World state management
    
    # Enhancement personas
    STORY_ENHANCER = "story_enhancer"                # Content improvement
    CONTINUITY_CHECKER = "continuity_checker"        # Consistency validation
    PLAUSIBILITY_JUDGE = "plausibility_judge"        # Realism assessment

@dataclass
class PersonaConfig:
    """Configuration for each LLM persona"""
    persona: LLMPersona
    template_path: str
    default_temperature: float = 0.7
    default_max_tokens: int = 500
    system_prompt_template: Optional[str] = None
    response_format: str = "text"  # "text" or "json"
    validation_schema: Optional[Dict] = None

class UnifiedLLMOrchestrator:
    """Unified orchestrator that standardizes all LLM queries using personas and POML templates."""

    def __init__(
        self,
        orchestrator: LLMOrchestrator,
        poml_config: Optional[Dict] = None,
        persona_config_path: str = "personas.yaml",
    ):
        self.orchestrator = orchestrator
        self.poml = POMLEngine(config_file=poml_config.get('config_file') if poml_config else None)
        self.adapter = StoryEnginePOMLAdapter(self.poml)
        
        # Initialize persona configurations
        self.persona_configs = self._load_persona_configs_from_file(persona_config_path)
        
        # Track usage metrics
        self.metrics = {
            'total_calls': 0,
            'persona_usage': {persona.value: 0 for persona in LLMPersona},
            'template_usage': {},
            'success_rate': {},
        }
        
        logger.info(f"UnifiedLLMOrchestrator initialized with {len(self.persona_configs)} personas")

    @classmethod
    def from_env_and_config(
        cls,
        config: Optional[Dict[str, Any]] = None,
        persona_config_path: str = "personas.yaml"
    ) -> "UnifiedLLMOrchestrator":
        """Construct a UnifiedLLMOrchestrator using the engine's YAML/JSON config.

        - Prefers `config.yaml` via `create_orchestrator_from_yaml`, which by default
          points the active provider to the ai-lb endpoint (e.g., http://localhost:8000).
        - Falls back to legacy `llm_config.json` if YAML load fails.
        - Accepts an optional in-memory `config` dict for POML/persona settings.
        """
        try:
            from story_engine.core.orchestration.orchestrator_loader import (
                create_orchestrator_from_yaml,
            )
            orch = create_orchestrator_from_yaml("config.yaml")
        except Exception:
            from .llm_orchestrator import LLMOrchestrator
            orch = LLMOrchestrator.from_config_file("llm_config.json")

        poml_cfg = None
        if isinstance(config, dict):
            # Thread through optional POML config if present
            poml_cfg = (config or {}).get("poml") or (config or {}).get("templates")

        return cls(orch, poml_cfg, persona_config_path=persona_config_path)

    # Lightweight passthroughs to underlying LLMOrchestrator to maintain
    # compatibility with callers that expect an object exposing `.generate(...)`.
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        provider_name: Optional[str] = None,
        allow_fallback: bool = False,
        fallback_providers: Optional[List[str]] = None,
        **kwargs,
    ) -> LLMResponse:
        return await self.orchestrator.generate(
            prompt=prompt,
            system=system,
            provider_name=provider_name,
            allow_fallback=allow_fallback,
            fallback_providers=fallback_providers or [],
            **kwargs,
        )
    
    def _load_persona_configs_from_file(self, config_path: str) -> Dict[LLMPersona, PersonaConfig]:
        """Load persona configurations from a YAML file."""
        try:
            import yaml
            with open(config_path, 'r') as f:
                raw_configs = yaml.safe_load(f)
            
            configs = {}
            for persona_name, config_data in raw_configs.get('personas', {}).items():
                try:
                    persona_enum = LLMPersona(persona_name)
                    configs[persona_enum] = PersonaConfig(persona=persona_enum, **config_data)
                except ValueError:
                    logger.warning(f"Skipping unknown persona '{persona_name}' in config file.")
            return configs
        except (IOError, ImportError) as e:
            logger.error(f"Could not load persona config file '{config_path}': {e}. Using empty config.")
            return {}
    
    async def generate_with_persona(
        self,
        persona: LLMPersona,
        data: Dict[str, Any],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        provider_name: Optional[str] = None,
        allow_fallback: bool = True,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response using a specific persona and its POML template
        
        Args:
            persona: The LLM persona to use
            data: Data context for the POML template
            temperature: Override default temperature
            max_tokens: Override default max tokens
            provider_name: Specific provider to use
            allow_fallback: Allow fallback to other providers
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse object with generated content
        """
        config = self.persona_configs.get(persona)
        if not config:
            raise ValueError(f"Unknown persona: {persona}")
        
        # Update metrics
        self.metrics['total_calls'] += 1
        self.metrics['persona_usage'][persona.value] += 1
        
        try:
            # Render the POML template, preferring role-separated prompts when enabled
            import os as _os
            _roles_enabled = str(_os.environ.get('POML_ROLES') or '').lower() in {"1","true","yes","on"}
            if _roles_enabled:
                roles = self.poml.render_roles(config.template_path, data)
                user_prompt = roles.get('user', '')
                system_prompt = roles.get('system', None)
            else:
                user_prompt = self.poml.render(config.template_path, data)
                system_prompt = None
            
            # Track template usage
            template_name = config.template_path
            self.metrics['template_usage'][template_name] = \
                self.metrics['template_usage'].get(template_name, 0) + 1
            
            # Use persona defaults or overrides
            final_temperature = temperature if temperature is not None else config.default_temperature
            final_max_tokens = max_tokens if max_tokens is not None else config.default_max_tokens
            
            # Generate with orchestrator
            response = await self.orchestrator.generate(
                prompt=user_prompt,
                system=system_prompt,
                provider_name=provider_name,
                allow_fallback=allow_fallback,
                temperature=final_temperature,
                max_tokens=final_max_tokens,
                **kwargs
            )
            
            # Track success
            self.metrics['success_rate'][persona.value] = \
                self.metrics['success_rate'].get(persona.value, 0) + 1
            
            # Add metadata about the persona used
            if hasattr(response, 'metadata'):
                response.metadata = response.metadata or {}
                response.metadata.update({
                    'persona': persona.value,
                    'template': config.template_path,
                    'temperature': final_temperature,
                    'max_tokens': final_max_tokens
                })
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate with persona {persona.value}: {e}")
            raise
    
    # Convenience methods for each persona
    
    async def simulate_character(
        self,
        character: Dict[str, Any],
        situation: str,
        emphasis: str = "neutral",
        **kwargs
    ) -> LLMResponse:
        """Character simulation using CHARACTER_SIMULATOR persona"""
        return await self.generate_with_persona(
            LLMPersona.CHARACTER_SIMULATOR,
            {
                'character': character,
                'situation': situation,
                'emphasis': emphasis
            },
            **kwargs
        )
    
    async def facilitate_group(
        self,
        characters: List[Dict[str, Any]],
        situation: str,
        group_dynamics: Optional[Dict] = None,
        **kwargs
    ) -> LLMResponse:
        """Group interaction using GROUP_FACILITATOR persona"""
        return await self.generate_with_persona(
            LLMPersona.GROUP_FACILITATOR,
            {
                'characters': characters,
                'situation': situation,
                'group_dynamics': group_dynamics or {}
            },
            **kwargs
        )
    
    async def mediate_factions(
        self,
        factions: List[Dict[str, Any]],
        conflict: str,
        political_context: Dict[str, Any],
        **kwargs
    ) -> LLMResponse:
        """Faction dynamics using FACTION_MEDIATOR persona"""
        return await self.generate_with_persona(
            LLMPersona.FACTION_MEDIATOR,
            {
                'factions': factions,
                'conflict': conflict,
                'political_context': political_context
            },
            **kwargs
        )
    
    async def craft_scene(
        self,
        beat: Dict[str, Any],
        characters: List[Dict[str, Any]],
        previous_context: str = "",
        **kwargs
    ) -> LLMResponse:
        """Scene creation using SCENE_ARCHITECT persona"""
        return await self.generate_with_persona(
            LLMPersona.SCENE_ARCHITECT,
            {
                'beat': beat,
                'characters': characters,
                'previous_context': previous_context
            },
            **kwargs
        )
    
    async def coach_dialogue(
        self,
        character: Dict[str, Any],
        scene: Dict[str, Any],
        dialogue_context: Dict[str, Any],
        **kwargs
    ) -> LLMResponse:
        """Dialogue generation using DIALOGUE_COACH persona"""
        return await self.generate_with_persona(
            LLMPersona.DIALOGUE_COACH,
            {
                'character': character,
                'scene': scene,
                'context': dialogue_context
            },
            **kwargs
        )
    
    async def design_story(
        self,
        plot_elements: Dict[str, Any],
        characters: List[Dict[str, Any]],
        story_structure: str = "three_act",
        **kwargs
    ) -> LLMResponse:
        """Story design using STORY_DESIGNER persona"""
        return await self.generate_with_persona(
            LLMPersona.STORY_DESIGNER,
            {
                'plot_elements': plot_elements,
                'characters': characters,
                'structure': story_structure
            },
            **kwargs
        )
    
    async def analyze_narrative(
        self,
        content: str,
        analysis_type: str = "structure",
        criteria: List[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Narrative analysis using NARRATIVE_ANALYST persona"""
        return await self.generate_with_persona(
            LLMPersona.NARRATIVE_ANALYST,
            {
                'content': content,
                'analysis_type': analysis_type,
                'criteria': criteria or ['coherence', 'pacing', 'character_development']
            },
            **kwargs
        )
    
    async def assess_quality(
        self,
        content: str,
        metrics: List[str],
        standards: Dict[str, Any] = None,
        **kwargs
    ) -> LLMResponse:
        """Quality assessment using QUALITY_ASSESSOR persona"""
        return await self.generate_with_persona(
            LLMPersona.QUALITY_ASSESSOR,
            {
                'content': content,
                'metrics': metrics,
                'standards': standards or {}
            },
            **kwargs
        )
    
    async def build_world(
        self,
        world_elements: Dict[str, Any],
        characters: List[str] = None,
        location: str = "",
        **kwargs
    ) -> LLMResponse:
        """World building using WORLD_BUILDER persona"""
        return await self.generate_with_persona(
            LLMPersona.WORLD_BUILDER,
            {
                'world_elements': world_elements,
                'characters': characters or [],
                'location': location
            },
            **kwargs
        )
    
    async def enhance_story(
        self,
        content: str,
        enhancement_focus: List[str],
        style_guide: Dict[str, Any] = None,
        **kwargs
    ) -> LLMResponse:
        """Story enhancement using STORY_ENHANCER persona"""
        return await self.generate_with_persona(
            LLMPersona.STORY_ENHANCER,
            {
                'content': content,
                'focus_areas': enhancement_focus,
                'style_guide': style_guide or {}
            },
            **kwargs
        )
    
    async def check_continuity(
        self,
        current_content: str,
        previous_content: str,
        continuity_elements: List[str],
        **kwargs
    ) -> LLMResponse:
        """Continuity checking using CONTINUITY_CHECKER persona"""
        return await self.generate_with_persona(
            LLMPersona.CONTINUITY_CHECKER,
            {
                'current': current_content,
                'previous': previous_content,
                'elements': continuity_elements
            },
            **kwargs
        )
    
    async def judge_plausibility(
        self,
        scenario: str,
        context: Dict[str, Any],
        plausibility_criteria: List[str],
        **kwargs
    ) -> LLMResponse:
        """Plausibility assessment using PLAUSIBILITY_JUDGE persona"""
        return await self.generate_with_persona(
            LLMPersona.PLAUSIBILITY_JUDGE,
            {
                'scenario': scenario,
                'context': context,
                'criteria': plausibility_criteria
            },
            **kwargs
        )
    
    # Batch processing
    
    async def batch_generate_with_personas(
        self,
        requests: List[Dict[str, Any]],
        max_concurrent: int = 5
    ) -> List[LLMResponse]:
        """
        Process multiple persona-based requests concurrently
        
        Args:
            requests: List of dicts with 'persona', 'data', and optional parameters
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of LLMResponse objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_request(request: Dict[str, Any]) -> LLMResponse:
            async with semaphore:
                persona = LLMPersona(request['persona'])
                data = request['data']
                kwargs = {k: v for k, v in request.items() 
                         if k not in ['persona', 'data']}
                
                return await self.generate_with_persona(persona, data, **kwargs)
        
        tasks = [process_request(req) for req in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    # Metrics and monitoring
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get usage metrics for monitoring"""
        return {
            'total_calls': self.metrics['total_calls'],
            'persona_usage': dict(self.metrics['persona_usage']),
            'template_usage': dict(self.metrics['template_usage']),
            'success_rates': {
                persona: (successes / self.metrics['persona_usage'][persona] 
                         if self.metrics['persona_usage'][persona] > 0 else 0.0)
                for persona, successes in self.metrics['success_rate'].items()
            },
            'cache_stats': self.poml.cache.cache if self.poml.cache else {}
        }
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics = {
            'total_calls': 0,
            'persona_usage': {persona.value: 0 for persona in LLMPersona},
            'template_usage': {},
            'success_rate': {},
        }
    
    # Configuration management
    
    def update_persona_config(self, persona: LLMPersona, **updates):
        """Update configuration for a specific persona"""
        if persona in self.persona_configs:
            config = self.persona_configs[persona]
            for key, value in updates.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            logger.info(f"Updated config for {persona.value}: {updates}")
    
    def get_persona_config(self, persona: LLMPersona) -> Optional[PersonaConfig]:
        """Get configuration for a specific persona"""
        return self.persona_configs.get(persona)
    
    def list_personas(self) -> List[str]:
        """List all available personas"""
        return [persona.value for persona in LLMPersona]

# Convenience factory function
def create_unified_orchestrator(
    orchestrator: LLMOrchestrator,
    poml_config: Optional[Dict] = None
) -> UnifiedLLMOrchestrator:
    """Create a UnifiedLLMOrchestrator with default configuration"""
    return UnifiedLLMOrchestrator(orchestrator, poml_config)

# Legacy compatibility adapter
class LegacyLLMAdapter:
    """Adapter to maintain compatibility with existing LLM interfaces"""
    
    def __init__(self, unified_orchestrator: UnifiedLLMOrchestrator):
        self.orchestrator = unified_orchestrator
    
    async def generate_response(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        **kwargs
    ) -> Any:
        """Legacy generate_response interface"""
        # Use a generic persona for legacy calls
        response = await self.orchestrator.generate_with_persona(
            LLMPersona.NARRATIVE_ANALYST,  # Default persona
            {'prompt': prompt},
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return response
    
    async def call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.8
    ) -> Optional[Dict]:
        """Legacy call_llm interface"""
        response = await self.orchestrator.orchestrator.generate(
            prompt=user_prompt,
            system=system_prompt,
            temperature=temperature,
            allow_fallback=True
        )
        
        # Convert to expected format
        if hasattr(response, 'text') and response.text:
            try:
                import json
                return json.loads(response.text)
            except (json.JSONDecodeError, ValueError):
                return {'content': response.text}
        
        return None
