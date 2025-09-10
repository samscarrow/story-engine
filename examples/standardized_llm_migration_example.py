"""
Example: Migrating Story Engine Components to Standardized LLM Interface
Shows how to migrate existing engines to use the new standardized LLM interface with personas
"""

import logging
from typing import Dict, Any, List, Optional

# Import the standardized components
from story_engine.core.orchestration.llm_orchestrator import StrictLLMOrchestrator
from story_engine.core.orchestration.unified_llm_orchestrator import create_unified_orchestrator
from story_engine.core.orchestration.standardized_llm_interface import (
    StandardizedLLMInterface, 
    create_standardized_interface,
    LegacyLLMAdapter
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MigratedCharacterSimulationEngine:
    """
    Example of migrating CharacterSimulationEngine to use standardized LLM interface
    """
    
    def __init__(self, standardized_interface: StandardizedLLMInterface):
        self.llm_interface = standardized_interface
        self.simulation_history = []
        
        logger.info("MigratedCharacterSimulationEngine initialized with standardized interface")
    
    async def run_character_simulation(
        self,
        character: Dict[str, Any],
        situation: str,
        emphasis: str = "neutral",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run character simulation using standardized interface
        
        Before: Used hardcoded string prompts
        After: Uses CHARACTER_SIMULATOR persona with POML template
        """
        
        # Create standardized query
        query = self.llm_interface.create_character_query(
            character=character,
            situation=situation,
            emphasis=emphasis,
            temperature_override=kwargs.get('temperature'),
            max_tokens_override=kwargs.get('max_tokens')
        )
        
        # Execute query using appropriate persona
        response = await self.llm_interface.query(query)
        
        # Process response
        result = {
            'character_id': character.get('id', 'unknown'),
            'situation': situation,
            'emphasis': emphasis,
            'response': response.text if hasattr(response, 'text') else response.content,
            'metadata': {
                'persona_used': getattr(response, 'metadata', {}).get('persona'),
                'temperature': getattr(response, 'metadata', {}).get('temperature'),
                'timestamp': import_time.time()
            }
        }
        
        # Add to history
        self.simulation_history.append(result)
        
        return result
    
    async def run_group_simulation(
        self,
        characters: List[Dict[str, Any]],
        situation: str,
        group_dynamics: Optional[Dict] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run multi-character group simulation
        
        Before: Used complex string concatenation for multiple characters
        After: Uses GROUP_FACILITATOR persona with structured interaction handling
        """
        
        query = self.llm_interface.create_group_query(
            characters=characters,
            situation=situation,
            group_dynamics=group_dynamics,
            temperature_override=kwargs.get('temperature'),
            max_tokens_override=kwargs.get('max_tokens')
        )
        
        response = await self.llm_interface.query(query)
        
        return {
            'characters_involved': [c.get('id', 'unknown') for c in characters],
            'situation': situation,
            'group_dynamics': group_dynamics or {},
            'group_response': response.text if hasattr(response, 'text') else response.content,
            'metadata': {
                'persona_used': getattr(response, 'metadata', {}).get('persona'),
                'participant_count': len(characters),
                'timestamp': import_time.time()
            }
        }

class MigratedNarrativePipeline:
    """
    Example of migrating NarrativePipeline to use standardized LLM interface
    """
    
    def __init__(self, standardized_interface: StandardizedLLMInterface):
        self.llm_interface = standardized_interface
        self.scenes = []
        
        logger.info("MigratedNarrativePipeline initialized with standardized interface")
    
    async def craft_scene(
        self,
        beat: Dict[str, Any],
        characters: List[Dict[str, Any]],
        previous_context: str = "",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Craft a scene using SCENE_ARCHITECT persona
        
        Before: Used hardcoded f-string prompt
        After: Uses SCENE_ARCHITECT persona with structured scene design
        """
        
        query = self.llm_interface.create_scene_query(
            beat=beat,
            characters=characters,
            previous_context=previous_context,
            temperature_override=kwargs.get('temperature', 0.8)
        )
        
        response = await self.llm_interface.query(query)
        
        scene_descriptor = {
            'id': len(self.scenes),
            'beat_name': beat.get('name', 'Unnamed Beat'),
            'situation': response.text if hasattr(response, 'text') else response.content,
            'characters': [c.get('name', 'Unknown') for c in characters],
            'previous_context': previous_context,
            'metadata': {
                'tension_level': beat.get('tension', 5),
                'persona_used': getattr(response, 'metadata', {}).get('persona'),
                'timestamp': import_time.time()
            }
        }
        
        self.scenes.append(scene_descriptor)
        return scene_descriptor
    
    async def generate_character_dialogue(
        self,
        character: Dict[str, Any],
        scene: Dict[str, Any],
        dialogue_context: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate character dialogue using DIALOGUE_COACH persona
        
        Before: Basic prompt with character context
        After: Uses DIALOGUE_COACH persona for natural dialogue generation
        """
        
        query = self.llm_interface.create_dialogue_query(
            character=character,
            scene=scene,
            dialogue_context=dialogue_context,
            temperature_override=kwargs.get('temperature', 0.9)
        )
        
        response = await self.llm_interface.query(query)
        
        return {
            'character_id': character.get('id', 'unknown'),
            'scene_id': scene.get('id', 'unknown'),
            'dialogue': response.text if hasattr(response, 'text') else response.content,
            'context': dialogue_context,
            'metadata': {
                'persona_used': getattr(response, 'metadata', {}).get('persona'),
                'timestamp': import_time.time()
            }
        }
    
    async def analyze_scene_quality(
        self,
        scene_content: str,
        analysis_criteria: List[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze scene quality using NARRATIVE_ANALYST persona
        
        Before: No systematic scene analysis
        After: Uses NARRATIVE_ANALYST persona for comprehensive evaluation
        """
        
        query = self.llm_interface.create_analysis_query(
            content=scene_content,
            analysis_type="scene_analysis",
            criteria=analysis_criteria or ['dramatic_tension', 'character_development', 'pacing'],
            temperature_override=kwargs.get('temperature', 0.3)
        )
        
        response = await self.llm_interface.query(query)
        
        return {
            'scene_content_length': len(scene_content),
            'analysis_criteria': analysis_criteria,
            'analysis_result': response.text if hasattr(response, 'text') else response.content,
            'metadata': {
                'persona_used': getattr(response, 'metadata', {}).get('persona'),
                'timestamp': import_time.time()
            }
        }

class LegacyEngineAdapter:
    """
    Example of how to provide backward compatibility for engines that can't be immediately migrated
    """
    
    def __init__(self, standardized_interface: StandardizedLLMInterface):
        self.legacy_adapter = LegacyLLMAdapter(standardized_interface)
        
        logger.info("LegacyEngineAdapter initialized for backward compatibility")
    
    async def legacy_generate_response(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        **kwargs
    ):
        """
        Maintain legacy generate_response interface while using new backend
        """
        # Try to determine context from the prompt content
        context_type = "general"
        if "character" in prompt.lower() and "respond" in prompt.lower():
            context_type = "character"
        elif "scene" in prompt.lower():
            context_type = "scene"
        elif "dialogue" in prompt.lower():
            context_type = "dialogue"
        
        return await self.legacy_adapter.generate_response(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            context_type=context_type,
            **kwargs
        )
    
    async def legacy_call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.8
    ):
        """
        Maintain legacy call_llm interface
        """
        return await self.legacy_adapter.call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature
        )

async def demonstrate_migration():
    """
    Demonstrate the migration process and show the benefits
    """
    
    logger.info("Starting migration demonstration...")
    
    # 1. Setup the standardized infrastructure
    
    # Create the base orchestrator (this would normally come from your existing setup)
    providers = {
        'kobold': {'endpoint': 'http://localhost:5001', 'api_type': 'openai'},
        # Add other providers as needed
    }
    
    base_orchestrator = StrictLLMOrchestrator(providers)
    unified_orchestrator = create_unified_orchestrator(base_orchestrator)
    standardized_interface = create_standardized_interface(unified_orchestrator)
    
    # 2. Initialize migrated engines
    
    character_engine = MigratedCharacterSimulationEngine(standardized_interface)
    narrative_pipeline = MigratedNarrativePipeline(standardized_interface)
    legacy_adapter = LegacyEngineAdapter(standardized_interface)
    
    # 3. Example character data
    
    test_character = {
        'id': 'pontius_pilate',
        'name': 'Pontius Pilate',
        'traits': ['pragmatic', 'cautious', 'politically_minded'],
        'emotional_state': {
            'anger': 0.2,
            'doubt': 0.7,
            'fear': 0.5,
            'compassion': 0.3,
            'confidence': 0.4
        },
        'backstory': {
            'origin': 'Roman nobility',
            'career': 'Prefect of Judaea'
        }
    }
    
    test_situation = "The crowd grows violent, demanding blood for justice."
    
    # 4. Demonstrate character simulation with standardized interface
    
    logger.info("Running character simulation with standardized interface...")
    
    try:
        character_result = await character_engine.run_character_simulation(
            character=test_character,
            situation=test_situation,
            emphasis="doubt",
            temperature=0.8
        )
        
        logger.info("Character simulation successful!")
        logger.info(f"Persona used: {character_result['metadata']['persona_used']}")
        logger.info(f"Response length: {len(character_result['response'])}")
        
    except Exception as e:
        logger.error(f"Character simulation failed: {e}")
    
    # 5. Demonstrate scene creation
    
    logger.info("Creating scene with SCENE_ARCHITECT persona...")
    
    test_beat = {
        'name': 'The Judgment',
        'purpose': 'Force Pilate to make a crucial decision',
        'tension': 9
    }
    
    try:
        scene_result = await narrative_pipeline.craft_scene(
            beat=test_beat,
            characters=[test_character],
            previous_context="Pilate has just finished questioning the accused privately."
        )
        
        logger.info("Scene creation successful!")
        logger.info(f"Scene ID: {scene_result['id']}")
        logger.info(f"Tension level: {scene_result['metadata']['tension_level']}")
        
    except Exception as e:
        logger.error(f"Scene creation failed: {e}")
    
    # 6. Demonstrate group interaction
    
    logger.info("Running group simulation...")
    
    crowd_character = {
        'id': 'crowd',
        'name': 'The Crowd',
        'traits': ['angry', 'demanding', 'unified_in_purpose']
    }
    
    try:
        group_result = await character_engine.run_group_simulation(
            characters=[test_character, crowd_character],
            situation=test_situation,
            group_dynamics={'tension': 8, 'power_balance': 'crowd_dominant'}
        )
        
        logger.info("Group simulation successful!")
        logger.info(f"Participants: {group_result['characters_involved']}")
        
    except Exception as e:
        logger.error(f"Group simulation failed: {e}")
    
    # 7. Demonstrate backward compatibility
    
    logger.info("Testing backward compatibility...")
    
    try:
        legacy_result = await legacy_adapter.legacy_generate_response(
            prompt="You are Pontius Pilate. The crowd is demanding justice. How do you respond?",
            temperature=0.7,
            max_tokens=400
        )
        
        logger.info("Legacy compatibility successful!")
        logger.info(f"Legacy response received: {bool(hasattr(legacy_result, 'text') or hasattr(legacy_result, 'content'))}")
        
    except Exception as e:
        logger.error(f"Legacy compatibility failed: {e}")
    
    # 8. Show performance metrics
    
    performance = standardized_interface.get_performance_summary()
    logger.info("Performance Summary:")
    logger.info(f"  Total queries: {performance['total_queries']}")
    logger.info(f"  Query distribution: {performance['query_type_distribution']}")
    logger.info(f"  Average response times: {performance.get('average_response_times', {})}")
    
    # 9. Demonstrate batch processing
    
    logger.info("Testing batch processing...")
    
    batch_queries = [
        standardized_interface.create_character_query(
            character=test_character,
            situation="A messenger arrives with urgent news",
            emphasis="curiosity"
        ),
        standardized_interface.create_analysis_query(
            content="The scene unfolds with dramatic tension as characters face impossible choices.",
            analysis_type="structure"
        ),
        standardized_interface.create_scene_query(
            beat={'name': 'The Decision', 'purpose': 'Resolution', 'tension': 10},
            characters=[test_character]
        )
    ]
    
    try:
        batch_results = await standardized_interface.batch_query(
            batch_queries,
            max_concurrent=3
        )
        
        successful_results = [r for r in batch_results if not isinstance(r, Exception)]
        logger.info(f"Batch processing: {len(successful_results)}/{len(batch_queries)} queries successful")
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
    
    logger.info("Migration demonstration complete!")
    
    return {
        'standardized_interface': standardized_interface,
        'character_engine': character_engine,
        'narrative_pipeline': narrative_pipeline,
        'performance': performance
    }

# Benefits summary function
def print_migration_benefits():
    """
    Print the benefits of migrating to the standardized LLM interface
    """
    
    benefits = """
    
    🎯 BENEFITS OF STANDARDIZED LLM INTERFACE:
    
    ✅ CONSISTENCY:
       - All LLM interactions use the same interface
       - Standardized query types and response formats
       - Consistent error handling and retry logic
    
    ✅ PERSONA-BASED DESIGN:
       - Each query type uses an appropriate LLM persona
       - POML templates ensure consistent, high-quality prompts
       - Persona-specific configurations (temperature, tokens, etc.)
    
    ✅ MAINTAINABILITY:
       - Single point of control for all LLM interactions
       - Easy to update prompts without touching engine code
       - Centralized performance monitoring and logging
    
    ✅ PERFORMANCE:
       - Built-in caching and optimization
       - Batch processing capabilities
       - Automatic fallback and retry mechanisms
    
    ✅ MIGRATION PATH:
       - Legacy adapters provide backward compatibility
       - Gradual migration with feature flags
       - No need to rewrite engines all at once
    
    ✅ MONITORING:
       - Comprehensive metrics and performance tracking
       - Query history and debugging capabilities
       - Success rates and response time monitoring
    
    ✅ FLEXIBILITY:
       - Easy to add new query types and personas
       - Provider-agnostic (works with any LLM backend)
       - Configurable per-query parameters
    
    """
    
    print(benefits)

if __name__ == "__main__":
    # Run the demonstration
    print_migration_benefits()
    
    # Note: Uncomment the line below to run the actual demonstration
    # This requires a running LLM instance (KoboldCpp, LMStudio, etc.)
    # asyncio.run(demonstrate_migration())
    
    print("\n🚀 Ready to migrate your Story Engine to standardized LLM interface!")

import time as import_time
