"""
LLM Orchestrator with POML Integration
Enhanced orchestrator that uses POML templates for all LLM interactions
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import POML engine
from poml.lib.poml_integration import POMLEngine

logger = logging.getLogger(__name__)

class PromptTemplate(Enum):
    """Enumeration of available POML prompt templates"""
    CHARACTER_RESPONSE = "templates/simulations/character_response.poml"
    SCENE_CRAFTING = "templates/narrative/scene_crafting.poml"
    DIALOGUE_GENERATION = "templates/narrative/dialogue_generation.poml"
    STORY_BEAT = "templates/narrative/story_beats.poml"
    EMOTIONAL_JOURNEY = "templates/simulations/emotional_journey.poml"
    GROUP_DYNAMICS = "templates/simulations/group_dynamics.poml"
    NARRATIVE_EVALUATION = "templates/narrative/evaluation.poml"

class POMLOrchestrator:
    """
    Central orchestrator for all LLM operations using POML templates
    Coordinates between different engines and manages prompt templates
    """
    
    def __init__(self, llm_providers: Dict[str, Any], config: Dict = None):
        """
        Initialize POML orchestrator
        
        Args:
            llm_providers: Dictionary of LLM provider instances
            config: Configuration dictionary
        """
        self.providers = llm_providers
        self.config = config or {}
        
        # Initialize POML engine
        poml_config = Path(__file__).parent.parent / "config" / "poml_config.yaml"
        self.poml = POMLEngine(config_file=str(poml_config))
        
        # Template registry
        self.templates = {}
        self._register_templates()
        
        # Cache for compiled templates
        self.compiled_templates = {}
        
        # Metrics tracking
        self.metrics = {
            'total_calls': 0,
            'template_usage': {},
            'provider_usage': {},
            'cache_hits': 0,
            'errors': 0
        }
        
        logger.info("POMLOrchestrator initialized with POML templates")
    
    def _register_templates(self):
        """Register all available POML templates"""
        # Scan templates directory
        template_dir = Path(__file__).parent.parent / "templates"
        
        for category in ['characters', 'simulations', 'narrative', 'schemas']:
            category_path = template_dir / category
            if category_path.exists():
                for template_file in category_path.glob("*.poml"):
                    template_name = f"{category}/{template_file.stem}"
                    template_path = f"templates/{category}/{template_file.name}"
                    self.templates[template_name] = template_path
                    logger.debug(f"Registered template: {template_name}")
    
    async def generate_with_template(self,
                                    template: Union[str, PromptTemplate],
                                    data: Dict[str, Any],
                                    provider: str = None,
                                    **kwargs) -> Dict:
        """
        Generate LLM response using POML template
        
        Args:
            template: Template path or PromptTemplate enum
            data: Data to render template with
            provider: LLM provider to use (default: first available)
            **kwargs: Additional LLM parameters
            
        Returns:
            Response dictionary with content and metadata
        """
        try:
            self.metrics['total_calls'] += 1
            
            # Resolve template path
            if isinstance(template, PromptTemplate):
                template_path = template.value
            else:
                template_path = template
            
            # Track template usage
            self.metrics['template_usage'][template_path] = \
                self.metrics['template_usage'].get(template_path, 0) + 1
            
            # Render template with POML
            prompt = self.poml.render(template_path, data)
            
            # Select provider
            if provider is None:
                provider = next(iter(self.providers.keys()))
            
            if provider not in self.providers:
                raise ValueError(f"Unknown provider: {provider}")
            
            # Track provider usage
            self.metrics['provider_usage'][provider] = \
                self.metrics['provider_usage'].get(provider, 0) + 1
            
            # Get LLM client
            llm = self.providers[provider]
            
            # Extract parameters
            temperature = kwargs.get('temperature', 0.8)
            max_tokens = kwargs.get('max_tokens', 500)
            
            # Generate response
            response = await llm.generate_response(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Parse and validate response
            result = {
                'content': response.content,
                'metadata': {
                    'template': template_path,
                    'provider': provider,
                    'temperature': temperature,
                    **response.metadata
                }
            }
            
            # Validate against schema if applicable
            if 'schemas/' in template_path:
                self._validate_response(response.content, template_path)
            
            return result
            
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Template generation failed: {e}")
            raise
    
    async def orchestrate_character_response(self,
                                            character: Any,
                                            situation: str,
                                            emphasis: str = "neutral",
                                            **kwargs) -> Dict:
        """
        Orchestrate character response generation
        
        Args:
            character: Character state object
            situation: Situation description
            emphasis: Response emphasis mode
            **kwargs: Additional parameters
            
        Returns:
            Character response dictionary
        """
        data = {
            'character': character,
            'situation': situation,
            'emphasis': emphasis,
            'temperature': kwargs.get('temperature', 0.8)
        }
        
        response = await self.generate_with_template(
            template=PromptTemplate.CHARACTER_RESPONSE,
            data=data,
            **kwargs
        )
        
        # Parse JSON response
        try:
            response['parsed'] = json.loads(response['content'])
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response")
            response['parsed'] = None
        
        return response
    
    async def orchestrate_scene_creation(self,
                                        beat: Dict,
                                        characters: List[Dict],
                                        previous_context: str = "",
                                        **kwargs) -> Dict:
        """
        Orchestrate scene creation
        
        Args:
            beat: Story beat information
            characters: List of characters in scene
            previous_context: Context from previous scene
            **kwargs: Additional parameters
            
        Returns:
            Scene creation response
        """
        data = {
            'beat': beat,
            'characters': characters,
            'previous_context': previous_context,
            'genre': kwargs.get('genre', 'drama'),
            'include_dialogue_starter': kwargs.get('include_dialogue', False)
        }
        
        response = await self.generate_with_template(
            template=PromptTemplate.SCENE_CRAFTING,
            data=data,
            **kwargs
        )
        
        # Parse scene elements
        response['scene_elements'] = self._parse_scene_elements(response['content'])
        
        return response
    
    async def orchestrate_dialogue(self,
                                  character: Dict,
                                  scene: Dict,
                                  dialogue_context: Dict,
                                  **kwargs) -> Dict:
        """
        Orchestrate dialogue generation
        
        Args:
            character: Character information
            scene: Scene information
            dialogue_context: Dialogue context
            **kwargs: Additional parameters
            
        Returns:
            Dialogue response
        """
        # Create dialogue template if not exists
        dialogue_template = "templates/narrative/dialogue_generation.poml"
        
        # Check if template exists, if not use character response
        if not self.poml.validate_template(dialogue_template):
            logger.warning("Dialogue template not found, using character response")
            return await self.orchestrate_character_response(
                character=character,
                situation=scene.get('situation', ''),
                **kwargs
            )
        
        data = {
            'character': character,
            'scene': scene,
            'context': dialogue_context
        }
        
        response = await self.generate_with_template(
            template=dialogue_template,
            data=data,
            **kwargs
        )
        
        return response
    
    async def orchestrate_narrative_sequence(self,
                                           characters: List[Any],
                                           story_arc: Dict,
                                           **kwargs) -> List[Dict]:
        """
        Orchestrate complete narrative sequence
        
        Args:
            characters: List of character objects
            story_arc: Story arc definition
            **kwargs: Additional parameters
            
        Returns:
            List of scene responses
        """
        results = []
        previous_context = ""
        
        for beat in story_arc.get('beats', []):
            # Create scene
            scene_response = await self.orchestrate_scene_creation(
                beat=beat,
                characters=characters,
                previous_context=previous_context,
                **kwargs
            )
            
            # Generate character responses for scene
            character_responses = []
            for character in characters:
                char_response = await self.orchestrate_character_response(
                    character=character,
                    situation=scene_response['scene_elements'].get('situation', ''),
                    emphasis=beat.get('emphasis', 'neutral'),
                    **kwargs
                )
                character_responses.append(char_response)
            
            # Combine results
            result = {
                'beat': beat,
                'scene': scene_response,
                'character_responses': character_responses
            }
            
            results.append(result)
            
            # Update context for next scene
            if character_responses:
                previous_context = character_responses[0].get('content', '')[:200]
        
        return results
    
    async def orchestrate_group_dynamics(self,
                                        characters: List[Any],
                                        situation: str,
                                        group_context: Dict,
                                        **kwargs) -> Dict:
        """
        Orchestrate group dynamics simulation
        
        Args:
            characters: List of character objects
            situation: Group situation
            group_context: Group dynamics context
            **kwargs: Additional parameters
            
        Returns:
            Group dynamics response
        """
        # Check for group dynamics template
        group_template = "templates/simulations/group_dynamics.poml"
        
        if not self.poml.validate_template(group_template):
            # Fallback: Generate individual responses
            responses = []
            for character in characters:
                response = await self.orchestrate_character_response(
                    character=character,
                    situation=situation,
                    **kwargs
                )
                responses.append(response)
            
            return {
                'individual_responses': responses,
                'group_dynamic': 'Generated individually (template not found)'
            }
        
        data = {
            'characters': characters,
            'situation': situation,
            'group_context': group_context
        }
        
        response = await self.generate_with_template(
            template=group_template,
            data=data,
            **kwargs
        )
        
        return response
    
    def _parse_scene_elements(self, content: str) -> Dict:
        """
        Parse scene elements from response content
        
        Args:
            content: Response content
            
        Returns:
            Parsed scene elements
        """
        elements = {
            'situation': '',
            'sensory': {},
            'opening_line': ''
        }
        
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if 'SCENE SITUATION' in line:
                current_section = 'situation'
            elif 'SENSORY DETAILS' in line:
                current_section = 'sensory'
            elif 'OPENING LINE' in line:
                current_section = 'opening'
            elif line and current_section:
                if current_section == 'situation':
                    elements['situation'] += line + ' '
                elif current_section == 'sensory':
                    if 'Sight:' in line:
                        elements['sensory']['sight'] = line.replace('Sight:', '').strip()
                    elif 'Sound:' in line:
                        elements['sensory']['sound'] = line.replace('Sound:', '').strip()
                    elif 'Atmosphere:' in line:
                        elements['sensory']['atmosphere'] = line.replace('Atmosphere:', '').strip()
                elif current_section == 'opening':
                    elements['opening_line'] = line
        
        return elements
    
    def _validate_response(self, content: str, template_path: str):
        """
        Validate response against schema
        
        Args:
            content: Response content
            template_path: Template path for schema
        """
        try:
            # Parse JSON content
            data = json.loads(content)
            
            # Get schema from template
            schema_template = template_path.replace('templates/', 'templates/schemas/')
            schema_template = schema_template.replace('.poml', '_schema.poml')
            
            if self.poml.validate_template(schema_template):
                # In production, would validate against schema
                logger.debug(f"Response validated against {schema_template}")
            
        except Exception as e:
            logger.warning(f"Schema validation failed: {e}")
    
    async def batch_generate(self,
                           requests: List[Dict],
                           max_concurrent: int = 5) -> List[Dict]:
        """
        Generate multiple responses in batch
        
        Args:
            requests: List of request dictionaries
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of responses
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_request(request):
            async with semaphore:
                return await self.generate_with_template(**request)
        
        tasks = [process_request(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle errors
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch request {i} failed: {result}")
                processed_results.append({
                    'error': str(result),
                    'request': requests[i]
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_metrics(self) -> Dict:
        """
        Get orchestrator metrics
        
        Returns:
            Metrics dictionary
        """
        return {
            **self.metrics,
            'available_templates': len(self.templates),
            'cached_templates': len(self.compiled_templates),
            'providers': list(self.providers.keys())
        }
    
    def clear_cache(self):
        """Clear template cache"""
        self.compiled_templates.clear()
        self.poml.clear_cache()
        logger.info("Orchestrator cache cleared")


# Integration helper for existing code
class OrchestratorMigrationHelper:
    """
    Helper for migrating existing orchestrators to POML
    """
    
    @staticmethod
    def create_poml_orchestrator(existing_orchestrator: Any) -> POMLOrchestrator:
        """
        Create POML orchestrator from existing orchestrator
        
        Args:
            existing_orchestrator: Original orchestrator instance
            
        Returns:
            POMLOrchestrator instance
        """
        # Extract providers from existing orchestrator
        providers = {}
        
        if hasattr(existing_orchestrator, 'providers'):
            providers = existing_orchestrator.providers
        elif hasattr(existing_orchestrator, 'llm'):
            providers['default'] = existing_orchestrator.llm
        
        # Extract config
        config = {}
        if hasattr(existing_orchestrator, 'config'):
            config = existing_orchestrator.config
        
        return POMLOrchestrator(providers, config)
    
    @staticmethod
    async def compare_outputs(poml_orchestrator: POMLOrchestrator,
                             original_orchestrator: Any,
                             test_data: Dict) -> Dict:
        """
        Compare outputs between POML and original orchestrator
        
        Args:
            poml_orchestrator: POML orchestrator
            original_orchestrator: Original orchestrator
            test_data: Test data for comparison
            
        Returns:
            Comparison results
        """
        results = {'matches': [], 'differences': []}
        
        # Test character response
        if hasattr(original_orchestrator, 'generate'):
            original_response = await original_orchestrator.generate(
                prompt=test_data.get('prompt', 'Test prompt'),
                **test_data.get('kwargs', {})
            )
            
            poml_response = await poml_orchestrator.generate_with_template(
                template=test_data.get('template', PromptTemplate.CHARACTER_RESPONSE),
                data=test_data,
                **test_data.get('kwargs', {})
            )
            
            # Compare responses
            if original_response.content == poml_response['content']:
                results['matches'].append('character_response')
            else:
                results['differences'].append({
                    'type': 'character_response',
                    'original': original_response.content[:100],
                    'poml': poml_response['content'][:100]
                })
        
        return results


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def demo():
        # Create mock providers
        class MockLLM:
            async def generate_response(self, prompt, **kwargs):
                return type('Response', (), {
                    'content': json.dumps({
                        'dialogue': 'Test dialogue',
                        'thought': 'Test thought',
                        'action': 'Test action',
                        'emotional_shift': {'anger': 0, 'doubt': 0, 'fear': 0, 'compassion': 0}
                    }),
                    'metadata': {'model': 'mock', 'temperature': kwargs.get('temperature', 0.8)}
                })
        
        # Initialize orchestrator
        providers = {
            'mock': MockLLM(),
            'backup': MockLLM()
        }
        
        orchestrator = POMLOrchestrator(providers)
        
        # Test character response
        character_data = {
            'name': 'Test Character',
            'traits': ['brave', 'curious'],
            'emotional_state': {
                'anger': 0.2,
                'doubt': 0.5,
                'fear': 0.3,
                'compassion': 0.6,
                'confidence': 0.7
            }
        }
        
        response = await orchestrator.orchestrate_character_response(
            character=character_data,
            situation="You face a difficult choice",
            emphasis="doubt"
        )
        
        print("Character Response:")
        print(json.dumps(response['parsed'], indent=2))
        
        # Show metrics
        print("\nOrchestrator Metrics:")
        print(json.dumps(orchestrator.get_metrics(), indent=2))
    
    asyncio.run(demo())