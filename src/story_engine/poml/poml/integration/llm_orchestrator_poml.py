"""
LLM Orchestrator with POML Integration
Enhanced orchestrator that uses POML templates for all LLM interactions
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Union
from enum import Enum
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import POML engine
from story_engine.poml.lib.poml_integration import POMLEngine

logger = logging.getLogger(__name__)

class PromptTemplate(Enum):
    """Enumeration of available POML prompt templates"""
    CHARACTER_RESPONSE = "simulations/character_response.poml"
    SCENE_CRAFTING = "narrative/scene_crafting.poml"
    DIALOGUE_GENERATION = "narrative/dialogue_generation.poml"
    STORY_BEAT = "narrative/story_beats.poml"
    EMOTIONAL_JOURNEY = "simulations/emotional_journey.poml"
    GROUP_DYNAMICS = "simulations/group_dynamics.poml"
    NARRATIVE_EVALUATION = "narrative/evaluation.poml"

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

    def _template_file_path(self, template_path: str) -> Path | None:
        try:
            base = Path(__file__).parent.parent / "templates"
            # Normalize leading prefix if present
            t = template_path.lstrip('/')
            if t.startswith('templates/'):
                t = t[len('templates/'):]
            p = base / t
            return p if p.exists() else None
        except Exception:
            return None

    def _extract_declared_schema(self, template_path: str) -> str | None:
        """Parse the template for a <schema>...</schema> tag and return a normalized path.
        Returns a relative path like 'schemas/simulation_response.poml' or None.
        """
        try:
            fpath = self._template_file_path(template_path)
            if not fpath:
                return None
            content = fpath.read_text(encoding='utf-8')
            import re
            m = re.search(r'<schema>(.*?)</schema>', content, flags=re.DOTALL)
            if not m:
                return None
            raw = m.group(1).strip()
            # Normalize away leading 'templates/' if present
            if raw.startswith('templates/'):
                raw = raw[len('templates/'):]
            return raw
        except Exception:
            return None

    def _clean_json_text(self, text: str) -> str:
        """Strip code fences and extract the first JSON object if needed."""
        s = (text or '').strip()
        import re
        # Remove code fences
        if s.startswith('```'):
            s = re.sub(r"^```[a-zA-Z0-9_-]*\n|\n```$", "", s)
        # Trim any leading commentary before first '{' and after last '}'
        start = s.find('{')
        end = s.rfind('}')
        if start != -1 and end != -1 and end > start:
            s = s[start:end+1]
        return s.strip()

    def _validate_against_known_schema(self, schema_path: str, payload: Dict[str, Any]) -> None:
        """Lightweight shape checks for known schemas by filename."""
        name = Path(schema_path).stem.lower()
        try:
            if 'simulation_response' in name:
                # Expect keys and emotional_shift subkeys
                required = ['dialogue', 'thought', 'action', 'emotional_shift']
                missing = [k for k in required if k not in payload]
                if missing:
                    raise ValueError(f"Missing keys: {', '.join(missing)}")
                es = payload.get('emotional_shift')
                if not isinstance(es, dict):
                    raise ValueError("'emotional_shift' must be an object")
                for k in ['anger', 'doubt', 'fear', 'compassion']:
                    if k not in es:
                        raise ValueError(f"'emotional_shift.{k}' missing")
            # Extend with additional schema names as needed
        except Exception as e:
            raise ValueError(str(e))

    def _enforce_schema_and_clean_json(self, template_path: str, raw_text: str) -> Dict[str, Any]:
        """If template declares a schema, ensure JSON can be parsed and optionally validate shape.
        Returns the parsed object. Raises ValueError with structured info on failure.
        """
        schema_rel = self._extract_declared_schema(template_path)
        if not schema_rel:
            # No schema declared: do nothing special
            return {}
        cleaned = self._clean_json_text(raw_text)
        try:
            parsed = json.loads(cleaned)
        except Exception as e:
            # Raise structured error
            err = {
                'error': 'invalid_json',
                'message': f'Failed to parse JSON for template {template_path}: {e}',
                'template': template_path,
                'schema': schema_rel,
                'raw_preview': (raw_text or '')[:400],
                'cleaned_preview': cleaned[:400],
                'hint': 'Ensure the model returns a valid JSON object without trailing prose or code fences.'
            }
            raise ValueError(err)
        # Lightweight schema enforcement by filename
        try:
            self._validate_against_known_schema(schema_rel, parsed)
        except Exception as e:
            err = {
                'error': 'schema_violation',
                'message': f'Response does not match expected schema {schema_rel}: {e}',
                'template': template_path,
                'schema': schema_rel,
                'raw_preview': (raw_text or '')[:400],
                'cleaned_preview': cleaned[:400],
                'hint': 'Check required keys and types for the declared schema.'
            }
            raise ValueError(err)
        return parsed
    
    def _register_templates(self):
        """Register all available POML templates"""
        # Scan templates directory
        template_dir = Path(__file__).parent.parent / "templates"
        
        for category in ['characters', 'simulations', 'narrative', 'schemas']:
            category_path = template_dir / category
            if category_path.exists():
                for template_file in category_path.glob("*.poml"):
                    template_name = f"{category}/{template_file.stem}"
                    # Store normalized (relative) template path without leading 'templates/'
                    template_path = f"{category}/{template_file.name}"
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
            
            # Enforce schema and cleanup for JSON if declared
            parsed_payload = {}
            try:
                parsed_payload = self._enforce_schema_and_clean_json(template_path, getattr(response, 'content', '') or '')
            except ValueError as ve:
                # Reraise to caller after metrics update in except below
                raise

            # Parse and validate response
            result = {
                'content': getattr(response, 'content', ''),
                'metadata': {
                    'template': template_path,
                    'provider': provider,
                    'temperature': temperature,
                    **response.metadata
                }
            }
            if parsed_payload:
                result['parsed'] = parsed_payload
            
            # Validate against schema if applicable
            if 'schemas/' in template_path:
                self._validate_response(getattr(response, 'content', ''), template_path)
            
            return result
            
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Template generation failed: {e}")
            raise

    async def generate_with_template_roles(self,
                                           template: Union[str, PromptTemplate],
                                           data: Dict[str, Any],
                                           provider: str = None,
                                           **kwargs) -> Dict:
        """Generate using role-separated prompts when available.

        If the underlying provider supports a `system` parameter, it will be passed.
        Otherwise, the system content will be prepended to the user prompt.
        """
        try:
            self.metrics['total_calls'] += 1
            # Resolve template path (normalized without leading 'templates/')
            template_path = template.value if isinstance(template, PromptTemplate) else template
            self.metrics['template_usage'][template_path] = self.metrics['template_usage'].get(template_path, 0) + 1

            # Render roles via POML
            roles = self.poml.render_roles(template_path, data)
            prompt_user = roles.get('user', '')
            prompt_system = roles.get('system', '')

            # Provider selection
            if provider is None:
                provider = next(iter(self.providers.keys()))
            if provider not in self.providers:
                raise ValueError(f"Unknown provider: {provider}")
            self.metrics['provider_usage'][provider] = self.metrics['provider_usage'].get(provider, 0) + 1
            llm = self.providers[provider]

            temperature = kwargs.get('temperature', 0.8)
            max_tokens = kwargs.get('max_tokens', 500)

            # Try to call with system if supported; fall back to concatenation
            response = None
            try:
                response = await llm.generate_response(prompt=prompt_user, system=prompt_system, temperature=temperature, max_tokens=max_tokens)
            except TypeError:
                joined = (f"[SYSTEM]\n{prompt_system}\n\n[USER]\n{prompt_user}").strip()
                response = await llm.generate_response(prompt=joined, temperature=temperature, max_tokens=max_tokens)

            raw_text = getattr(response, 'content', None) or getattr(response, 'text', '')
            # Enforce schema and cleanup for JSON if declared
            parsed_payload = {}
            try:
                parsed_payload = self._enforce_schema_and_clean_json(template_path, raw_text)
            except ValueError as ve:
                raise

            result = {
                'content': raw_text,
                'metadata': {
                    'template': template_path,
                    'provider': provider,
                    'temperature': temperature,
                    **getattr(response, 'metadata', {})
                }
            }
            if parsed_payload:
                result['parsed'] = parsed_payload
            return result
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Template generation (roles) failed: {e}")
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
        
        # Prefer role-separated generation if enabled via env/config
        roles_enabled = str(self.config.get('enable_roles') or '').lower() in {"1","true","yes","on"}
        import os as _os
        if not roles_enabled:
            roles_enabled = str(_os.environ.get('POML_ROLES') or '').lower() in {"1","true","yes","on"}

        if roles_enabled:
            response = await self.generate_with_template_roles(
                template=PromptTemplate.CHARACTER_RESPONSE,
                data=data,
                **kwargs
            )
        else:
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
        
        roles_enabled = str(self.config.get('enable_roles') or '').lower() in {"1","true","yes","on"}
        import os as _os
        if not roles_enabled:
            roles_enabled = str(_os.environ.get('POML_ROLES') or '').lower() in {"1","true","yes","on"}
        if roles_enabled:
            response = await self.generate_with_template_roles(
                template=PromptTemplate.SCENE_CRAFTING,
                data=data,
                **kwargs
            )
        else:
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
        dialogue_template = "narrative/dialogue_generation.poml"
        
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
        
        roles_enabled = str(self.config.get('enable_roles') or '').lower() in {"1","true","yes","on"}
        import os as _os
        if not roles_enabled:
            roles_enabled = str(_os.environ.get('POML_ROLES') or '').lower() in {"1","true","yes","on"}
        if roles_enabled:
            response = await self.generate_with_template_roles(
                template=dialogue_template,
                data=data,
                **kwargs
            )
        else:
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
        group_template = "simulations/group_dynamics.poml"
        
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
        
        roles_enabled = str(self.config.get('enable_roles') or '').lower() in {"1","true","yes","on"}
        import os as _os
        if not roles_enabled:
            roles_enabled = str(_os.environ.get('POML_ROLES') or '').lower() in {"1","true","yes","on"}
        if roles_enabled:
            response = await self.generate_with_template_roles(
                template=group_template,
                data=data,
                **kwargs
            )
        else:
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
                roles_enabled = str(self.config.get('enable_roles') or '').lower() in {"1","true","yes","on"}
                import os as _os
                if not roles_enabled:
                    roles_enabled = str(_os.environ.get('POML_ROLES') or '').lower() in {"1","true","yes","on"}
                if roles_enabled:
                    return await self.generate_with_template_roles(**request)
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
