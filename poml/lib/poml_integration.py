"""
POML Integration Layer for Story Engine
Provides Python interface to POML template system for prompt management
"""

import os
import json
import yaml
import asyncio
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class POMLConfig:
    """Configuration for POML engine"""
    template_paths: List[str] = None
    cache_enabled: bool = True
    cache_ttl: int = 3600
    debug: bool = False
    strict_mode: bool = False
    
    def __post_init__(self):
        if self.template_paths is None:
            self.template_paths = ['templates/', 'components/', 'gallery/']

class POMLCache:
    """Simple in-memory cache for rendered templates"""
    
    def __init__(self, ttl_seconds: int = 3600):
        self.cache = {}
        self.ttl = timedelta(seconds=ttl_seconds)
        
    def _get_key(self, template_path: str, data: Dict) -> str:
        """Generate cache key from template and data"""
        data_str = json.dumps(data, sort_keys=True, default=str)
        combined = f"{template_path}:{data_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, template_path: str, data: Dict) -> Optional[str]:
        """Get cached template if available and not expired"""
        key = self._get_key(template_path, data)
        
        if key in self.cache:
            cached_item = self.cache[key]
            if datetime.now() - cached_item['timestamp'] < self.ttl:
                logger.debug(f"Cache hit for template: {template_path}")
                return cached_item['content']
            else:
                # Remove expired item
                del self.cache[key]
                
        return None
    
    def set(self, template_path: str, data: Dict, content: str):
        """Cache rendered template"""
        key = self._get_key(template_path, data)
        self.cache[key] = {
            'content': content,
            'timestamp': datetime.now()
        }
        logger.debug(f"Cached template: {template_path}")
    
    def clear(self):
        """Clear all cached items"""
        self.cache.clear()

class POMLEngine:
    """
    Main POML engine for template rendering
    Integrates with Node.js POML processor via subprocess or REST API
    """
    
    def __init__(self, config: Optional[POMLConfig] = None, config_file: Optional[str] = None):
        """
        Initialize POML engine
        
        Args:
            config: POMLConfig object
            config_file: Path to YAML config file
        """
        self.config = config or POMLConfig()
        
        # Load config from file if provided
        if config_file:
            self._load_config(config_file)
            
        # Initialize cache if enabled
        self.cache = POMLCache(self.config.cache_ttl) if self.config.cache_enabled else None
        
        # Set up template search paths
        self.template_dirs = self._setup_template_paths()
        
        # Track loaded templates for hot reload
        self.loaded_templates = {}
        
        logger.info(f"POML Engine initialized with template paths: {self.template_dirs}")
    
    def _load_config(self, config_file: str):
        """Load configuration from YAML file"""
        config_path = Path(config_file)
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                
            # Update config with loaded values
            if 'cache' in config_data:
                self.config.cache_enabled = config_data['cache'].get('enabled', True)
                self.config.cache_ttl = config_data['cache'].get('ttl_seconds', 3600)
            
            if 'rendering' in config_data:
                self.config.debug = config_data['rendering'].get('debug', False)
                self.config.strict_mode = config_data['rendering'].get('strict_mode', False)
            
            if 'template_paths' in config_data:
                self.config.template_paths = config_data['template_paths']
                
            logger.info(f"Loaded configuration from {config_file}")
    
    def _setup_template_paths(self) -> List[Path]:
        """Set up and validate template search paths"""
        base_path = Path(__file__).parent.parent  # Go up to poml/ directory
        paths = []
        
        for template_dir in self.config.template_paths:
            full_path = base_path / template_dir
            if full_path.exists():
                paths.append(full_path)
            else:
                logger.warning(f"Template path does not exist: {full_path}")
                
        return paths
    
    def _find_template(self, template_name: str) -> Optional[Path]:
        """Find template file in search paths"""
        # If absolute path, use directly
        if os.path.isabs(template_name):
            template_path = Path(template_name)
            if template_path.exists():
                return template_path
                
        # Search in template directories
        for template_dir in self.template_dirs:
            template_path = template_dir / template_name
            if template_path.exists():
                return template_path
                
        logger.error(f"Template not found: {template_name}")
        return None
    
    def _preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess data for template rendering"""
        processed = {}
        
        for key, value in data.items():
            # Convert dataclasses to dictionaries
            if hasattr(value, '__dataclass_fields__'):
                processed[key] = asdict(value)
            # Handle special types
            elif isinstance(value, datetime):
                processed[key] = value.isoformat()
            elif hasattr(value, '__dict__'):
                # Convert objects to dictionaries
                processed[key] = value.__dict__
            else:
                processed[key] = value
                
        return processed
    
    def render(self, template_name: str, data: Dict[str, Any], 
               use_cache: bool = True) -> str:
        """
        Render a POML template with provided data
        
        Args:
            template_name: Path to template file (relative or absolute)
            data: Data context for template rendering
            use_cache: Whether to use cache for this render
            
        Returns:
            Rendered template as string
        """
        # Check cache first
        if use_cache and self.cache:
            cached = self.cache.get(template_name, data)
            if cached:
                return cached
        
        # Find template file
        template_path = self._find_template(template_name)
        if not template_path:
            raise FileNotFoundError(f"Template not found: {template_name}")
        
        # Preprocess data
        processed_data = self._preprocess_data(data)
        
        # For now, use native Python rendering
        # In production, this would call Node.js POML processor
        rendered = self._render_native(template_path, processed_data)
        
        # Cache result
        if use_cache and self.cache:
            self.cache.set(template_name, data, rendered)
        
        return rendered

    def render_roles(self, template_name: str, data: Dict[str, Any]) -> Dict[str, str]:
        """Render template and split into chat roles: system and user.
        - Extracts <system>...</system> content as system message (after substitution)
        - Renders the rest as user message
        """
        # Find template file
        template_path = self._find_template(template_name)
        if not template_path:
            raise FileNotFoundError(f"Template not found: {template_name}")
        processed_data = self._preprocess_data(data)
        with open(template_path, 'r', encoding='utf-8') as f:
            content = f.read()

        import re
        # Extract system block
        m = re.search(r'<system>(.*?)</system>', content, flags=re.DOTALL)
        sys_block = m.group(1) if m else ''
        # Remove system block from user content
        user_content = re.sub(r'<system>.*?</system>', '', content, flags=re.DOTALL)

        # Simple variable replacement for both
        def _subst(text: str) -> str:
            def replace_var(match):
                var_path = match.group(1).strip()
                parts = var_path.split('.')
                value = processed_data
                for part in parts:
                    if '|' in part:
                        part = part.split('|')[0].strip()
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        return match.group(0)
                return str(value)
            text = re.sub(r'\{\{([^}]+)\}\}', replace_var, text)
            # Strip tags
            text = re.sub(r'<metadata>.*?</metadata>', '', text, flags=re.DOTALL)
            text = re.sub(r'<style>.*?</style>', '', text, flags=re.DOTALL)
            text = re.sub(r'<import[^>]*>', '', text)
            text = re.sub(r'<[^>]+>', '', text)
            text = re.sub(r'\n\s*\n', '\n\n', text)
            return text.strip()

        return {"system": _subst(sys_block), "user": _subst(user_content)}
    
    def _render_native(self, template_path: Path, data: Dict[str, Any]) -> str:
        """
        Native Python rendering (simplified)
        In production, this would call the Node.js POML processor
        """
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
        
        # Simple variable substitution for demonstration
        # Real implementation would use full POML processor
        result = template_content
        
        # Basic variable replacement
        import re
        
        def replace_var(match):
            var_path = match.group(1).strip()
            
            # Handle dot notation
            parts = var_path.split('.')
            value = data
            
            for part in parts:
                # Handle filters (simplified)
                if '|' in part:
                    part = part.split('|')[0].strip()
                
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    # Default value handling
                    if 'default:' in var_path:
                        default_val = var_path.split('default:')[1].split('}')[0].strip().strip('"\'')
                        return default_val
                    return f"{{{{ {var_path} }}}}"  # Return unchanged if not found
            
            return str(value)
        
        # Replace variables
        result = re.sub(r'\{\{([^}]+)\}\}', replace_var, result)
        
        # Remove POML-specific tags for text output
        result = re.sub(r'<document[^>]*>', '', result)
        result = re.sub(r'</document>', '', result)
        result = re.sub(r'<metadata>.*?</metadata>', '', result, flags=re.DOTALL)
        result = re.sub(r'<style>.*?</style>', '', result, flags=re.DOTALL)
        result = re.sub(r'<import[^>]*>', '', result)
        
        # Basic if statement processing
        def process_if(match):
            condition = match.group(1)
            content = match.group(2)
            
            # Very simplified condition evaluation
            # Real implementation would properly evaluate expressions
            if 'true' in condition.lower() or '>' in condition:
                return content
            return ''
        
        result = re.sub(r'<if test="([^"]+)">(.*?)</if>', process_if, result, flags=re.DOTALL)
        
        # Clean up remaining XML tags (simplified)
        result = re.sub(r'<[^>]+>', '', result)
        
        # Clean up extra whitespace
        result = re.sub(r'\n\s*\n', '\n\n', result)
        
        return result.strip()
    
    async def render_async(self, template_name: str, data: Dict[str, Any]) -> str:
        """Async version of render for compatibility"""
        return self.render(template_name, data)
    
    def render_batch(self, renders: List[Dict[str, Any]]) -> List[str]:
        """
        Render multiple templates in batch
        
        Args:
            renders: List of dicts with 'template' and 'data' keys
            
        Returns:
            List of rendered templates
        """
        results = []
        
        for render_spec in renders:
            template = render_spec.get('template')
            data = render_spec.get('data', {})
            
            try:
                result = self.render(template, data)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to render {template}: {e}")
                results.append(f"Error rendering {template}: {e}")
                
        return results
    
    def validate_template(self, template_name: str) -> bool:
        """Validate that a template exists and is syntactically correct"""
        template_path = self._find_template(template_name)
        
        if not template_path:
            return False
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Basic validation - check for XML well-formedness
            # Real implementation would use POML parser
            if content.count('<') != content.count('>'):
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Template validation failed: {e}")
            return False
    
    def list_templates(self, pattern: str = "**/*.poml") -> List[str]:
        """List all available templates"""
        templates = []
        
        for template_dir in self.template_dirs:
            for template_path in template_dir.glob(pattern):
                relative_path = template_path.relative_to(template_dir)
                templates.append(str(relative_path))
                
        return sorted(set(templates))
    
    def clear_cache(self):
        """Clear template cache"""
        if self.cache:
            self.cache.clear()
            logger.info("Template cache cleared")

# Convenience functions
def create_engine(config_file: str = "config/poml_config.yaml") -> POMLEngine:
    """Create POML engine with default configuration"""
    # Look for config file relative to poml directory
    base_path = Path(__file__).parent.parent
    config_path = base_path / config_file
    
    if config_path.exists():
        return POMLEngine(config_file=str(config_path))
    else:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return POMLEngine()

def render_template(template: str, data: Dict[str, Any]) -> str:
    """Quick render without persistent engine"""
    engine = create_engine()
    return engine.render(template, data)

# Integration with existing Story Engine
class StoryEnginePOMLAdapter:
    """Adapter for integrating POML with existing Story Engine code"""
    
    def __init__(self, engine: Optional[POMLEngine] = None, runtime_flags: Optional[Dict[str, Dict[str, Any]]] = None):
        self.engine = engine or create_engine()
        # Character persona cache
        self._persona_cache: Dict[str, Dict[str, Any]] = {}
        # Context cache
        self._context_cache: Dict[str, str] = {}
        # Per-character runtime flags (e.g., {"pontius_pilate": {"era_mode": "mark_i"}})
        self._runtime_flags: Dict[str, Dict[str, Any]] = dict(runtime_flags or {})

    def _load_persona(self, character: Dict[str, Any]) -> Dict[str, Any]:
        """Load optional character persona YAML and merge onto character data."""
        try:
            name = (character.get('id') or character.get('name') or '').lower().replace(' ', '_')
            if not name:
                return character
            if name in self._persona_cache:
                overlay = self._persona_cache[name]
            else:
                base_path = Path(__file__).parent.parent / 'config' / 'characters' / f'{name}.yaml'
                overlay = {}
                if base_path.exists():
                    with open(base_path, 'r', encoding='utf-8') as f:
                        overlay = yaml.safe_load(f) or {}
                self._persona_cache[name] = overlay

            if not overlay:
                return character

            # Shallow merge lists for constraints/traits/values/fears/desires
            merged = dict(character)
            for key in ['constraints', 'traits', 'values', 'fears', 'desires']:
                if overlay.get(key):
                    merged[key] = list({*list(merged.get(key, [])), *list(overlay.get(key, []))})
            # Merge backstory/memory sub-objects if provided
            for key in ['backstory', 'memory']:
                if isinstance(overlay.get(key), dict):
                    base = dict(merged.get(key) or {})
                    base.update(overlay[key])
                    merged[key] = base
            # Add style or voice hints
            for key in ['style', 'voice']:
                if overlay.get(key):
                    merged[key] = overlay[key]
            return merged
        except Exception:
            return character
    
    def get_character_prompt(self, character, situation: str, emphasis: str = "neutral") -> str:
        """
        Replace character_simulation_engine_v2.get_simulation_prompt
        """
        # Merge persona overlay if present
        if isinstance(character, dict):
            character = self._load_persona(character)
        # Apply runtime flags overlay
        character = self._apply_runtime_flags(character)
        # Load optional world/character context briefs
        context_text = self._load_context(character)
        return self.engine.render(
            'simulations/character_response.poml',
            {
                'character': character,
                'situation': situation,
                'context': context_text,
                'emphasis': emphasis,
                'temperature': 0.8,
                'flags': self._current_flags(character),
            }
        )

    def get_character_prompt_roles(self, character, situation: str, emphasis: str = "neutral", world_pov: str = "") -> Dict[str, str]:
        if isinstance(character, dict):
            character = self._load_persona(character)
        # Apply runtime flags overlay
        character = self._apply_runtime_flags(character)
        context_text = self._load_context(character)
        return self.engine.render_roles(
            'simulations/character_response.poml',
            {
                'character': character,
                'situation': situation,
                'context': context_text,
                'world_pov': world_pov,
                'emphasis': emphasis,
                'temperature': 0.8,
                'flags': self._current_flags(character),
            }
        )

    def _current_flags(self, character: Dict[str, Any]) -> Dict[str, Any]:
        try:
            cid = (character.get('id') or character.get('name') or '').lower().replace(' ', '_')
            return dict(self._runtime_flags.get(cid, {}))
        except Exception:
            return {}

    def _apply_runtime_flags(self, character: Dict[str, Any]) -> Dict[str, Any]:
        try:
            cid = (character.get('id') or character.get('name') or '').lower().replace(' ', '_')
            flags = self._runtime_flags.get(cid) or {}
            if not isinstance(flags, dict) or not flags:
                return character
            merged = dict(character)
            for k, v in flags.items():
                merged[k] = v
            return merged
        except Exception:
            return character

    def _load_context(self, character: Dict[str, Any]) -> str:
        """Load global + character-specific context briefs (markdown)."""
        try:
            name = (character.get('id') or character.get('name') or '').lower().replace(' ', '_')
            cache_key = name or 'global'
            if cache_key in self._context_cache:
                return self._context_cache[cache_key]

            base_dir = Path(__file__).parent.parent / 'config' / 'context'
            parts: list[str] = []
            # Global/world context
            g = base_dir / 'global.md'
            if g.exists():
                parts.append(g.read_text(encoding='utf-8'))
            # Character-specific context
            if name:
                c = base_dir / f'{name}.md'
                if c.exists():
                    parts.append(c.read_text(encoding='utf-8'))

            text = '\n\n'.join(p.strip() for p in parts if p.strip())
            self._context_cache[cache_key] = text
            return text
        except Exception:
            return ''
    
    def get_scene_prompt(self, beat: Dict, characters: List[Dict], 
                        previous_context: str = "") -> str:
        """
        Replace narrative_pipeline.craft_scene prompt generation
        """
        return self.engine.render(
            'narrative/scene_crafting.poml',
            {
                'beat': beat,
                'characters': characters,
                'previous_context': previous_context
            }
        )
    
    def get_dialogue_prompt(self, character: Dict, scene: Dict, 
                           dialogue_context: Dict) -> str:
        """
        Generate dialogue for a character in a scene
        """
        return self.engine.render(
            'narrative/dialogue_generation.poml',
            {
                'character': character,
                'scene': scene,
                'context': dialogue_context
            }
        )

    def get_plot_structure_prompt(self, request: Dict[str, Any]) -> str:
        """Generate plot structure request prompt"""
        # Accept either dataclass or dict
        data = request
        if hasattr(request, "__dict__") or hasattr(request, "__dataclass_fields__"):
            try:
                from dataclasses import asdict
                data = asdict(request)
            except Exception:
                data = request.__dict__

        return self.engine.render(
            'narrative/plot_structure.poml',
            {
                'title': data.get('title', ''),
                'premise': data.get('premise', ''),
                'genre': data.get('genre', ''),
                'tone': data.get('tone', ''),
                'setting': data.get('setting', ''),
                'structure': data.get('structure', 'three_act'),
            }
        )

    def get_quality_evaluation_prompt(self, story_content: str, metrics: List[str]) -> str:
        """Generate evaluation prompt for story content"""
        return self.engine.render(
            'narrative/quality_evaluation.poml',
            {
                'story': story_content,
                'metrics': metrics,
            }
        )

    def get_enhancement_prompt(self, content: str, evaluation_text: str, focus: str = "general", metrics: Optional[Dict[str, Any]] = None) -> str:
        """Generate enhancement prompt based on evaluation and focus.
        Optionally include structured metrics as JSON for tighter control.
        """
        import json as _json
        return self.engine.render(
            'narrative/enhancement.poml',
            {
                'content': content,
                'evaluation': evaluation_text or 'No evaluation',
                'focus': focus or 'general',
                'metrics_json': _json.dumps(metrics) if metrics else '',
            }
        )

    def get_world_state_brief(self, world_state: Dict[str, Any]) -> str:
        return self.engine.render(
            'meta/world_state_brief.poml',
            {
                'world': world_state
            }
        )

    def get_world_state_brief_for(self, world_state: Dict[str, Any], characters: Optional[List[str]] = None, location: Optional[str] = None, last_n_events: int = 5) -> str:
        """Render a targeted world brief focusing on characters and/or a location."""
        # Lightweight filter mirroring WorldStateManager.targeted_subset behavior
        chars = set([c.lower() for c in (characters or [])])
        ws = {
            'facts': dict((world_state.get('facts') or {})),
            'relationships': {},
            'timeline': list((world_state.get('timeline') or []))[-last_n_events:],
            'availability': {},
            'locations': {},
            'props': dict((world_state.get('props') or {})),
        }
        rels = world_state.get('relationships') or {}
        if chars:
            for k, v in rels.items():
                try:
                    src, dst = k.split('->', 1)
                except ValueError:
                    continue
                if src.lower() in chars or dst.lower() in chars:
                    ws['relationships'][k] = v
        else:
            ws['relationships'] = rels
        av = world_state.get('availability') or {}
        ws['availability'] = {k: v for k, v in av.items() if not chars or k.lower() in chars}
        locs = world_state.get('locations') or {}
        if location and location in locs:
            ws['locations'][location] = locs[location]
        else:
            ws['locations'] = locs
        return self.get_world_state_brief(ws)

    def get_world_state_pov_brief(self, character: Dict[str, Any], world_subset: Dict[str, Any]) -> str:
        return self.engine.render(
            'meta/world_state_pov.poml',
            {
                'character': character,
                'world_subset': world_subset,
            }
        )

    # --- Meta narrative helpers ---
    def get_review_throughlines_prompt(
        self,
        character: Dict[str, Any],
        situations: List[str],
        simulations: List[Dict[str, Any]],
        target_criteria: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> str:
        import json as _json
        character = self._load_persona(character)
        return self.engine.render(
            'meta/reviewer_throughline.poml',
            {
                'character': character,
                'situations': situations,
                'simulations_json': _json.dumps(simulations, ensure_ascii=False),
                'target_criteria': target_criteria or [],
                'weights_json': _json.dumps(weights or {}, ensure_ascii=False),
            }
        )

    def get_throughline_synthesis_prompt(self, character: Dict[str, Any], throughline: Dict[str, Any]) -> str:
        import json as _json
        character = self._load_persona(character)
        return self.engine.render(
            'meta/throughline_synthesis.poml',
            {
                'character': character,
                'throughline': throughline,
                'throughline_evidence_json': _json.dumps(throughline.get('evidence', []), ensure_ascii=False)
            }
        )

    def get_screenplay_draft_prompt(self, meta_outline: str, style: str = "HBO Rome", focus: str = "pilot sequence") -> str:
        return self.engine.render(
            'narrative/screenplay_draft.poml',
            {
                'meta_outline': meta_outline,
                'style': style,
                'focus': focus,
            }
        )

    def get_persona_check_prompt(self, character: Dict[str, Any], response_json: Dict[str, Any]) -> str:
        """Generate a persona adherence check prompt.
        Accepts a character dict and the response payload (dict)."""
        import json as _json
        character = self._load_persona(character)
        return self.engine.render(
            'meta/persona_check.poml',
            {
                'character': character,
                'response_json': _json.dumps(response_json, ensure_ascii=False)
            }
        )

    def get_beat_extraction_prompt(self, sim: Dict[str, Any]) -> str:
        """Prompt to extract a beat atom from a single simulation result dict."""
        import json as _json
        character = sim.get('character') or {}
        # handle CharacterState dataclass or dict
        if hasattr(character, '__dataclass_fields__'):
            from dataclasses import asdict as _asdict
            character = _asdict(character)
        payload = {
            'character': character or {'name': sim.get('character_id', 'Character'), 'id': sim.get('character_id', 'char')},
            'character_id': sim.get('character_id', ''),
            'situation': sim.get('situation', ''),
            'emphasis': sim.get('emphasis', ''),
            'response_json': _json.dumps(sim.get('response') or {}, ensure_ascii=False),
        }
        return self.engine.render('meta/beat_extraction.poml', payload)

    def get_scene_plan_prompt(self, beats: list[dict], objective: str = '', style: str = '', continuity_fix: str = '') -> str:
        import json as _json
        return self.engine.render(
            'narrative/scene_plan.poml',
            {
                'beats_json': _json.dumps(beats, ensure_ascii=False),
                'objective': objective,
                'style': style,
                'continuity_fix': continuity_fix,
            }
        )

    def get_continuity_check_prompt(self, plan: Dict[str, Any], world_state: Dict[str, Any]) -> str:
        import json as _json
        return self.engine.render(
            'meta/continuity_check.poml',
            {
                'plan_json': _json.dumps(plan, ensure_ascii=False),
                'world_json': _json.dumps(world_state, ensure_ascii=False),
            }
        )

    def get_scenario_prompt(self, world_brief_markdown: str) -> str:
        return self.engine.render(
            'meta/scenario_crafting.poml',
            {
                'world_brief': world_brief_markdown,
            }
        )

    def get_plausibility_check_prompt(self, simulation: Dict[str, Any], world_state: Dict[str, Any]) -> str:
        import json as _json
        return self.engine.render(
            'meta/plausibility_check.poml',
            {
                'simulation_json': _json.dumps(simulation, ensure_ascii=False),
                'world_json': _json.dumps(world_state, ensure_ascii=False),
            }
        )

    def get_persona_iterative_review_prompt(
        self,
        character: Dict[str, Any],
        current_response: Dict[str, Any],
        previous_responses: List[Dict[str, Any]],
        threshold: int = 80,
    ) -> str:
        import json as _json
        character = self._load_persona(character)
        return self.engine.render(
            'meta/persona_iterative_review.poml',
            {
                'character': character,
                'current_response_json': _json.dumps(current_response, ensure_ascii=False),
                'previous_responses_json': _json.dumps(previous_responses[-3:], ensure_ascii=False),
                'threshold': threshold,
            }
        )

    def get_character_prompt_iterative(
        self,
        character,
        situation: str,
        emphasis: str = "neutral",
        previous_responses: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Generate character prompt that includes up to the last 3 responses to steer improvements."""
        import json as _json
        if isinstance(character, dict):
            character = self._load_persona(character)
        return self.engine.render(
            'simulations/character_response_iterative.poml',
            {
                'character': character,
                'situation': situation,
                'emphasis': emphasis,
                'temperature': 0.8,
                'previous_responses_json': _json.dumps((previous_responses or [])[-3:], ensure_ascii=False),
            }
        )

    def get_character_prompt_iterative_roles(
        self,
        character,
        situation: str,
        emphasis: str = "neutral",
        previous_responses: Optional[List[Dict[str, Any]]] = None,
        world_pov: str = "",
    ) -> Dict[str, str]:
        import json as _json
        if isinstance(character, dict):
            character = self._load_persona(character)
        # Apply runtime flags overlay
        character = self._apply_runtime_flags(character)
        context_text = self._load_context(character)
        return self.engine.render_roles(
            'simulations/character_response_iterative.poml',
            {
                'character': character,
                'situation': situation,
                'context': context_text,
                'world_pov': world_pov,
                'emphasis': emphasis,
                'temperature': 0.8,
                'previous_responses_json': _json.dumps((previous_responses or [])[-3:], ensure_ascii=False),
            }
        )

# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create engine
    engine = create_engine()
    
    # Example character data
    character_data = {
        'id': 'pontius_pilate',
        'name': 'Pontius Pilate',
        'backstory': {
            'origin': 'Roman nobility',
            'career': 'Prefect of Judaea'
        },
        'traits': ['pragmatic', 'cautious', 'politically minded'],
        'values': ['order', 'Roman authority', 'self-preservation'],
        'fears': ['uprising', 'loss of position', 'divine judgment'],
        'desires': ['peace', 'advancement', 'understanding'],
        'emotional_state': {
            'anger': 0.2,
            'doubt': 0.7,
            'fear': 0.5,
            'compassion': 0.3,
            'confidence': 0.4
        },
        'memory': {
            'recent_events': [
                'Received warning from wife about dream',
                'Interrogated the accused privately',
                'Crowd demands crucifixion'
            ]
        },
        'current_goal': 'Maintain order while avoiding injustice',
        'internal_conflict': 'Duty to Rome vs. sense of justice'
    }
    
    # Render character response template
    prompt = engine.render(
        'templates/simulations/character_response.poml',
        {
            'character': character_data,
            'situation': 'The crowd grows violent, demanding blood',
            'emphasis': 'fear',
            'temperature': 0.9
        }
    )
    
    print("Rendered Character Response Prompt:")
    print("-" * 50)
    print(prompt)
    print("-" * 50)
    
    # List available templates
    print("\nAvailable Templates:")
    for template in engine.list_templates():
        print(f"  - {template}")
