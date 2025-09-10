# POML Integration Guide for Story Engine

## Overview

This guide explains how to integrate POML (Prompt Orchestration Markup Language) into the Story Engine's main engines and orchestrators. POML replaces string-based prompt generation with structured, maintainable templates.

## Architecture Overview

```
Story Engine
├── Core Engines (Original)
│   ├── character_simulation_engine_v2.py
│   ├── narrative_pipeline.py
│   └── llm_orchestrator.py
│
└── POML Integration Layer
    ├── poml/lib/poml_integration.py         # Core POML engine
    ├── poml/integration/
    │   ├── character_simulation_poml.py     # POML-enabled character engine
    │   └── llm_orchestrator_poml.py        # POML-enabled orchestrator
    └── poml/templates/                      # POML prompt templates
```

## Integration Strategies

### 1. Drop-in Replacement Strategy

Replace existing engines with POML-enabled versions:

```python
# Before - Original implementation
from character_simulation_engine_v2 import SimulationEngine

engine = SimulationEngine(llm_client, config)
result = await engine.run_simulation(character, situation)

# After - POML implementation
from poml.integration.character_simulation_poml import POMLCharacterSimulationEngine

engine = POMLCharacterSimulationEngine(llm_client, config)
result = await engine.run_simulation(character, situation, use_poml=True)
```

### 2. Adapter Pattern Strategy

Use adapters to add POML support to existing engines:

```python
from poml.lib.poml_integration import StoryEnginePOMLAdapter

# Wrap existing engine
adapter = StoryEnginePOMLAdapter()

# Use POML for prompt generation
prompt = adapter.get_character_prompt(character, situation, emphasis)

# Continue using existing engine for execution
response = await existing_engine.llm.generate_response(prompt)
```

### 3. Gradual Migration Strategy

Enable POML selectively with feature flags:

```python
class HybridSimulationEngine(SimulationEngine):
    def __init__(self, llm_client, config):
        super().__init__(llm_client, config)
        self.poml_enabled = config.get('use_poml', False)
        if self.poml_enabled:
            self.poml = POMLEngine()
    
    async def run_simulation(self, character, situation, **kwargs):
        if self.poml_enabled and kwargs.get('use_poml', True):
            prompt = self.poml.render('templates/simulations/character_response.poml', {...})
        else:
            prompt = character.get_simulation_prompt(situation, emphasis)
        
        return await self._execute_simulation(prompt, **kwargs)
```

## Step-by-Step Integration

### Step 1: Install Dependencies

```bash
# Install POML SDK
npm install @poml/sdk

# Install Python dependencies
pip install pyyaml jsonschema aiofiles

# Copy POML integration files
cp -r poml/ /path/to/story-engine/
```

### Step 2: Update Character Simulation Engine

**Option A: Modify Existing Engine**

```python
# In character_simulation_engine_v2.py

from poml.lib.poml_integration import POMLEngine

class SimulationEngine:
    def __init__(self, llm_client, config=None):
        # ... existing initialization ...
        
        # Add POML support
        self.poml = POMLEngine()
        self.use_poml = config.get('use_poml', True)
    
    def get_simulation_prompt(self, character, situation, emphasis="neutral"):
        if self.use_poml:
            return self.poml.render(
                'templates/simulations/character_response.poml',
                {
                    'character': character,
                    'situation': situation,
                    'emphasis': emphasis
                }
            )
        else:
            # Fallback to original string-based prompt
            return character.get_simulation_prompt(situation, emphasis)
```

**Option B: Use POML-Enabled Replacement**

```python
# Replace imports in your main script
# from character_simulation_engine_v2 import SimulationEngine
from poml.integration.character_simulation_poml import POMLCharacterSimulationEngine as SimulationEngine

# Rest of code remains unchanged
engine = SimulationEngine(llm_client, config)
```

### Step 3: Update Narrative Pipeline

```python
# In narrative_pipeline.py

from poml.lib.poml_integration import StoryEnginePOMLAdapter

class NarrativePipeline:
    def __init__(self):
        # ... existing initialization ...
        self.poml_adapter = StoryEnginePOMLAdapter()
    
    async def craft_scene(self, beat, characters, previous_context=""):
        # Use POML for scene crafting
        prompt = self.poml_adapter.get_scene_prompt(
            beat=beat,
            characters=characters,
            previous_context=previous_context
        )
        
        # Generate scene with LLM
        situation = await self.generate_with_llm(prompt)
        
        # ... rest of method unchanged ...
```

### Step 4: Update LLM Orchestrator

```python
# In llm_orchestrator.py

from poml.integration.llm_orchestrator_poml import POMLOrchestrator

class LLMOrchestrator:
    def __init__(self, providers):
        # ... existing initialization ...
        
        # Add POML orchestrator
        self.poml_orchestrator = POMLOrchestrator(providers)
    
    async def generate(self, prompt=None, template=None, data=None, **kwargs):
        # Support both old and new interfaces
        if template and data:
            # Use POML template
            return await self.poml_orchestrator.generate_with_template(
                template=template,
                data=data,
                **kwargs
            )
        else:
            # Use traditional prompt
            return await self._generate_traditional(prompt, **kwargs)
```

### Step 5: Update Configuration

```yaml
# In simulation_config.yaml

# Add POML configuration section
poml:
  enabled: true
  template_paths:
    - poml/templates/
    - poml/components/
  cache:
    enabled: true
    ttl_seconds: 3600
  validation:
    schemas: true
    strict_mode: false

# Update simulation settings
simulation:
  use_poml: true  # Enable POML by default
  fallback_to_string: true  # Allow fallback to string prompts
```

## Migration Examples

### Example 1: Migrating Character Response Generation

**Before (String-based):**
```python
# In character_simulation_engine_v2.py
prompt = f"""You are {character.name}.
Background: {character.backstory}
Traits: {', '.join(character.traits)}
Current emotion: Anger {character.emotional_state.anger}
Situation: {situation}
Respond with emphasis on {emphasis}."""
```

**After (POML-based):**
```python
# In character_simulation_engine_v2.py
prompt = self.poml.render(
    'templates/simulations/character_response.poml',
    {
        'character': character,
        'situation': situation,
        'emphasis': emphasis
    }
)
```

### Example 2: Migrating Scene Generation

**Before:**
```python
# In narrative_pipeline.py
prompt = f"""Create a dramatic scene for:
Beat: {beat['name']} - {beat['purpose']}
Tension level: {beat['tension']}
Characters: {', '.join([c['name'] for c in characters])}"""
```

**After:**
```python
# In narrative_pipeline.py
prompt = self.poml.render(
    'templates/narrative/scene_crafting.poml',
    {
        'beat': beat,
        'characters': characters,
        'previous_context': previous_context
    }
)
```

## Testing Integration

### Unit Tests

```python
# test_poml_integration.py

import pytest
from poml.integration.character_simulation_poml import POMLCharacterSimulationEngine

@pytest.mark.asyncio
async def test_poml_character_simulation():
    # Create test engine
    engine = POMLCharacterSimulationEngine(mock_llm, config)
    
    # Test POML template rendering
    result = await engine.run_simulation(
        character=test_character,
        situation="Test situation",
        use_poml=True
    )
    
    assert result['metadata']['used_poml'] == True
    assert 'response' in result

@pytest.mark.asyncio  
async def test_poml_fallback():
    # Test fallback to string prompts
    result = await engine.run_simulation(
        character=test_character,
        situation="Test situation", 
        use_poml=False
    )
    
    assert result['metadata']['used_poml'] == False
```

### A/B Testing

```python
# ab_test_poml.py

async def ab_test_prompts(engine, character, situations):
    results = {'poml': [], 'string': []}
    
    for situation in situations:
        # Test with POML
        poml_result = await engine.run_simulation(
            character, situation, use_poml=True
        )
        results['poml'].append(poml_result)
        
        # Test with string
        string_result = await engine.run_simulation(
            character, situation, use_poml=False  
        )
        results['string'].append(string_result)
    
    # Compare quality metrics
    return compare_results(results)
```

## Performance Optimization

### Template Caching

```python
# Enable template caching in config
config = {
    'poml': {
        'cache': {
            'enabled': True,
            'ttl_seconds': 3600,
            'max_size_mb': 100
        }
    }
}

# Templates are automatically cached after first use
engine = POMLCharacterSimulationEngine(llm, config)
```

### Batch Processing

```python
# Process multiple simulations efficiently
orchestrator = POMLOrchestrator(providers)

requests = [
    {
        'template': 'templates/simulations/character_response.poml',
        'data': {'character': char1, 'situation': sit1}
    },
    {
        'template': 'templates/simulations/character_response.poml',
        'data': {'character': char2, 'situation': sit2}
    }
]

results = await orchestrator.batch_generate(requests, max_concurrent=5)
```

## Monitoring and Metrics

```python
# Get POML usage metrics
metrics = orchestrator.get_metrics()
print(f"Total POML calls: {metrics['total_calls']}")
print(f"Template usage: {metrics['template_usage']}")
print(f"Cache hit rate: {metrics['cache_hits'] / metrics['total_calls']}")

# Benchmark POML performance
benchmark = await engine.benchmark_poml_performance(
    character=test_character,
    test_situations=situations,
    iterations=100
)
print(f"POML avg time: {benchmark['poml']['avg_time']}s")
print(f"String avg time: {benchmark['string']['avg_time']}s")
```

## Troubleshooting

### Common Issues and Solutions

**Template Not Found**
```python
# Check template exists
if not poml.validate_template('templates/my_template.poml'):
    logger.error("Template not found")
    # Fallback to string prompt
    prompt = generate_string_prompt(data)
```

**Data Binding Errors**
```python
# Validate data before rendering
try:
    prompt = poml.render(template, data)
except Exception as e:
    logger.error(f"POML render failed: {e}")
    # Use fallback
    prompt = fallback_prompt_generator(data)
```

**Schema Validation Failures**
```python
# Disable strict validation in development
config = {
    'poml': {
        'validation': {
            'strict_mode': False,
            'schemas': False  # Disable in dev
        }
    }
}
```

## Best Practices

### 1. Template Organization
- Keep templates under 200 lines
- Use components for reusable elements
- One template per specific use case

### 2. Data Preparation
```python
# Always preprocess data
def prepare_character_data(character):
    return {
        'id': character.id,
        'name': character.name,
        'traits': character.traits,
        'emotional_state': {
            'anger': float(character.emotional_state.anger),
            'doubt': float(character.emotional_state.doubt),
            # ... ensure all numeric values
        }
    }
```

### 3. Error Handling
```python
async def safe_poml_render(template, data, fallback_fn):
    try:
        return poml.render(template, data)
    except Exception as e:
        logger.error(f"POML failed: {e}")
        return fallback_fn(data)
```

### 4. Progressive Enhancement
```python
# Start with high-value templates
priorities = [
    'character_response',    # High complexity, high value
    'scene_crafting',        # Complex prompt structure
    'dialogue_generation',   # Frequently used
    'narrative_evaluation'   # Complex scoring logic
]

for template_type in priorities:
    migrate_to_poml(template_type)
    test_migration(template_type)
    deploy_if_successful(template_type)
```

## Migration Checklist

- [ ] Install POML dependencies
- [ ] Copy POML templates and configuration
- [ ] Create POML engine instance
- [ ] Update character simulation engine
- [ ] Update narrative pipeline
- [ ] Update LLM orchestrator
- [ ] Add POML configuration to settings
- [ ] Create unit tests for POML integration
- [ ] Run A/B tests comparing outputs
- [ ] Monitor performance metrics
- [ ] Document template usage
- [ ] Train team on POML syntax
- [ ] Set up template version control
- [ ] Establish template review process
- [ ] Deploy with feature flags

## Support and Resources

- **POML Documentation**: See `poml/README.md`
- **Template Gallery**: Browse `poml/gallery/`
- **Integration Examples**: Check `poml/integration/`
- **VS Code Extension**: Install POML Language Support
- **Testing Suite**: Run `pytest poml/tests/`

## Conclusion

POML integration transforms the Story Engine from string-based prompt management to a professional, maintainable template system. The integration can be done gradually, with immediate benefits in code organization, reusability, and maintainability.