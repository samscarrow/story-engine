# Hardcoded Prompts Analysis & POML Integration Points

## Overview
This document identifies all hardcoded prompt templates in the Story Engine codebase and their corresponding POML integration points.

## 1. Character Simulation Engine (`character_simulation_engine_v2.py`)

### Hardcoded Prompts Found

#### Location: Lines 387-413 - Character Simulation Prompt
```python
context = f"""You are {self.name}.
            
Background: {self.backstory.get('origin', 'Unknown')}
Career: {self.backstory.get('career', 'Unknown')}

Core Traits: {', '.join(self.traits)}
Core Values: {', '.join(self.values)}
Core Fears: {', '.join(self.fears)}
Core Desires: {', '.join(self.desires)}

Relationships:
{chr(10).join([f"  - {char_id}: {rel.value}" for char_id, rel in self.relationships.items()])}

Current emotional state:
  Anger: {self.emotional_state.anger:.2f}
  Doubt: {self.emotional_state.doubt:.2f}  
  Fear: {self.emotional_state.fear:.2f}
  Compassion: {self.emotional_state.compassion:.2f}
  Confidence: {self.emotional_state.confidence:.2f}

Emphasis for this response: {emphasis.upper()}

Current Situation: {situation}

Based on your character, emotional state, and the emphasis mode, provide your response.
"""
```

**POML Replacement:** `templates/simulations/character_response.poml`

### Integration Point
```python
# Replace lines 387-413 with:
from poml.lib.poml_integration import POMLEngine

class CharacterState:
    def __init__(self):
        self.poml = POMLEngine()
    
    def get_simulation_prompt(self, situation, emphasis="neutral"):
        return self.poml.render(
            'templates/simulations/character_response.poml',
            {
                'character': self,
                'situation': situation,
                'emphasis': emphasis
            }
        )
```

## 2. Multi-Character Simulation (`multi_character_simulation.py`)

### Hardcoded Prompts Found

#### Location: Lines 107-117 - Multi-Character Response Prompt
```python
system_prompt = f"""You are {character.name}, {character.role}.
Personality: {', '.join(character.personality)}
Goals: {', '.join(character.goals)}
Fears: {', '.join(character.fears)}
Current emotions: Anger={character.emotional_state['anger']:.1f}, Fear={character.emotional_state['fear']:.1f}, Confidence={character.emotional_state['confidence']:.1f}
Others present: {', '.join(others_context)}
Recent memories: {'; '.join(character.memories[-3:])}
Emphasis: {emphasis}

Respond with JSON only:
{{"dialogue": "what you say", "thought": "inner thoughts", "action": "physical action", "target": "who you're addressing/acting toward", "emotional_shift": {{"anger": 0, "fear": 0, "confidence": 0}}}}"""
```

**POML Replacement:** `templates/simulations/multi_character_response.poml`

### Integration Point
```python
# In generate_response method (line 94):
async def generate_response(self, character, situation, other_characters, emphasis="neutral"):
    if hasattr(self, 'poml'):
        system_prompt = self.poml.render(
            'templates/simulations/multi_character_response.poml',
            {
                'character': character,
                'situation': situation,
                'others': [self.characters[oid] for oid in other_characters if oid in self.characters],
                'emphasis': emphasis
            }
        )
    else:
        # Fallback to original string prompt
        system_prompt = f"""You are {character.name}..."""
```

## 3. Complex Group Dynamics (`complex_group_dynamics.py`)

### Hardcoded Prompts Found

#### Location: Lines 151-169 - Group Dynamics Prompt
```python
system_prompt = f"""You are {character.name}, {character.role} of the {character.faction.value}.
Personality: {', '.join(character.personality)}
Motivations: {', '.join([f"{m}({v:.1f})" for m, v in character.motivations.items()])}
Faction loyalty: {character.faction_loyalty:.1f}
Influence: {character.influence:.1f}
Current emotions: Anger={character.emotional_state['anger']:.1f}, Fear={character.emotional_state['fear']:.1f}, Confidence={character.emotional_state['confidence']:.1f}
Others present: {', '.join(others_context)}
Alliances: {', '.join(alliances)}
Recent events: {'; '.join(character.memories[-2:])}
Recent actions: {'; '.join(action_context)}

Current situation: {situation}
Stage: {dynamics.stage.value}
{phase_context}

Respond with dialogue, thought, action, and who you're targeting/influencing.
Format: {{"dialogue": "...", "thought": "...", "action": "...", "target": "...", "action_type": "speak/whisper/gesture/move/confront"}}"""
```

**POML Replacement:** `templates/simulations/group_dynamics_response.poml`

## 4. Narrative Pipeline (`narrative_pipeline.py`)

### Hardcoded Prompts Found

#### Location: Lines 101-110 - Scene Creation Prompt
```python
prompt = f"""Create a dramatic scene for:
Beat: {beat['name']} - {beat['purpose']}
Tension level: {beat['tension']}
Characters: {', '.join([c['name'] for c in characters])}
Previous context: {previous_context if previous_context else 'Opening scene'}

Provide a detailed situation description (2-3 sentences) that gives characters clear dramatic opportunities.
Include: location, time of day, immediate conflict or tension, and what's at stake.

Scene situation:"""
```

**POML Replacement:** `templates/narrative/scene_crafting.poml`

#### Location: Lines 204-214 - Character Dialogue Prompt
```python
char_context = f"""You are {character['name']}, {character.get('role', 'character')}.
Traits: {', '.join(character.get('traits', ['complex']))}
Current emotion focus: {scene.emphasis.get(character['id'], 'neutral')}
Goal: {scene.goals.get(character['id'], 'respond appropriately')}

Scene atmosphere: {scene.sensory.get('atmosphere', 'tense')}
You can hear: {scene.sensory.get('sound', 'ambient noise')}
You can see: {scene.sensory.get('sight', 'the scene before you')}

Respond with JSON only:
{{"dialogue": "what you say", "thought": "inner monologue", "action": "physical action"}}"""
```

**POML Replacement:** `templates/narrative/dialogue_generation.poml`

## 5. Story Arc Engine (`story_arc_engine.py`)

### Hardcoded Prompts Found

#### Location: Lines 356-375 - Scene Design Prompt
```python
system_prompt = f"""You are a narrative scene designer. Create a detailed scene.
Plot point: {plot_point.name} - {plot_point.description}
Scene type: {plot_point.scene_type.value}
Tension level: {plot_point.tension_level}
Required characters: {', '.join(plot_point.required_characters)}
{context}

Requirements:
1. Setting (location, time, atmosphere)
2. Starting situation (what's happening)
3. Central conflict or tension
4. Character objectives
5. Sensory details (sights, sounds, atmosphere)
6. Emotional tone

Format as JSON with keys: setting, situation, conflict, objectives (dict), sensory (dict), tone"""

user_prompt = f"Create scene: {plot_point.name} at {plot_point.location or 'appropriate location'}"
```

**POML Replacement:** `templates/narrative/scene_design.poml`

## Integration Strategy

### Phase 1: Add POML Support (Non-Breaking)
1. Install POML dependencies
2. Add POMLEngine instances to existing classes
3. Add configuration flags for POML usage
4. Implement fallback logic

### Phase 2: Create POML Templates
Templates needed:
- [x] `character_response.poml` - Basic character response
- [ ] `multi_character_response.poml` - Multi-character interactions
- [ ] `group_dynamics_response.poml` - Complex group dynamics
- [x] `scene_crafting.poml` - Scene generation
- [ ] `dialogue_generation.poml` - Character dialogue
- [ ] `scene_design.poml` - Detailed scene design

### Phase 3: Implementation Code

#### Step 1: Update Character Simulation Engine
```python
# character_simulation_engine_v2.py
from poml.lib.poml_integration import POMLEngine

class CharacterState:
    def __init__(self):
        # Add POML support
        self.use_poml = True  # Feature flag
        if self.use_poml:
            self.poml = POMLEngine()
    
    def get_simulation_prompt(self, situation, emphasis="neutral"):
        if self.use_poml and hasattr(self, 'poml'):
            return self.poml.render(
                'templates/simulations/character_response.poml',
                {
                    'character': self.__dict__,
                    'situation': situation,
                    'emphasis': emphasis
                }
            )
        else:
            # Fallback to original implementation
            return f"""You are {self.name}..."""
```

#### Step 2: Update Narrative Pipeline
```python
# narrative_pipeline.py
from poml.lib.poml_integration import StoryEnginePOMLAdapter

class NarrativePipeline:
    def __init__(self):
        self.poml_adapter = StoryEnginePOMLAdapter()
        self.use_poml = True
    
    async def craft_scene(self, beat, characters, previous_context=""):
        if self.use_poml:
            prompt = self.poml_adapter.get_scene_prompt(
                beat, characters, previous_context
            )
        else:
            # Original string prompt
            prompt = f"""Create a dramatic scene..."""
        
        return await self.generate_with_llm(prompt)
```

#### Step 3: Update Multi-Character Simulation
```python
# multi_character_simulation.py
class MultiCharacterEngine:
    def __init__(self, model="google/gemma-2-27b"):
        self.model = model
        self.poml = POMLEngine()
        self.use_poml = True
    
    async def generate_response(self, character, situation, other_characters, emphasis="neutral"):
        if self.use_poml:
            system_prompt = self.poml.render(
                'templates/simulations/multi_character_response.poml',
                {
                    'character': character,
                    'situation': situation,
                    'others': [self.characters[oid] for oid in other_characters],
                    'emphasis': emphasis
                }
            )
        else:
            # Original implementation
            system_prompt = self._build_string_prompt(character, situation, other_characters, emphasis)
```

## Testing Strategy

### Unit Tests
```python
# test_poml_integration.py
import pytest
from poml.lib.poml_integration import POMLEngine

def test_character_prompt_rendering():
    engine = POMLEngine()
    character_data = {
        'name': 'Test Character',
        'traits': ['brave', 'loyal'],
        'emotional_state': {'anger': 0.5}
    }
    
    prompt = engine.render(
        'templates/simulations/character_response.poml',
        {'character': character_data, 'situation': 'test'}
    )
    
    assert 'Test Character' in prompt
    assert 'brave' in prompt
```

### A/B Testing
```python
async def compare_prompt_quality():
    results = {
        'poml': [],
        'string': []
    }
    
    # Test with POML
    engine.use_poml = True
    poml_result = await engine.run_simulation(character, situation)
    
    # Test with strings
    engine.use_poml = False
    string_result = await engine.run_simulation(character, situation)
    
    # Compare coherence, format compliance, etc.
```

## Benefits of Migration

1. **Maintainability**: Prompts in separate files, not buried in code
2. **Reusability**: Components shared across templates
3. **Version Control**: Track prompt evolution separately
4. **Type Safety**: Validated data binding
5. **Visual Editing**: VS Code extension support
6. **A/B Testing**: Easy prompt variant testing
7. **Documentation**: Self-documenting template structure

## Migration Priority

1. **High Priority** (Complex, frequently modified):
   - `character_response.poml` ✅
   - `scene_crafting.poml` ✅
   - `multi_character_response.poml`

2. **Medium Priority** (Stable but complex):
   - `group_dynamics_response.poml`
   - `dialogue_generation.poml`

3. **Low Priority** (Simple or rarely modified):
   - `scene_design.poml`
   - `sensory_details.poml`

## Configuration

Add to project configuration:
```yaml
# config.yaml
poml:
  enabled: true
  template_paths:
    - poml/templates/
    - poml/components/
  cache:
    enabled: true
    ttl: 3600
  fallback_to_string: true  # Safety net during migration
```

## Next Steps

1. [ ] Create missing POML templates
2. [ ] Add POMLEngine to each core class
3. [ ] Implement feature flags for gradual rollout
4. [ ] Write comprehensive tests
5. [ ] Document template schemas
6. [ ] Train team on POML syntax
7. [ ] Deploy with monitoring