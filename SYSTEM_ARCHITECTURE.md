# Story Engine System Architecture

## Overview
A multi-layered narrative generation system that creates, simulates, and iteratively enhances stories through character-driven interactions.

## System Components Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         STORY ENGINE SYSTEM                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Layer 1: Story Structure Generation                                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    StoryArcEngine                            │   │
│  │  ├── Generates narrative arc (3-act, 5-act, Hero's Journey) │   │
│  │  ├── Creates plot points with tension curves                │   │
│  │  └── Defines character arcs and thematic elements           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                       │
│  Layer 2: Scene Crafting                                            │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  SceneDescriptor Generator                   │   │
│  │  ├── Transforms plot points into detailed scenes            │   │
│  │  ├── Adds sensory details (sight, sound, atmosphere)       │   │
│  │  ├── Assigns character goals and emphasis                   │   │
│  │  └── Sets tension levels and pacing                         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                       │
│  Layer 3: Character Simulation                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │               CharacterSimulationEngine                     │   │
│  │  ├── CharacterState (emotions, memories, goals)            │   │
│  │  ├── LLM Integration (Gemma-2-27b)                         │   │
│  │  ├── Generates dialogue, thoughts, actions                 │   │
│  │  └── Tracks emotional evolution                            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                       │
│  Layer 4: Quality Evaluation                                        │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  StoryMetrics Evaluator                     │   │
│  │  ├── Pacing Score (scene length variance)                  │   │
│  │  ├── Consistency Score (character continuity)              │   │
│  │  ├── Emotional Range (diversity of emotions)               │   │
│  │  ├── Dialogue Naturalism (sentence variety)                │   │
│  │  ├── Plot Coherence (tension curve alignment)              │   │
│  │  └── Character Arc Completion (growth tracking)            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                       │
│  Layer 5: Iterative Enhancement                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                Enhancement Feedback Loop                    │   │
│  │  ├── Identifies weaknesses (pacing, emotion, conflict)     │   │
│  │  ├── Applies revision strategies (AMPLIFY, DEEPEN, etc)    │   │
│  │  ├── Explores narrative branches at high-tension points    │   │
│  │  └── Selects best path based on quality metrics            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↑                                       │
│                              └──────── Iterate until quality ────────┘
│                                        threshold met                 │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow Pipeline

```
1. INPUT STAGE
   ├── Story Title
   ├── Premise/Theme
   ├── Character Definitions
   └── Setting/World Details
           ↓
2. ARC GENERATION
   ├── Plot Points (with prerequisites & consequences)
   ├── Tension Curve (0.0 → 1.0 → 0.3)
   └── Character Arc Stages
           ↓
3. SCENE CRAFTING
   ├── Situation Description
   ├── Dramatic Question
   ├── Character Goals
   └── Sensory Details
           ↓
4. CHARACTER SIMULATION
   ├── Emotional State Calculation
   ├── LLM Prompt Construction
   ├── Response Generation (dialogue, thought, action)
   └── Emotional Shift Application
           ↓
5. EVALUATION
   ├── Scene Quality Metrics
   ├── Overall Story Coherence
   └── Feedback Generation
           ↓
6. ENHANCEMENT
   ├── Weakness Identification
   ├── Revision Strategy Selection
   ├── Scene Regeneration
   └── Branch Exploration
           ↓
7. OUTPUT
   ├── Final Story Scenes
   ├── Character Responses
   └── Quality Report
```

## Core Data Structures

### CharacterState
```python
CharacterState:
  - id: str
  - name: str
  - backstory: Dict
  - traits: List[str]
  - values: List[str]
  - fears: List[str]
  - desires: List[str]
  - emotional_state: EmotionalState
    - anger: float (0-1)
    - doubt: float (0-1)
    - fear: float (0-1)
    - compassion: float (0-1)
    - confidence: float (0-1)
  - memory: CharacterMemory
  - current_goal: str
  - internal_conflict: str
```

### SceneDescriptor
```python
SceneDescriptor:
  - id: int
  - name: str
  - situation: str
  - location: str
  - characters: List[str]
  - tension: float (0-1)
  - goals: Dict[str, str]
  - emphasis: Dict[str, str]
  - sensory: Dict[str, str]
```

### StoryMetrics
```python
StoryMetrics:
  - pacing_score: float (0-1)
  - consistency_score: float (0-1)
  - emotional_range: float (0-1)
  - dialogue_naturalism: float (0-1)
  - plot_coherence: float (0-1)
  - character_arc_completion: float (0-1)
  - tension_effectiveness: float (0-1)
  - thematic_clarity: float (0-1)
```

## Enhancement Strategies

```
AMPLIFY     → Intensify weak elements
SOFTEN      → Reduce overwhelming intensity
ACCELERATE  → Speed up pacing
DECELERATE  → Slow down pacing
DEEPEN      → Add complexity
SIMPLIFY    → Remove complexity
REDIRECT    → Change direction
MAINTAIN    → Keep as is
```

## Iteration Logic

```
START
  │
  ├─→ Generate Initial Story
  │
  ├─→ Evaluate Quality
  │     ├── IF quality >= threshold → END
  │     └── IF quality < threshold → Continue
  │
  ├─→ Generate Feedback
  │     ├── Identify weaknesses
  │     └── Select revision strategy
  │
  ├─→ Apply Enhancements
  │     ├── Revise scenes
  │     ├── Explore branches
  │     └── Re-simulate characters
  │
  └─→ Loop (max iterations)
```

## File Structure

```
story-engine/
│
├── Core Engines
│   ├── character_simulation_engine_v2.py  # Character behavior & emotions
│   ├── story_arc_engine.py               # Narrative structure generation
│   └── narrative_pipeline.py             # Arc → Scene → Character pipeline
│
├── Enhancement Systems
│   ├── iterative_story_system.py         # Quality metrics & revision
│   └── story_enhancement_loop.py         # Real-time enhancement
│
├── Multi-Character Systems
│   ├── multi_character_simulation.py     # Group interactions
│   └── complex_group_dynamics.py         # Faction dynamics
│
├── Configuration
│   ├── simulation_config.yaml            # LLM settings, parameters
│   └── cache_manager.py                  # Response caching
│
└── Testing
    ├── test_character_simulation.py      # Unit tests
    └── test_enhancement.py               # Enhancement tests
```

## API Usage Example

```python
# 1. Initialize pipeline
pipeline = NarrativePipeline(model="google/gemma-2-27b")

# 2. Define characters
characters = [
    {
        'id': 'protagonist',
        'name': 'Elena',
        'traits': ['brave', 'conflicted'],
        'goals': ['find truth', 'survive']
    }
]

# 3. Generate story
await pipeline.run_full_pipeline(
    title="The Discovery",
    premise="A scientist uncovers a dangerous secret",
    characters=characters,
    num_scenes=5
)

# 4. Enhance iteratively
enhancer = StoryEnhancementLoop()
enhanced_story = await enhancer.iterative_story_enhancement(
    title="The Discovery",
    premise="A scientist uncovers a dangerous secret",
    characters=characters,
    target_quality=0.75
)
```

## Key Features

1. **Modular Architecture**: Each layer can be used independently
2. **LLM Agnostic**: Supports multiple LLM providers (Gemma, OpenAI)
3. **Iterative Refinement**: Automatic quality improvement loops
4. **Branch Exploration**: Tests multiple narrative paths
5. **Emotion Tracking**: Characters evolve emotionally through story
6. **Quality Metrics**: 8-dimensional story quality assessment
7. **Memory Systems**: Characters remember previous events
8. **Sensory Details**: Atmospheric scene descriptions

## Performance Characteristics

- **Scene Generation**: 10-20 seconds per scene
- **Character Response**: 10-15 seconds per character
- **Enhancement Cycle**: 30-60 seconds per iteration
- **Quality Improvement**: Typically 20-40% after 3 iterations
- **Memory Usage**: ~100MB for typical 10-scene story

## Future Extension Points

1. **Agent Personas**: Specialized agents (Architect, Director, Critic)
2. **Reader Feedback**: Learn from audience responses
3. **Genre Templates**: Pre-configured story structures
4. **Collaborative Writing**: Multiple human/AI contributors
5. **Visual Generation**: Scene illustrations via image models
6. **Voice Acting**: Character dialogue synthesis
7. **Interactive Mode**: Reader choices affect story direction
## AI Load Balancer Client

The engine talks to an external AI load balancer (ai-lb) using an OpenAI‑style API. Client‑side resilience (retries, budgets, circuit breaker) and routing hints are documented in `docs/ai-lb-integration.md`.
