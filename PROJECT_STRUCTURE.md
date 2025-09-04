# Story Engine Project Structure

## Overview
A modular AI-powered story generation system with LLM-agnostic orchestration.

## Directory Structure

```
story-engine/
│
├── README.md                     # Project overview
├── SYSTEM_ARCHITECTURE.md        # Detailed architecture documentation
├── PROJECT_STRUCTURE.md          # This file
├── llm_config.json              # LLM provider configuration
├── simulation_config.yaml       # Character simulation settings
│
├── core/                        # Core system modules
│   ├── orchestration/          # LLM orchestration layer
│   │   ├── llm_orchestrator.py         # Base orchestration with fallbacks
│   │   └── llm_orchestrator_strict.py  # Strict version (no silent failures)
│   │
│   ├── story_engine/           # Story generation components
│   │   ├── story_arc_engine.py         # Plot structure generation
│   │   ├── narrative_pipeline.py       # Scene crafting pipeline
│   │   ├── iterative_story_system.py   # Quality evaluation & metrics
│   │   ├── story_enhancement_loop.py   # Enhancement strategies
│   │   └── story_engine_orchestrated.py # Integrated story engine
│   │
│   └── character_engine/       # Character simulation
│       ├── character_simulation_engine_v2.py  # Core character engine
│       ├── multi_character_simulation.py      # Multi-character interactions
│       └── complex_group_dynamics.py          # Group dynamics & factions
│
├── tests/archive/              # Archived test files (19 files)
└── feedback/archive/           # Archived feedback/analysis (13 files)
```

## Core Components

### 1. LLM Orchestration Layer
- **Purpose**: Provider-agnostic LLM interface
- **Key Features**:
  - Supports multiple providers (LMStudio, KoboldCpp, Ollama, etc.)
  - Automatic fallback chains
  - Health monitoring
  - No silent failures (strict mode)
  - Detailed error reporting

### 2. Story Engine
- **Purpose**: Generate complete narratives
- **Components**:
  - **Story Arc Engine**: Creates plot structures (3-act, 5-act, Hero's Journey)
  - **Narrative Pipeline**: Transforms plot points into detailed scenes
  - **Iterative Story System**: 8 quality metrics for evaluation
  - **Enhancement Loop**: Revision strategies (AMPLIFY, DEEPEN, etc.)

### 3. Character Engine
- **Purpose**: Simulate believable character behaviors
- **Features**:
  - Emotional state tracking (5+ dimensions)
  - Multi-character interactions
  - Relationship dynamics
  - Faction systems
  - Memory and context awareness

## Configuration Files

### llm_config.json
Defines LLM providers, endpoints, and generation parameters:
- Provider configurations
- Active provider selection
- Fallback chains
- Generation profiles (creative, balanced, focused)

### simulation_config.yaml
Character simulation settings:
- Emotional dimensions
- Interaction parameters
- Memory settings

## Usage

### Basic Story Generation
```python
from core.story_engine.story_engine_orchestrated import OrchestratedStoryEngine

engine = OrchestratedStoryEngine("llm_config.json")
story = await engine.generate_complete_story(story_request)
```

### Character Simulation
```python
from core.character_engine.character_simulation_engine_v2 import SimulationEngine

engine = SimulationEngine(llm_client)
response = await engine.simulate_character_response(scene, character)
```

### LLM Orchestration
```python
from core.orchestration.llm_orchestrator_strict import StrictLLMOrchestrator

orchestrator = StrictLLMOrchestrator()
response = await orchestrator.generate(prompt, allow_fallback=True)
```

## Key Design Principles

1. **No Mock Data**: All responses from real LLMs
2. **No Silent Failures**: Explicit error handling and reporting
3. **Modular Architecture**: Each component can be used independently
4. **Provider Agnostic**: Easy to switch between LLM providers
5. **Quality Focused**: Multiple evaluation metrics and enhancement strategies

## Archived Files
- **tests/archive/**: 19 test and experimental files
- **feedback/archive/**: 13 feedback collection and analysis files

These files are preserved for reference but not part of the active codebase.