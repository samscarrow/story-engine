# Character Simulation Engine - Story Engine

[![Orchestrator Suite](https://github.com/samscarrow/story-engine/actions/workflows/orchestrator.yml/badge.svg?branch=main)](https://github.com/samscarrow/story-engine/actions/workflows/orchestrator.yml)

[![CI](https://github.com/samscarrow/story-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/samscarrow/story-engine/actions/workflows/ci.yml)
[![Tests](https://github.com/samscarrow/story-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/samscarrow/story-engine/actions/workflows/tests.yml)

A production-ready character behavior simulation system that generates authentic narrative content through psychological modeling rather than templated writing.

## 🎭 What We Built

**Core Engine:**
- `character_simulation_engine_v2.py` - Production character simulation with LLM abstraction
- `simulation_config.yaml` - Comprehensive configuration system
- `cache_manager.py` - Performance optimization through caching
- `test_character_simulation.py` - Complete test suite (28 tests ✅)

**Experiments:**
- `experiment.py` - Basic character behavior exploration
- `advanced_experiment.py` - Complex scenarios and character variants
- `dramatic_emotional_journey.py` - Emotional evolution through story sequences  
- `interactive_experiment.py` - Interactive character playground
- `real_llm_emotional_sequence.py` - Real LLM integration with emotional tracking
- `lmstudio_setup_guide.py` - LMStudio setup and model recommendations

## 🚀 Key Features

### Character Psychology System
- **Emotional States**: Anger, doubt, fear, compassion, confidence with dynamic evolution
- **Memory System**: Thread-safe character memory with event tracking
- **Internal Conflicts**: Complex psychological tensions and motivations
- **Trait-based Behavior**: Character responses based on personality traits and values

### LLM Integration
- **Multiple Providers**: MockLLM (testing), OpenAI, LMStudio (local models)
- **Structured Output**: JSON schema enforcement for consistent responses
- **Error Handling**: Retry logic with exponential backoff
- **Caching**: Performance optimization for repeated scenarios

### Production Features
- **Async Concurrency**: Configurable concurrent simulation limits
- **Configuration Management**: YAML-based settings for all parameters
- **Comprehensive Testing**: Unit tests, integration tests, performance testing
- **Logging**: Detailed logging for debugging and monitoring

## 🎬 Dramatic Results Achieved

### Emotional Evolution Visualization
```
🎭 DRAMATIC EMOTIONAL JOURNEY - PONTIUS PILATE'S TRANSFORMATION
================================================================================
Event                     Anger    Doubt    Fear     Comp     Conf    
--------------------------------------------------------------------------------
Initial State             0.10🧊    0.20❄️   0.20❄️   0.40💧    0.80🔥   
Initial Encounter         0.10🧊    0.30❄️   0.20❄️   0.40💧    0.80🔥   
Private Interrogation     0.10🧊    0.50💧    0.30❄️   0.40💧    0.80🔥   
Crowd Demands Blood       0.10🧊    0.70🌡️   0.50💧    0.40💧    0.80🔥   
Wife's Prophetic Dream    0.10🧊    0.95🔥    0.80🔥    0.60🌡️   0.80🔥   
Final Decision            0.30❄️   1.00🔥    1.00🔥    0.50💧    0.10🧊   
Washing Hands             0.30❄️   1.00🔥    0.90🔥    0.80🔥    0.10🧊   
```

**Most Dramatic Changes:**
- **DOUBT**: 📈 0.80 (confident → completely uncertain)
- **FEAR**: 📈 0.70 (calm → terrified)  
- **CONFIDENCE**: 📉 0.70 (authoritative → broken)
- **COMPASSION**: 📈 0.40 (dutiful → empathetic)

## 🎯 Character Simulation Examples

### Different Emphasis Modes
- **Power**: "Enough! I am the authority here! This man dies if I say he dies!"
- **Doubt**: "What is truth? You speak as one who knows certainties I cannot grasp..."
- **Fear**: "The crowd grows restless! If I don't act, Rome will have my head!"
- **Compassion**: "I find no fault in this man. Surely there must be another way..."

### Character Variants Tested
- **The Ruthless Administrator**: High anger/confidence, low compassion
- **The Tormented Idealist**: High doubt/compassion, philosophical responses
- **The Coward**: High fear, survival-focused decisions

## 🧪 How to Use

### Quick Start
```bash
# Basic experiment
python experiment.py

# Advanced scenarios
python advanced_experiment.py

# Dramatic emotional journey
python dramatic_emotional_journey.py

# Interactive playground
python interactive_experiment.py
```

### Environment & direnv (recommended)

- This repo uses direnv to manage an isolated virtualenv per project under `~/.venvs`.
- The `.envrc` provided will auto-create a venv at `~/.venvs/<slug>-py<major.minor>` and activate it on `direnv allow`.
- To set up:
  - Install direnv (https://direnv.net/), hook your shell, then run:
    - `direnv allow`
  - The venv path is exported as `VENV_PATH`; `PATH` is updated so `python`/`pip` use that venv.
  - The helper `scripts/venv-create.sh` is invoked automatically by `.envrc` if needed.

Using uv with direnv
- `.envrc` aliases uv to use the active venv: `uv -> uv --active`, `uv-sync`, and `uv-pip` are provided.
- Avoid creating a project-local `.venv` (which confuses tools). If you see a warning, remove it: `rm -rf .venv`.
- Common flows:
  - Full dev sync: `uv-sync --group dev`
  - Install a single package into the active venv: `uv-pip install pytest-asyncio`
  - Upgrade pytest stack: `uv-pip install -U pytest pytest-asyncio`

Deprecation note
- Using a local `.venv` directory is deprecated in this repo. Prefer direnv + `~/.venvs`.
- CI does not rely on `.venv`; it uses the runner’s Python and installs into an ephemeral environment.

Quick venv commands (without direnv)
- Create/activate the standard venv: `bash scripts/venv-create.sh && source "$VENV_PATH/bin/activate"`
- Install dev deps: `pip install -e . pytest pytest-asyncio`
- Run focused tests: `pytest -k lmstudio -q`


### Real LLM Integration
```bash
# Setup guide and test
python lmstudio_setup_guide.py

# Real LLM emotional sequence
python real_llm_emotional_sequence.py
```

### ai-lb Integration
- See `docs/ai-lb-integration.md` for a concise setup and verification guide.

### Observability & Metrics
- Orchestrator metrics and logging guidance: `docs/observability.md`.

### E2E (Live LM Studio)
- Use your running LM Studio instance for end‑to‑end checks:
  - Ensure LM Studio API is running (default `http://127.0.0.1:1234`).
  - `export LM_ENDPOINT=http://127.0.0.1:1234` (or your URL)
  - `make e2e` (tests will skip if LM Studio isn’t reachable)
  - CI: trigger the “E2E (LMStudio Live)” workflow manually and pass `lm_endpoint`.

### Database Health
- Oracle connectivity check:
  - Env: set `DB_TYPE=oracle` and provide `DB_USER`, `DB_PASSWORD`, `DB_DSN`, `DB_WALLET_LOCATION` (or `ORACLE_*` equivalents)
  - Health check: `python scripts/db_health.py --verbose` (use `--json` for machine output)
  - Minimal fast healthcheck: `python scripts/oracle_healthcheck.py --pool`
- Quick smoke test (insert/fetch): `python scripts/db_smoke_test.py --workflow oracle_smoke`
- Optional pooling/tuning via env:
  - `ORACLE_USE_POOL=1`, `ORACLE_POOL_MIN=1`, `ORACLE_POOL_MAX=4`, `ORACLE_POOL_INC=1`, `ORACLE_POOL_TIMEOUT=60`
  - Retries: `ORACLE_RETRY_ATTEMPTS=3`, `ORACLE_RETRY_BACKOFF=1.0`

#### Local Oracle XE (Docker)
- Start a local Oracle XE for development with:
  - `docker compose -f docker-compose.oracle.yml up -d`
- Configure env (no wallet needed):
  - `DB_TYPE=oracle`, `DB_USER=STORY_DB`, `DB_PASSWORD=story_pwd`, `DB_DSN=localhost/XEPDB1`
- Verify:
  - `python diagnose_oracle_connection.py` or `pytest -q test_simple_oracle.py`
- See `docs/oracle/local_dev.md` for details and troubleshooting.

#### Logging Configuration
- Configure logging via env (applies to CLI and workers):
  - `LOG_FORMAT=json|text`, `LOG_LEVEL=DEBUG|INFO|WARNING|ERROR`, `LOG_DEST=stdout|stderr|file`, `LOG_FILE_PATH=story_engine.log`, `LOG_SERVICE_NAME=story-engine`
  - Optional sampling: `LOG_SAMPLING_RATE=0.1` (10% info sampling)
  - Entry points call `init_logging_from_env()` from `core.common.observability`.
  - Context is automatic: after init, all logs include `service`, `trace_id`, and (when available) `correlation_id`. Message bus consumers propagate `correlation_id` for each handled message.

### Model Selection
- The engine can auto-select a viable text model when `LM_MODEL` and provider model are unset (LB routing). Use `model="auto"` for flexible routing (default).
- Prefer small models via:
  - Env: `LM_PREFER_SMALL=1`
  - YAML: `llm.prefer_small_models: true`
- Pinning (optional):
  - Use `LM_MODEL` to pin a specific model id (e.g., `LM_MODEL=qwen/qwen3-8b`).
  - `LMSTUDIO_MODEL` is deprecated; if set and `LM_MODEL` is not, it will be mapped to `LM_MODEL` for backward compatibility.

One-off selection helper:
  - Print chosen model: `python scripts/choose_model.py`
  - Prefer small: `python scripts/choose_model.py --prefer-small`
  - Export for current shell: `eval "$(python scripts/choose_model.py --prefer-small --export)"`
  - Write to .env: `python scripts/choose_model.py --prefer-small --write-env .env`

### Testing
```bash
# Run full test suite
python test_character_simulation.py -v

# Specific test categories
python test_character_simulation.py TestEmotionalState -v
```

#### Pytest (recommended)
#### Make targets
- `make test` – run the default pytest suite (excludes slow/oracle)
- `make test-slow` – run golden/silver prompt suites (`pytest -m slow`)
- `make test-golden-core` – run the core golden narratives (`pytest -m golden_core`)
- `make test-golden-extended` – run the extended golden narratives (`pytest -m golden_extended`)
- `make test-oracle` – run Oracle integration tests (`pytest -m oracle`)
- `make test-live-e2e` – run live LM Studio acceptance tests (`RUN_LIVE_E2E=1`)
- `make test-lmstudio` – smoke-test the LM Studio circuit
- `make test-orchestrator` – targeted orchestrator checks
- `make e2e` – deterministic e2e stubs

- Fast (excludes slow by default):
  - `pytest -q` or `pytest -n auto --dist=loadfile -q`
- Only slow/live tests:
  - `pytest -q -m slow`
- Opt-in live run with shorter LLM timeouts:
  - `STORY_ENGINE_LIVE=1 LLM_TEST_TIMEOUT=20 pytest -q -m slow`

CI
- GitHub Actions workflow `.github/workflows/tests.yml` runs unit tests on push/PR.
- Live tests are opt-in via workflow_dispatch; set `run_live=true` when triggering manually.

## 🤖 LLM Setup for Real Testing

### Recommended Models for Character Simulation:
1. **Qwen2.5-7B-Instruct** ⭐⭐⭐⭐⭐ - Best overall choice
2. **Llama-3.2-3B-Instruct** ⭐⭐⭐⭐ - Fast and efficient  
3. **Mistral-7B-Instruct-v0.3** ⭐⭐⭐⭐ - Creative responses
4. **Phi-3.5-Mini-Instruct** ⭐⭐⭐ - Compact and fast

### LMStudio Setup:
1. Download LMStudio from https://lmstudio.ai/
2. Load a recommended model
3. Start local server (localhost:1234)
4. Enable structured output in settings
5. Run `python lmstudio_setup_guide.py` to test

### Model Selection
- The engine filters out non-text models (embeddings, TTS/STT) when discovering models from `/v1/models`.
- Prefer small models (<= ~4B) by setting either:
  - Env var: `LM_PREFER_SMALL=1`
  - YAML: in `config.yaml` under `llm.prefer_small_models: true`
  The orchestrator and CLI will then pick small text models first when auto-selecting.

## 📊 Performance Stats

- **Test Coverage**: 28 comprehensive tests passing
- **Concurrent Simulations**: Configurable (default: 10)
- **Cache Hit Rate**: ~85% with properly configured caching
- **Response Time**: <100ms with MockLLM, 1-3s with real LLM
- **Memory Usage**: Efficient with automatic cleanup

## 🎨 Character Creation System

### Define Characters with:
```python
character = CharacterState(
    id="unique_id",
    name="Character Name",
    backstory={"origin": "...", "career": "..."},
    traits=["trait1", "trait2"],
    values=["value1", "value2"],
    fears=["fear1", "fear2"],
    desires=["desire1", "desire2"],
    emotional_state=EmotionalState(anger=0.3, doubt=0.7, ...),
    memory=CharacterMemory(),
    current_goal="Character's objective",
    internal_conflict="Core psychological tension"
)
```

### Generate Responses:
```python
result = await engine.run_simulation(
    character,
    situation="Your scenario here",
    emphasis="power|doubt|fear|compassion|duty",
    temperature=0.8
)
```

## 🏆 What Makes This Special

### Authentic Narrative Generation
- **Psychology-First**: Characters driven by internal states, not templates
- **Dynamic Evolution**: Emotional states change based on story events
- **Contextual Memory**: Characters remember and reference past events
- **Multiple Perspectives**: Same situation generates different responses based on emphasis

### Production Ready
- **Error Handling**: Comprehensive retry and fallback systems
- **Performance**: Caching, concurrency limits, resource management
- **Testing**: Full test coverage with mocking and integration tests
- **Configuration**: Flexible YAML-based configuration system

### Research Applications
- **Narrative Psychology**: Study how characters evolve through story events
- **Decision Making**: Explore how emotional states influence choices
- **Character Development**: Test different personality combinations
- **Interactive Fiction**: Dynamic character responses for games/stories

## 🎯 Next Steps

1. **Connect Real LLM**: Follow LMStudio setup guide for realistic responses
2. **Create New Characters**: Build your own character templates
3. **Design Story Sequences**: Create emotional journey experiments
4. **Integrate into Projects**: Use as backend for interactive fiction or games
5. **Extend Psychology Model**: Add new emotional dimensions or memory types

The system is ready for both experimentation and production use - from character psychology research to powering dynamic narrative experiences!

## 🏃 One-Command Demo

Generate an evaluatable narrative, continuity report, metrics, and a human-readable summary:

```bash
python -m story_engine.scripts.run_demo --runs 3 --emphasis doubt
```

Outputs are written to `dist/run-<timestamp>/`:
- `story.json` – scene plan, runs, narrative graph
- `continuity_report.json` – violations and suggested fixes
- `metrics.json` – schema validity, continuity, repetition
- `console.md` – readable sample with key stats
- `config.snapshot.yaml`, `env.capture` – reproducibility snapshot

Use `--live` to route through the unified orchestrator (if configured), `--use-poml` to enable POML prompts, and `--strict-persona` to enforce persona guardrails.

Note: The demo smoke test is included in CI and runs on push/PR to ensure artifacts are generated and well-formed.
