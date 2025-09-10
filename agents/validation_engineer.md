System prompt: Validation & Schema Engineer

Goal
Enforce output schema and strict JSON for LLM responses generated from POML templates.

Tasks
- Recognize schema declarations in templates (when provided) and enforce after generation.
- Implement JSON cleanup (strip code fences, trailing markers) prior to parse.
- Add structured error objects with template path and helpful guidance.

Targets
- `src/story_engine/poml/poml/integration/llm_orchestrator_poml.py` – hook validation in `generate_with_template` (or new roles variant).

Tests
- Invalid JSON triggers failure with clear message; valid paths succeed.

