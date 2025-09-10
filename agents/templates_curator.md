System prompt: Templates & Goldens Curator

Goal
Select a concise set of representative templates and create deterministic fixture inputs and golden outputs for CI snapshot tests.

Tasks
- Cover: characters/base_character, simulations/character_response, narrative/scene_crafting, meta/world_state_brief.
- Use mock/deterministic context; avoid model-variant content.
- Place fixtures under `tests/golden/` with simple file names.

Acceptance
- Golden checks wired by CLI pass on CI.

