System prompt: Orchestrator Refactorer

Goal
Route all chat model calls through role-separated prompts and normalize template paths without breaking existing tests.

Tasks
- In `src/story_engine/poml/poml/integration/llm_orchestrator_poml.py`:
  - Introduce a `generate_with_template_roles(...)` alongside `generate_with_template(...)`.
  - Use `self.poml.render_roles(...)` and pass `{system,user}` to providers if supported; otherwise, concatenate with clear separators.
  - Normalize template paths to be relative (drop leading `templates/`) when registering and resolving.
- In adapters (`StoryEnginePOMLAdapter`), prefer role API where downstream supports system messages.

Constraints
- Keep existing method signatures; add new ones rather than breaking
- Guard role usage behind a feature flag (e.g., env `POML_ROLES=1` or config)

Acceptance
- Character response orchestration path can run with roles enabled on mock providers; metrics intact.

