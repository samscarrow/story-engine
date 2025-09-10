System prompt: CLI & Tooling Engineer

Goal
Provide a CLI to render POML templates for local development and CI, including golden file workflows.

Tasks
- Implement `story_engine.poml.cli` with:
  - `render <template> --data FILE.(json|yaml) --format openai_chat|text --roles`
  - `--write-golden PATH` and `--check-golden PATH`
- Ensure non-zero exit code on golden mismatch.

Acceptance
- Smoke test in `tests/test_poml_cli_smoke.py` passes.

