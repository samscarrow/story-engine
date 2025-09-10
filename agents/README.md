This folder contains scoped prompts for specialized coding agents working on the POML overhaul. Each sub-project is self-contained and references concrete files in this repo.

Projects:
- engine_integrator.md – swap in Microsoft POML SDK, strict mode, cache fingerprint
- orchestrator_refactorer.md – roles-first APIs, path normalization
- validation_engineer.md – schema enforcement and strict JSON
- tooling_engineer.md – CLI for rendering, goldens management
- templates_curator.md – fixtures and golden outputs
- cache_engineer.md – edits-aware cache + optional hot reload
- docs_ci_writer.md – CI wiring and docs
- security_engineer.md – enforce allowed functions and sandboxing

Follow AGENTS.md conventions if present and keep changes minimal and targeted.

