---
name: "Agent Task: Engine Integrator"
about: Integrate Microsoft POML SDK and strict rendering
title: "ENG-01: Integrate POML SDK (primary) + native fallback"
labels: ["agent:engine_integrator", "area:engine", "type:feature", "priority:p0", "size:M"]
assignees: []
---

## Summary
Implement SDK-backed rendering in `src/story_engine/poml/poml/lib/poml_integration.py` with native fallback, strict mode, and edits-aware caching.

## Acceptance
- SDK path renders; fallback works if SDK absent
- Cache includes template content hash
- `render_roles` passes `format=openai_chat` when SDK available
- Tests green

## Notes
System prompt is auto-commented by workflow when the `agent:engine_integrator` label is applied.

