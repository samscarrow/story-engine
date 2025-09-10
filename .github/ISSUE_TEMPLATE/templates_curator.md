---
name: "Agent Task: Templates Curator"
about: Curate fixtures and golden outputs; scrub path prefixes
title: "TPL-01: Goldens + path normalization"
labels: ["agent:templates_curator", "area:templates", "type:feature", "priority:p1", "size:S"]
assignees: []
---

## Summary
Curate representative templates and deterministic fixtures; migrate references to relative paths.

## Acceptance
- Goldens committed and pass in CI
- No `templates/` prefixes in code render calls

