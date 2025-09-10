---
name: "Agent Task: Orchestrator Refactorer"
about: Route all chat calls via role-separated prompts and normalize paths
title: "ENG-02: Adapter/orchestrator use roles under POML_ROLES"
labels: ["agent:orchestrator_refactorer", "area:orchestrator", "type:feature", "priority:p1", "size:S"]
assignees: []
---

## Summary
Adopt `generate_with_template_roles` when enabled and remove duplicate JSON parsing.

## Acceptance
- Character, iterative, and narrative paths use roles when flag enabled
- Providers without system support receive concatenated prompt

