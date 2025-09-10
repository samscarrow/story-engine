---
name: "Agent Task: Security Engineer"
about: Enforce allowed functions and block remote imports
title: "SEC-01: Template sandboxing + allowlist"
labels: ["agent:security_engineer", "area:security", "type:feature", "priority:p1", "size:S"]
assignees: []
---

## Summary
Honor `security.allowed_functions` and block remote imports in POML rendering; add tests for disallowed constructs.

## Acceptance
- Disallowed filters/imports fail with clear errors; tests cover both deny and allow

