---
name: "Agent Task: Validation Engineer"
about: Enforce schema and strict JSON in orchestrator paths
title: "VAL-01: Schema enforcement + JSON cleanup"
labels: ["agent:validation_engineer", "area:validation", "type:feature", "priority:p0", "size:M"]
assignees: []
---

## Summary
Add schema detection from templates and enforce required keys; clean JSON and raise structured errors on failure.

## Acceptance
- Templates with declared schema are validated
- Structured errors include template, schema, and previews

