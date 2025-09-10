---
name: "Agent Task: Cache Engineer"
about: Edits-aware cache and optional hot reload
title: "PERF-01: Cache fingerprint + hot reload (dev)"
labels: ["agent:cache_engineer", "area:engine", "type:feature", "priority:p2", "size:S"]
assignees: []
---

## Summary
Ensure template edits invalidate cache; add optional file watcher in dev mode.

## Acceptance
- Cache invalidates on mtime/content hash change
- Dev hot reload clears stale cache entries

