# Agents Workflow: POML Overhaul

This repo includes automation to create and hand off scoped tickets to specialized agents.

## One-time setup (GitHub)
- Ensure GitHub Actions are enabled for the repo.
- Optional: create a GitHub Project for tracking (copy the URL for the bootstrap input).

## Create all tickets (bootstrap)
1. Go to Actions → "Bootstrap POML Overhaul Issues" → Run workflow
2. Inputs:
   - create_labels: true (default)
   - create_milestones: true (default)
   - project_url: optional URL of your GitHub Project (to auto-add issues)
3. The workflow will:
   - Create standard labels and milestones
   - Create issues from `agents/tickets.yml`, label and (optionally) assign
   - Post the agent system prompt as an issue comment (via `Agent Handoff` workflow)

## Manual issues (ad hoc)
- New issue → choose the appropriate "Agent Task: …" template
- The template applies an `agent:*` label; the Agent Handoff workflow posts the system prompt automatically.

## Local utilities
- List agent keys: `python -m story_engine.agents.cli list`
- Show a prompt: `python -m story_engine.agents.cli show engine_integrator`

## Tips
- Use CODEOWNERS to route reviews to the right people/teams.
- Use labels: `priority:p0|p1|p2`, `size:S|M|L` for planning.
- Use the PR template checklist to maintain quality gates (roles on/off, schema enforcement, goldens).

