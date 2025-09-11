from pathlib import Path
from typing import List


AGENTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "agents"


def list_agents() -> List[str]:
    if not AGENTS_DIR.exists():
        return []
    return sorted(p.stem for p in AGENTS_DIR.glob("*.md"))


def get_agent_prompt(agent: str) -> str:
    path = AGENTS_DIR / f"{agent}.md"
    if not path.exists():
        # allow dashed vs underscored forms
        alt = agent.replace(":", "_").replace("-", "_")
        path = AGENTS_DIR / f"{alt}.md"
    if not path.exists():
        raise FileNotFoundError(f"Agent prompt not found: {agent}")
    return path.read_text(encoding="utf-8")
