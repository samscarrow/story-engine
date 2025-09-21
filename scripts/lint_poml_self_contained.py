#!/usr/bin/env python3
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = ROOT / "src" / "story_engine" / "poml" / "poml" / "templates"

PATTERNS = [
    (re.compile(r"\bsee\s+(world|schema|template|above|below|doc|docs)\b", re.I),
     "Avoid external references; inline the necessary content"),
    (re.compile(r"\brefer\s+to\b", re.I),
     "Avoid external references; make template self-contained"),
    (re.compile(r"\bas\s+(above|previously|described|noted)\b", re.I),
     "Avoid cross-references like 'as above'; repeat instructions"),
]

def scan_file(path: Path):
    issues = []
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        issues.append((0, "read_error", str(e)))
        return issues
    # Rule: if template demands strict/valid JSON, ensure a JSON code block with schema is present
    lower = text.lower()
    if ("valid json object" in lower or "strict json" in lower):
        has_code = "```json" in lower
        # Also allow plain JSON block starting with '{' on its own line in the template
        has_plain = any(line.strip().startswith("{") for line in text.splitlines())
        if not (has_code or has_plain):
            issues.append((1, "Template demands strict/valid JSON but no JSON schema/sample found", ""))
    lines = text.splitlines()
    for i, line in enumerate(lines, 1):
        for pat, msg in PATTERNS:
            m = pat.search(line)
            if m:
                snippet = line.strip()
                issues.append((i, msg, snippet))
    return issues

def main() -> int:
    if not TEMPLATES_DIR.exists():
        print(f"No templates dir: {TEMPLATES_DIR}")
        return 0
    failures = 0
    for p in TEMPLATES_DIR.rglob("*.poml"):
        file_issues = scan_file(p)
        if file_issues:
            failures += len(file_issues)
            for (lineno, msg, snippet) in file_issues:
                print(f"POML Lint: {p}:{lineno}: {msg}: {snippet}")
    if failures:
        print(f"POML Lint failed with {failures} issue(s)")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
