from __future__ import annotations

from pathlib import Path
import os
import tokenize
import io


EXCLUDE_DIRS = {".git", ".venv", "dist", "build", "__pycache__", "node_modules", "bench/.venv"}


def iter_py_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(Path.cwd()):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS and not d.endswith(".egg-info")]
        for fn in filenames:
            if fn.endswith(".py"):
                yield Path(dirpath) / fn


def has_extraneous_fstrings(path: Path) -> bool:
    try:
        src = path.read_text(encoding="utf-8")
    except Exception:
        return False
    try:
        tokens = tokenize.generate_tokens(io.StringIO(src).readline)
        for tok in tokens:
            if tok.type == tokenize.STRING:
                s = tok.string
                # quick check: any 'f' in prefix and no braces in body
                i = 0
                while i < len(s) and s[i] in "rRbBuUfF":
                    i += 1
                if i < len(s) and s[i] in ('"', "'"):
                    # naive: just check for braces anywhere in token text
                    if ('f' in s[:i].lower()) and ('{' not in s) and ('}' not in s):
                        return True
    except Exception:
        return False
    return False


def test_no_extraneous_fstrings():
    offenders = []
    for path in iter_py_files(Path.cwd()):
        if "site-packages" in str(path):
            continue
        if has_extraneous_fstrings(path):
            offenders.append(str(path))
    assert not offenders, f"Extraneous f-strings found in: {offenders}"
