#!/usr/bin/env python3
"""
Remove unnecessary f-string prefixes (f/F) from string literals that have no
interpolations (i.e., contain no '{' or '}' anywhere in the literal body).

Safe approach: use tokenize to only rewrite string tokens; preserve spacing and comments.
"""
from __future__ import annotations

import io
import os
import sys
import tokenize
from pathlib import Path
from typing import Iterable


EXCLUDE_DIRS = {".git", ".venv", "dist", "build", "__pycache__", "node_modules"}


def iter_py_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        # prune excluded dirs
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS and not d.endswith(".egg-info")]
        for fn in filenames:
            if fn.endswith(".py"):
                yield Path(dirpath) / fn


def strip_extraneous_fprefix(src: str) -> str:
    out_tokens = []
    changed = False
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except tokenize.TokenError:
        return src  # leave file unchanged on tokenization errors

    for tok in tokens:
        if tok.type == tokenize.STRING:
            s = tok.string
            # String token can include prefixes like rRbBuUfF in any order, and quotes (' " or triple)
            # We only want to remove an f/F prefix if the string body has no '{' or '}'.
            prefix = ""
            i = 0
            while i < len(s) and s[i] in "rRbBuUfF":
                prefix += s[i]
                i += 1
            # Extract quote type
            if i >= len(s):
                out_tokens.append(tok)
                continue
            if s[i] not in ('"', "'"):
                out_tokens.append(tok)
                continue
            quote = s[i]
            # Determine triple vs single
            is_triple = s[i:i+3] in {"'''", '"""'}
            if is_triple:
                q = s[i:i+3]
                j = s.rfind(q)
                body = s[i+3:j] if j >= 0 else ""
            else:
                # find closing quote; naive but fine since token is a full string literal already
                j = len(s) - 1
                body = s[i+1:j] if j > i else ""

            has_brace = ("{" in body) or ("}" in body)
            if not has_brace and ("f" in prefix.lower()):
                # rebuild prefix without f/F
                new_prefix = "".join(ch for ch in prefix if ch.lower() != "f")
                new_s = new_prefix + s[i:]
                # Replace token
                tok = tokenize.TokenInfo(tok.type, new_s, tok.start, tok.end, tok.line)
                changed = True
        out_tokens.append(tok)

    if not changed:
        return src
    return tokenize.untokenize(out_tokens)


def main(argv: list[str] | None = None) -> int:
    root = Path(argv[0]).resolve().parent if argv else Path.cwd()
    repo = Path.cwd()
    changed_files = 0
    for path in iter_py_files(repo):
        try:
            src = path.read_text(encoding="utf-8")
        except Exception:
            continue
        new = strip_extraneous_fprefix(src)
        if new != src:
            path.write_text(new, encoding="utf-8")
            print(f"Fixed extraneous f-strings: {path}")
            changed_files += 1
    print(f"Total files changed: {changed_files}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

