import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

from story_engine.poml.lib.poml_integration import create_engine


def _load_data(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() in {".yml", ".yaml"}:
        if yaml is None:
            raise RuntimeError("PyYAML required to read YAML files")
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return json.loads(p.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="story-engine-poml", description="Render POML templates")
    parser.add_argument("template", help="Template path relative to poml templates root, e.g. simulations/character_response.poml")
    parser.add_argument("--data", help="JSON or YAML input file")
    parser.add_argument("--format", default="openai_chat", choices=["openai_chat", "text"], help="Render format")
    parser.add_argument("--roles", action="store_true", help="Output system and user roles as JSON")
    parser.add_argument("--write-golden", help="Write output to golden file path")
    parser.add_argument("--check-golden", help="Compare output to golden file path")
    parser.add_argument("--strict", action="store_true", help="Enable strict mode for unresolved placeholders/tags")

    args = parser.parse_args(argv)

    ctx: Dict[str, Any] = {}
    if args.data:
        ctx = _load_data(args.data)

    engine = create_engine()
    if args.strict:
        engine.config.strict_mode = True

    if args.roles or args.format == "openai_chat":
        result = engine.render_roles(args.template, ctx)
        output = json.dumps(result, ensure_ascii=False, indent=2)
    else:
        output = engine.render(args.template, ctx, format=args.format)

    if args.write_golden:
        Path(args.write_golden).write_text(output, encoding="utf-8")
        print(f"Wrote golden: {args.write_golden}")
        return 0

    if args.check_golden:
        golden = Path(args.check_golden)
        if not golden.exists():
            print(f"Golden missing: {args.check_golden}", file=sys.stderr)
            return 2
        expected = golden.read_text(encoding="utf-8")
        if expected != output:
            print("Output differs from golden", file=sys.stderr)
            return 1
        print("Golden match")
        return 0

    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

