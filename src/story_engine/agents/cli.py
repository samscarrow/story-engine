import argparse
from .loader import list_agents, get_agent_prompt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="story-engine-agents", description="Agent prompt utilities"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list")
    show = sub.add_parser("show")
    show.add_argument("agent", help="Agent key, e.g., engine_integrator")

    args = parser.parse_args(argv)
    if args.cmd == "list":
        for a in list_agents():
            print(a)
        return 0
    if args.cmd == "show":
        print(get_agent_prompt(args.agent))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
