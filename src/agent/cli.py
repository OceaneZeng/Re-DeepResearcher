from __future__ import annotations

import argparse

from .react_agent import build_react_agent


def main() -> None:
    ap = argparse.ArgumentParser(description="Modular LangChain ReAct agent (CLI).")
    ap.add_argument("--quiet", action="store_true", help="Disable verbose tool traces.")
    args = ap.parse_args()

    executor = build_react_agent(verbose=not args.quiet)
    print("ReAct agent ready. Type a question, or 'exit'.")
    while True:
        q = input("> ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break
        r = executor.invoke({"input": q})
        print(r["output"])


if __name__ == "__main__":
    main()

