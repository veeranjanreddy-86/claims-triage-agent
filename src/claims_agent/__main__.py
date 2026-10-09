"""CLI: python -m claims_agent triage CLM-1004 "What should we do?" """

from __future__ import annotations

import argparse
import logging

from claims_agent.agent import TriageAgent
from claims_agent.config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(prog="claims_agent")
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("triage", help="triage one claim")
    t.add_argument("claim_id")
    t.add_argument("question", nargs="?", default="Why was this denied and what should we do?")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    result = TriageAgent(Settings.from_env()).run(args.claim_id, args.question)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
