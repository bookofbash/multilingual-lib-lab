from __future__ import annotations

import argparse
import sys

from lidlab import __version__
from lidlab.datasets import load_challenge, load_seed, validate_examples
from lidlab.eval.runner import run_eval
from lidlab.models import MODEL_KINDS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lidlab", description="Multilingual language-identification lab")
    parser.add_argument("--version", action="version", version=f"lidlab {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    eval_parser = sub.add_parser("eval", help="Run models on the challenge set")
    eval_parser.add_argument(
        "--models",
        default="tfidf",
        help=f"Comma-separated models. Choices: {', '.join(MODEL_KINDS)}",
    )
    eval_parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N items")

    check = sub.add_parser("check-data", help="Validate seed and challenge JSONL")
    check.add_argument("--quiet", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "eval":
        names = [part.strip() for part in args.models.split(",") if part.strip()]
        result = run_eval(names, limit=args.limit)
        print(f"wrote {result.run_dir / 'report.md'}")
        for metrics in result.metrics:
            print(
                f"{metrics.model}: exact={metrics.exact:.3f} "
                f"codeswitch={metrics.codeswitch_hit:.3f} "
                f"family={metrics.family:.3f} "
                f"covered_exact={metrics.covered_exact:.3f} "
                f"({metrics.n_covered}/{metrics.n})"
            )
        return 0
    if args.command == "check-data":
        challenge = load_challenge()
        seed = load_seed()
        problems = validate_examples(challenge) + validate_examples(seed)
        if problems:
            for problem in problems:
                print(problem, file=sys.stderr)
            return 1
        if not args.quiet:
            print(f"ok: {len(challenge)} challenge, {len(seed)} seed")
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
