import argparse
from pathlib import Path

from .config import GUARDIAN, METHODS, MODELS


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def add_run_options(parser):
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--pairs", type=Path, required=True, help="Frozen validation pairs JSONL")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)


def main():
    parser = argparse.ArgumentParser(description="Run the EmoRSS Table 3 evaluation")
    subcommands = parser.add_subparsers(dest="command", required=True)

    generate = subcommands.add_parser("generate", help="Generate one model/method's emotional responses")
    add_run_options(generate)
    generate.add_argument("--checkpoint", help="Model ID or local snapshot directory")
    generate.add_argument("--revision", help="Checkpoint revision override")
    generate.add_argument("--anchors", type=Path, help="Optional MDA/SCANS calibration pairs JSONL")
    generate.add_argument("--vector", type=Path, help="Required signed EmoRSS edit or SVA direction")
    generate.add_argument("--batch-size", type=positive_int)
    generate.add_argument("--max-new-tokens", type=positive_int, default=512)

    score = subcommands.add_parser("score", help="Score a response JSONL with the refusal rule and Granite Guardian")
    add_run_options(score)
    score.add_argument("--responses", type=Path, required=True)
    score.add_argument("--guardian", default=GUARDIAN, help="Granite Guardian model ID or local snapshot directory")
    score.add_argument("--guardian-revision", help="Granite Guardian checkpoint revision")
    score.add_argument("--batch-size", type=positive_int, default=16)

    table = subcommands.add_parser("table", help="Aggregate complete per-source judgments into a CSV")
    table.add_argument("--scores-dir", type=Path, required=True)
    table.add_argument("--model", choices=MODELS, help="Write one model block")
    table.add_argument("--out", type=Path, required=True)

    fit = subcommands.add_parser("fit-vector", help="Decode the reverse edit from paired discovery SAE codes")
    fit.add_argument("--codes", type=Path, required=True, help="NPZ of discovery codes and SAE decoder")
    fit.add_argument("--out", type=Path, required=True, help="Signed reverse edit in NumPy .npy format")

    args = parser.parse_args()
    try:
        if args.command == "generate":
            needs_vector = args.method in {"emorss", "sva"}
            if needs_vector and args.vector is None:
                generate.error(f"--method {args.method} requires --vector")
            if args.vector and not needs_vector:
                generate.error("--vector is only used by emorss and sva")
            if args.anchors and args.method not in {"mda", "scans"}:
                generate.error("--anchors is only used by mda and scans")
            from .generate import generate as run
        elif args.command == "score":
            from .score import score as run
        elif args.command == "fit-vector":
            from .vectors import fit_vector as run
        else:
            from .table import table as run
        run(args)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
