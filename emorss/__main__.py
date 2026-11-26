import argparse
from pathlib import Path

from .config import GUARDIAN, MODELS


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def add_run_options(parser):
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--pairs", type=Path, required=True, help="Frozen validation pairs JSONL")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)


def main():
    parser = argparse.ArgumentParser(description="Run EmoRSS steering and evaluation")
    subcommands = parser.add_subparsers(dest="command", required=True)

    generate = subcommands.add_parser("generate", help="Generate responses with EmoRSS steering")
    add_run_options(generate)
    generate.add_argument("--checkpoint", help="Model ID or local snapshot directory")
    generate.add_argument("--revision", help="Checkpoint revision override")
    steering = generate.add_mutually_exclusive_group(required=True)
    steering.add_argument("--vector", type=Path, help="Shared signed .npy vector or archived Qwen JSON")
    steering.add_argument(
        "--archived-risk-vectors",
        type=Path,
        help="Llama archive with offsets selected by the validation risk label",
    )
    generate.add_argument("--batch-size", type=positive_int, default=8)
    generate.add_argument("--max-new-tokens", type=positive_int, default=512)

    score = subcommands.add_parser("score", help="Score a response JSONL with the refusal rule and Granite Guardian")
    add_run_options(score)
    score.add_argument("--responses", type=Path, required=True)
    score.add_argument("--guardian", default=GUARDIAN, help="Granite Guardian model ID or local snapshot directory")
    score.add_argument("--guardian-revision", help="Granite Guardian checkpoint revision")
    score.add_argument("--batch-size", type=positive_int, default=16)

    fit = subcommands.add_parser("fit-vector", help="Decode the reverse edit from paired discovery SAE codes")
    fit.add_argument("--codes", type=Path, required=True, help="NPZ of discovery codes and SAE decoder")
    fit.add_argument("--out", type=Path, required=True, help="Signed reverse edit in NumPy .npy format")

    args = parser.parse_args()
    if args.command == "generate" and args.archived_risk_vectors and args.model != "llama":
        generate.error("--archived-risk-vectors is available for the Llama archive")
    try:
        if args.command == "generate":
            from .generate import generate as run
        elif args.command == "score":
            from .score import score as run
        else:
            from .vectors import fit_vector as run
        run(args)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
