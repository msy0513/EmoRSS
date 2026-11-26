import torch

from .config import MODELS
from .data import batches, load_pairs, write_jsonl
from .model import edit_prefill, encode, generate_batch, load_model
from .vectors import steering_vectors


def direction_tensor(value, model):
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.shape != (model.config.hidden_size,) or not torch.isfinite(tensor).all():
        raise ValueError(f"Expected a finite vector with shape ({model.config.hidden_size},)")
    return tensor


def generate(args):
    config = MODELS[args.model]
    pairs = load_pairs(args.pairs)
    vectors = steering_vectors(args.vector, args.model)
    model, tokenizer = load_model(
        args.checkpoint or config.checkpoint, args.revision or (None if args.checkpoint else config.revision),
        args.device, "sdpa",
    )

    def generate_rows():
        for risk in ("low", "high"):
            group = [pair for pair in pairs if pair.risk == risk]
            vector = direction_tensor(vectors[risk], model)
            for batch in batches(group, args.batch_size):
                sequences = [encode(tokenizer, pair.high_prompt, args.model) for pair in batch]
                with edit_prefill(model, config.layer, vector, config.strength):
                    responses = generate_batch(model, tokenizer, sequences, args.max_new_tokens)
                for pair, response in zip(batch, responses):
                    yield {
                        "core_id": pair.core_id, "source": pair.source, "risk": pair.risk,
                        "model": args.model, "method": "emorss", "condition": "distress", "response_text": response,
                    }

    write_jsonl(args.out, generate_rows())
    print(f"Wrote {len(pairs)} EmoRSS responses for {args.model} to {args.out}")
