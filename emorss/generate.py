import torch

from .config import MODELS
from .data import batches, load_pairs, write_jsonl
from .model import edit_prefill, encode, generate_batch, load_model
from .vectors import load_archived_risk_vectors, load_shared_vector


def vector_tensor(value, hidden_size):
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.shape != (hidden_size,) or not torch.isfinite(tensor).all():
        raise ValueError(f"Expected a finite vector with shape ({hidden_size},)")
    return tensor


def generate(args):
    config = MODELS[args.model]
    pairs = load_pairs(args.pairs)
    if args.archived_risk_vectors:
        vectors = load_archived_risk_vectors(args.archived_risk_vectors)
        groups = [([pair for pair in pairs if pair.risk == risk], vector) for risk, vector in vectors.items()]
    else:
        groups = [(pairs, load_shared_vector(args.vector, args.model))]

    model, tokenizer = load_model(
        args.checkpoint or config.checkpoint,
        args.revision or (None if args.checkpoint else config.revision),
        args.device,
        "sdpa",
    )
    if config.layer >= model.config.num_hidden_layers:
        raise ValueError(f"The selected checkpoint lacks layer {config.layer} for {args.model}")

    def generate_rows():
        for group, offset in groups:
            vector = vector_tensor(offset, model.config.hidden_size)
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
