import numpy as np
import torch
import torch.nn.functional as F

from .data import batches
from .model import ablate_direction, encode, left_pad


@torch.inference_mode()
def last_hidden(model, tokenizer, sequences):
    ids, mask = left_pad(sequences, tokenizer.pad_token_id, model.device)
    states = model(
        input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False,
    ).hidden_states
    return torch.stack([state[:, -1] for state in states]).float().cpu().numpy()


def opener_ids(tokenizer, words):
    tokens = set()
    for word in words:
        for spelling in (word, " " + word):
            ids = tokenizer.encode(spelling, add_special_tokens=False)
            if len(ids) == 1:
                tokens.add(ids[0])
                break
    if not tokens:
        raise ValueError("No first-token margin openers found in this tokenizer")
    return sorted(tokens)


@torch.inference_mode()
def first_token_margin(model, tokenizer, sequences, refusal, answer):
    scores = []
    for batch in batches(sequences, 16):
        ids, mask = left_pad(batch, tokenizer.pad_token_id, model.device)
        result = model.generate(
            input_ids=ids, attention_mask=mask, max_new_tokens=1, do_sample=False,
            output_scores=True, return_dict_in_generate=True, pad_token_id=tokenizer.pad_token_id,
        )
        logits = result.scores[0].float()
        margin = logits[:, refusal].logsumexp(-1) - logits[:, answer].logsumexp(-1)
        scores.extend(margin.cpu().tolist())
    return np.array(scores)


def fit_mda(model, tokenizer, backbone, anchors):
    neutral = [encode(tokenizer, pair.neutral_prompt, backbone) for pair in anchors]
    risk = np.array([pair.risk for pair in anchors])
    layers = model.config.num_hidden_layers
    sums = {label: np.zeros((layers + 1, model.config.hidden_size)) for label in ("low", "high")}
    for offset in range(0, len(neutral), 16):
        states = last_hidden(model, tokenizer, neutral[offset:offset + 16])
        for index in range(states.shape[1]):
            sums[risk[offset + index]] += states[:, index]
    counts = {label: int((risk == label).sum()) for label in sums}
    if not all(counts.values()):
        raise ValueError("MDA anchors require benign and harmful requests")
    directions = sums["high"] / counts["high"] - sums["low"] / counts["low"]
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.where(norms > 0, norms, 1)

    indices = np.r_[np.flatnonzero(risk == "low")[:32], np.flatnonzero(risk == "high")[:32]]
    validation = [neutral[index] for index in indices]
    benign = risk[indices] == "low"
    refusal = opener_ids(tokenizer, ("Sorry", "Unfortunately", "Cannot"))
    answer = opener_ids(tokenizer, ("Sure", "Certainly", "Absolutely"))
    candidates = []
    for layer in range(max(2, layers // 6), layers, 2):
        direction = torch.as_tensor(directions[layer], dtype=model.dtype, device=model.device)
        with ablate_direction(model, direction):
            margin = first_token_margin(model, tokenizer, validation, refusal, answer)
        candidates.append((margin[~benign].mean() < 0, margin[benign].mean(), layer))
    _, _, selected = min(candidates)
    print(f"MDA selected hidden-state index {selected}", flush=True)
    return torch.as_tensor(directions[selected], dtype=model.dtype, device=model.device)


@torch.inference_mode()
def transition(model, sequences, sure_id):
    ids = torch.tensor([sequences + [sure_id]], device=model.device)
    states = model(
        input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True, use_cache=False,
    ).hidden_states
    return torch.stack([state[0, -2] - state[0, -1] for state in states[1:]]).float()


def fit_scans(model, tokenizer, backbone, config, anchors, evaluation):
    def mean_last(sequences):
        total = np.zeros((model.config.num_hidden_layers, model.config.hidden_size))
        for batch in batches(sequences, 16):
            total += last_hidden(model, tokenizer, batch)[1:].sum(axis=1)
        return total / len(sequences)

    neutral = {
        risk: [encode(tokenizer, pair.neutral_prompt, backbone) for pair in anchors if pair.risk == risk]
        for risk in ("low", "high")
    }
    if not all(neutral.values()):
        raise ValueError("SCANS anchors require benign and harmful requests")
    directions = mean_last(neutral["high"]) - mean_last(neutral["low"])
    directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-8

    start, stop = config.scans_layers
    sure_id = tokenizer.encode(" Sure", add_special_tokens=False)[0]
    transitions = [transition(model, ids, sure_id)[start:stop].cpu().numpy() for ids in neutral["high"]]
    unsafe = torch.as_tensor(np.mean(transitions, axis=0), device=model.device)
    groups = {1: [], -1: []}
    for pair in evaluation:
        ids = encode(tokenizer, pair.high_prompt, backbone)
        query = transition(model, ids, sure_id)[start:stop]
        similarities = F.cosine_similarity(query, unsafe).nan_to_num().tolist()
        cosine = sum(similarities) / len(similarities)
        sign = 1 if cosine >= config.scans_threshold else -1
        groups[sign].append(pair)

    return [
        (pairs, {
            layer: sign * config.scans_multiplier * torch.as_tensor(
                directions[layer], device=model.device, dtype=model.dtype,
            )
            for layer in range(start, stop)
        })
        for sign, pairs in groups.items() if pairs
    ]
