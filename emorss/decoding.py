import torch

from .model import decode, encode, left_pad


SAFETY_SYSTEM = (
    "You are a helpful, respectful and honest assistant. Always answer as "
    "helpfully as possible, while being safe. Your answers should not include "
    "any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content."
)
REFUSAL_SYSTEM = "Please refuse to answer me!"


def position_ids(mask):
    return (mask.cumsum(-1) - 1).clamp_min(0)


def adacd_logits(instruction, contrast):
    ratio, agreement_threshold, plausibility = 4.5, 0.9, 0.01
    probabilities = instruction.softmax(-1)
    contrast_probabilities = contrast.softmax(-1)
    top = contrast.argmax(-1, keepdim=True)
    ranking = instruction.argsort(dim=-1, descending=True)
    rank = (ranking == top).float().argmax(-1)
    agreement = 1 / (rank + 1)
    confidence = probabilities.amax(-1)
    contrast_confidence = contrast_probabilities.amax(-1)
    reinforce = (agreement >= agreement_threshold) & (confidence >= agreement_threshold * contrast_confidence)
    shift = ratio * (contrast - instruction)
    adjusted = torch.where(reinforce[:, None], contrast + shift, contrast - shift)
    return adjusted.masked_fill(probabilities < plausibility * confidence[:, None], float("-inf"))


@torch.inference_mode()
def contrastive_batch(model, tokenizer, backbone, method, prompts, max_tokens):
    self_cd = method == "self-cd"
    main_system, contrast_system = (SAFETY_SYSTEM, None) if self_cd else (None, REFUSAL_SYSTEM)
    contrast_steps = max_tokens if self_cd else 10
    pad_id = tokenizer.pad_token_id

    def prefill(system):
        sequences = [encode(tokenizer, prompt, backbone, system) for prompt in prompts]
        ids, mask = left_pad(sequences, pad_id, model.device)
        positions = position_ids(mask) if self_cd else None
        result = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=True)
        return result.logits[:, -1], result.past_key_values, mask

    def advance(token, cache, mask):
        mask = torch.cat((mask, torch.ones_like(token)), dim=1)
        result = model(
            input_ids=token, attention_mask=mask, position_ids=position_ids(mask)[:, -1:],
            past_key_values=cache, use_cache=True,
        )
        return result.logits[:, -1], result.past_key_values, mask

    main, main_cache, main_mask = prefill(main_system)
    contrast, contrast_cache, contrast_mask = prefill(contrast_system)
    eos_ids = {tokenizer.eos_token_id}
    if self_cd:
        vocab = tokenizer.get_vocab()
        eos_ids.update(vocab[token] for token in ("<|im_end|>", "<|eot_id|>") if token in vocab)
    eos = torch.tensor(sorted(eos_ids - {None}), dtype=torch.long, device=model.device)
    finished = torch.zeros(len(prompts), dtype=torch.bool, device=model.device)
    continuations = [[] for _ in prompts]

    for step in range(max_tokens):
        if self_cd:
            logits = main.float() - 2.5 * (main.float() - contrast.float())
        elif step < contrast_steps:
            logits = adacd_logits(main, contrast)
        else:
            logits = main
        token = logits.argmax(-1).masked_fill(finished, pad_id)
        for index, (token_id, done) in enumerate(zip(token.tolist(), finished.tolist())):
            if not done:
                continuations[index].append(token_id)
        finished |= torch.isin(token, eos)
        if finished.all() or step + 1 == max_tokens:
            break
        main, main_cache, main_mask = advance(token[:, None], main_cache, main_mask)
        if step + 1 < contrast_steps:
            contrast, contrast_cache, contrast_mask = advance(token[:, None], contrast_cache, contrast_mask)

    return [decode(tokenizer, ids) for ids in continuations]
