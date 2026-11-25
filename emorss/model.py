from contextlib import contextmanager

import torch


def load_model(checkpoint, revision, device, attention=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    on_cuda = torch.device(device).type == "cuda"
    dtype = torch.bfloat16 if on_cuda else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, revision=revision, dtype=dtype, device_map={"": device} if on_cuda else None,
        attn_implementation=attention,
    )
    if not on_cuda:
        model.to(device)
    return model.eval(), tokenizer


def encode(tokenizer, prompt, backbone, system=None):
    messages = [{"role": "user", "content": prompt}]
    if system is not None:
        messages.insert(0, {"role": "system", "content": system})
    options = {"enable_thinking": False} if backbone == "qwen" else {}
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=False, **options,
    )


def left_pad(sequences, pad_id, device):
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for index, sequence in enumerate(sequences):
        ids[index, -len(sequence):] = torch.tensor(sequence, dtype=torch.long)
        mask[index, -len(sequence):] = 1
    return ids.to(device), mask.to(device)


def decode(tokenizer, ids):
    if tokenizer.eos_token_id in ids:
        ids = ids[:ids.index(tokenizer.eos_token_id)]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


@torch.inference_mode()
def generate_batch(model, tokenizer, sequences, max_new_tokens):
    ids, mask = left_pad(sequences, tokenizer.pad_token_id, model.device)
    output = model.generate(
        input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens, do_sample=False,
        temperature=None, top_p=None, top_k=None, pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return [decode(tokenizer, tokens) for tokens in output[:, ids.shape[1]:].tolist()]


def unit(vector):
    norm = torch.linalg.vector_norm(vector)
    if not torch.isfinite(norm) or norm <= 0:
        raise ValueError("Expected a finite, nonzero direction")
    return vector / norm


def map_hidden(output, edit):
    if isinstance(output, tuple):
        return (edit(output[0]), *output[1:])
    return edit(output)


@contextmanager
def edit_prefill(model, layer, direction, strength):
    length = float(direction.norm())
    if length == 0:
        yield
        return
    direction = unit(direction)
    edited = False

    def hook(_module, _inputs, output):
        nonlocal edited
        if edited:
            return output
        edited = True

        def add_delta(hidden):
            result = hidden.clone()
            result[:, -1, :] += (strength * length) * direction.to(hidden)
            return result

        return map_hidden(output, add_delta)

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def ablate_direction(model, direction, *, sublayers=False):
    def project(hidden):
        vector = direction.to(hidden)
        return hidden - (hidden @ vector).unsqueeze(-1) * vector

    def output_hook(_module, _inputs, output):
        return map_hidden(output, project)

    def block_input(_module, inputs):
        return (project(inputs[0]), *inputs[1:])

    handles = []
    try:
        if sublayers:
            for layer in model.model.layers:
                handles.append(layer.register_forward_pre_hook(block_input))
                handles.append(layer.self_attn.register_forward_hook(output_hook))
                handles.append(layer.mlp.register_forward_hook(output_hook))
        else:
            for module in [model.model.embed_tokens, *model.model.layers]:
                handles.append(module.register_forward_hook(output_hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def add_layer_vectors(model, vectors):
    def add(vector):
        def hook(_module, _inputs, output):
            return map_hidden(output, lambda hidden: hidden + vector.to(hidden))
        return hook

    handles = []
    try:
        for layer, vector in vectors.items():
            handles.append(model.model.layers[layer].register_forward_hook(add(vector)))
        yield
    finally:
        for handle in handles:
            handle.remove()
