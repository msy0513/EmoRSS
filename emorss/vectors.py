import json

import numpy as np


def validate_vector(value):
    vector = np.asarray(value)
    if vector.ndim != 1 or not vector.size or vector.dtype.kind not in "fiu" or not np.isfinite(vector).all():
        raise ValueError("Expected a nonempty, finite one-dimensional steering vector")
    return vector


def load_shared_vector(path, backbone):
    """Read one signed edit for all evaluation prompts."""
    if path.suffix == ".npy":
        return validate_vector(np.load(path, allow_pickle=False))
    if path.suffix == ".json" and backbone == "qwen":
        with path.open(encoding="utf-8") as handle:
            return validate_vector(json.load(handle)["seed17"]["v_cancel"])
    if path.suffix == ".npz" and backbone == "llama":
        raise ValueError("Use --archived-risk-vectors to replay the Llama archive")
    raise ValueError("Provide a signed .npy vector or the archived Qwen vector JSON")


def load_archived_risk_vectors(path):
    """Read the two label-conditioned offsets from the Llama archive."""
    if path.suffix != ".npz":
        raise ValueError("The archived risk vectors must be provided as a .npz file")
    with np.load(path, allow_pickle=False) as archive:
        return {risk: validate_vector(archive[f"{risk}__k32_restore"]) for risk in ("low", "high")}


def decode_shift(regular, emotional, decoder, direction, scale):
    """Estimate the signed reverse edit from paired discovery SAE codes."""
    if regular.ndim != 2 or emotional.shape != regular.shape or regular.shape[0] == 0:
        raise ValueError("Paired codes must have matching nonempty [pairs, features] shapes")
    if decoder.ndim != 2 or decoder.shape[1] != regular.shape[1] or decoder.shape[1] < 32:
        raise ValueError("Decoder must have shape [hidden_size, features] with at least 32 features")
    if direction.shape != (decoder.shape[0],):
        raise ValueError("Refusal direction must have shape [hidden_size]")
    if not all(np.isfinite(value).all() for value in (regular, emotional, decoder, direction, scale)):
        raise ValueError("Codes, decoder, direction, and activation scale must be finite")
    if scale <= 0 or np.linalg.norm(direction) == 0:
        raise ValueError("Expected a positive activation scale and a nonzero refusal direction")

    decoder = decoder.astype(np.float64, copy=False)
    norms = np.linalg.norm(decoder, axis=0)
    if np.count_nonzero(norms) < 32:
        raise ValueError("The decoder must contain at least 32 nonzero features")
    # The norm of the refusal direction is common to all cosine scores.
    alignment = np.divide(
        np.abs(np.einsum("i,ij->j", direction, decoder)),
        norms, out=np.full_like(norms, -np.inf), where=norms > 0,
    )
    selected = np.argsort(-alignment, kind="stable")[:32]
    change = (emotional[:, selected].astype(np.float64) - regular[:, selected]).mean(axis=0)
    decoded = np.einsum("ij,j->i", decoder[:, selected], change)
    return (-scale * decoded).astype(np.float32)


def fit_vector(args):
    if args.out.suffix != ".npy":
        raise ValueError("The fitted vector output must end in .npy")
    with np.load(args.codes, allow_pickle=False) as inputs:
        delta = decode_shift(
            inputs["regular_codes"], inputs["emotional_codes"], inputs["decoder"],
            inputs["refusal_direction"], float(inputs["activation_scale"].item()),
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, delta, allow_pickle=False)
    print(f"Wrote signed reverse edit to {args.out}")
