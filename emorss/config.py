from dataclasses import dataclass


@dataclass(frozen=True)
class Backbone:
    label: str
    checkpoint: str
    revision: str
    layer: int
    strength: int
    scans_layers: tuple[int, int]
    scans_threshold: float
    scans_multiplier: float


MODELS = {
    "llama": Backbone(
        "Llama-3.1-8B-it", "NousResearch/Meta-Llama-3.1-8B-Instruct",
        "d10aef7999a2b5ba950ab3974312feeedbfe0b77", 22, 20, (10, 20), 0.83, 0.5,
    ),
    "qwen": Backbone(
        "Qwen3-8B", "Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218",
        25, 10, (11, 23), 0.96, 1.0,
    ),
}

METHODS = {
    "default": "Default",
    "self-cd": "Self-CD",
    "mda": "MDA",
    "sva": "SVA",
    "scans": "SCANS",
    "adacd": "AdaCD",
    "prompt": "Prompt",
    "emorss": "Ours",
}

# Table 3 gives each source equal weight within its risk group.
CELLS = {
    ("high", "jailbreakbench"): 32,
    ("high", "xstest"): 32,
    ("high", "harmbench"): 64,
    ("low", "jailbreakbench"): 32,
    ("low", "xstest"): 32,
    ("low", "or_bench"): 64,
}

GUARDIAN = "ibm-granite/granite-guardian-3.1-2b"
