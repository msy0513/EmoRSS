from dataclasses import dataclass


@dataclass(frozen=True)
class Backbone:
    checkpoint: str
    revision: str
    layer: int
    strength: int


MODELS = {
    "llama": Backbone(
        "NousResearch/Meta-Llama-3.1-8B-Instruct", "d10aef7999a2b5ba950ab3974312feeedbfe0b77",
        layer=22, strength=20,
    ),
    "qwen": Backbone(
        "Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218", layer=25, strength=10,
    ),
}

GUARDIAN = "ibm-granite/granite-guardian-3.1-2b"
