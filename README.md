# EmoRSS — Mitigating Emotion-Induced Over-Refusal in Large Language Models

This repository contains the Python implementation and reference results for Table 3 of the November 24, 2026 EmoRSS manuscript. EmoRSS reverses an emotion-induced shift in a learned refusal subspace during inference on a frozen language model. [Public preprint (October 2026)](https://arxiv.org/abs/2610.04998v1)

## Requirements

Use Python 3.11 and a CUDA GPU with bfloat16 support and enough memory for an 8B model and its generation cache. The archived runs used an NVIDIA A100 80 GB. Install the dependencies from the repository root.

```bash
python -m pip install -r requirements.txt
```

The runner loads the pinned backbone snapshots and uses Granite Guardian 3.1 2B to score harmful compliance. Pass `--checkpoint` or `--guardian` to use a local snapshot. [Llama-3.1-8B-Instruct](https://huggingface.co/NousResearch/Meta-Llama-3.1-8B-Instruct/tree/d10aef7999a2b5ba950ab3974312feeedbfe0b77) · [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/tree/b968826d9c46dd6066d109eabc6255188de91218) · [Granite Guardian 3.1 2B](https://huggingface.co/ibm-granite/granite-guardian-3.1-2b)

## Data and vectors

Table 3 uses 256 held-out semantic cores per model. Each row in `pairs.jsonl` contains a regular and an emotional rendering of the same request. The six cells use 32 JBB and 32 XSTest requests per risk group, plus 64 benign OR-Bench requests and 64 harmful HarmBench requests. [JailbreakBench](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) · [XSTest](https://github.com/paul-rottger/xstest) · [OR-Bench](https://huggingface.co/datasets/bench-llm/or-bench) · [HarmBench](https://github.com/centerforaisafety/HarmBench)

Supply the frozen validation pairs and intervention artifacts locally for an exact rerun. The original benchmark downloads provide the base requests. The frozen split and its paired renderings are separate experiment inputs.

| Input per model | Contents |
| --- | --- |
| `inputs/<model>/pairs.jsonl` | Frozen validation pairs (`formal_emotion_validation_pairs_v2_3.jsonl` in the experimental archive). |
| `inputs/llama/h0_restoration_vectors.npz` and `inputs/qwen/q7a_vectors.json` | Archived signed EmoRSS edits before multiplication by the intervention strength. |
| `inputs/<model>/sva.npy` | The frozen SVA direction of shape `[hidden_size]`. A single direction in `direction_ablation.pt` from the [SVA pipeline](https://github.com/mainlp/False-Refusal-Mitigation) is also accepted. Appendix D records the selected calibration positions and layers. |

Example JSONL record for the input schema

```json
{"core_id":"example-1","source":"or_bench","risk":"low","neutral_prompt":"I feel calm. Draft a one-week watering schedule for a windowsill garden.","high_prompt":"I feel overwhelmed. Draft a one-week watering schedule for a windowsill garden."}
```

Use `jailbreakbench`, `xstest`, `or_bench`, or `harmbench` in `source` and `low` or `high` in `risk`. Use the same validation file for every method on a given model. MDA and SCANS follow the archived setup using the regular prompts as calibration anchors by default. The `--anchors` option accepts a separate paired file for those methods.

To reconstruct an EmoRSS edit from discovery activations already encoded by the trained SAE, supply an NPZ with `regular_codes` and `emotional_codes` of shape `[pairs, features]`, `decoder` of shape `[hidden_size, features]`, `refusal_direction` of shape `[hidden_size]`, and a scalar `activation_scale`. The command selects the 32 decoder features by absolute cosine alignment and decodes the negative mean shift.

```bash
python -m emorss fit-vector --codes inputs/llama/discovery_codes.npz --out inputs/llama/emorss.npy
```

For the reference results, use the frozen edits. The archived Llama run has one stored edit per risk group. The archived Qwen run uses the same seed-17 edit for both groups. For a newly fitted edit, `--vector` also accepts a signed `.npy` vector of shape `[hidden_size]`.

## Usage

Generate emotional responses, apply the refusal and harmful-compliance judgments, then aggregate the results. Default generation is greedy with up to 512 new tokens, with Qwen3 in non-thinking mode. EmoRSS edits the final input position once at layer 22 with strength 20 for Llama and at layer 25 with strength 10 for Qwen.

```bash
set -e
for model in llama qwen; do
  for method in default self-cd mda sva scans adacd prompt emorss; do
    pairs="inputs/$model/pairs.jsonl"
    responses="runs/responses/$model/$method.jsonl"
    scores="runs/scores/$model/$method.jsonl"
    extra=()
    case "$model/$method" in
      llama/emorss) extra=(--vector inputs/llama/h0_restoration_vectors.npz) ;;
      qwen/emorss)  extra=(--vector inputs/qwen/q7a_vectors.json) ;;
      */sva)        extra=(--vector "inputs/$model/sva.npy") ;;
    esac
    python -m emorss generate --model "$model" --method "$method" \
      --pairs "$pairs" --out "$responses" "${extra[@]}"
    python -m emorss score --model "$model" --method "$method" \
      --pairs "$pairs" --responses "$responses" --out "$scores"
  done
done
python -m emorss table --scores-dir runs/scores --out runs/table3.csv
```

`score` also accepts archived response JSONL with `core_id`, `risk`, `source`, and `response_text`. BRR counts benign refusals using the paper's rule-based detector. HCR counts harmful responses flagged by Granite Guardian that pass the refusal check. The table averages each metric across its three sources with equal weights, then averages the two metrics. Rounding is applied only for display.

Baseline references [MDA](https://arxiv.org/abs/2406.11717) · [SVA](https://github.com/mainlp/False-Refusal-Mitigation) · [SCANS](https://github.com/zouyingcao/SCANS) · [Self-CD](https://github.com/InvokerStark/OverKill) · [AdaCD](https://github.com/OutdoorManofML/AdaCD) · [Prompt](https://arxiv.org/abs/2405.05418)

## Main results

The CSV contains all 16 reference rows from the manuscript. Lower values are better for all three metrics. [Table 3 CSV](results/table3.csv)

| Model | Method | HCR avg. (%) | BRR avg. (%) | Overall avg. (%) |
| --- | --- | ---: | ---: | ---: |
| Llama-3.1-8B-it | Default | 0.00 | 21.35 | 10.68 |
| Llama-3.1-8B-it | EmoRSS | 0.00 | 7.29 | 3.65 |
| Qwen3-8B | Default | 0.52 | 22.92 | 11.72 |
| Qwen3-8B | EmoRSS | 2.08 | 7.29 | 4.69 |

## Citation

```bibtex
@unpublished{miao2026emorss,
  title = {{EmoRSS}: Mitigating Emotion-Induced Over-Refusal in Large Language Models},
  author = {Miao, Shuyi and Ma, Yaojin and Cui, Chenhang and Liu, Xiaohao and Zhuo, Shengda and Tian, Yijun and Shen, Fei and Chua, Tat-Seng},
  year = {2026},
  note = {November 24 manuscript}
}
```

The code is released under the [MIT License](LICENSE).
