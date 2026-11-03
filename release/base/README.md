---
license: cc-by-nc-4.0
language:
- en
library_name: transformers
pipeline_tag: text-generation
tags:
- small-language-model
- pretrained
- from-scratch
- qwen3
datasets:
- karpathy/climbmix-400b-shuffle
- IFM/TxT360-v2
- HuggingFaceTB/smollm-corpus
- HuggingFaceTB/cosmopedia
- HuggingFaceTB/finemath
- nvidia/OpenMathInstruct-2
- microsoft/orca-math-word-problems-200k
- kash8080/Mantis-50M-arith
---

# Mantis-50M-Base

Mantis-50M-Base is a **49.3M parameter** base language model, pretrained from scratch. It is **#1 in the under-50M class** of the [Open SLM Leaderboard](https://huggingface.co/spaces/AxiomicLabs/Open_SLM_Leaderboard), with an Intelligence Index of **18.36** ([submission](https://huggingface.co/spaces/AxiomicLabs/Open_SLM_Leaderboard/discussions/287), listed from 1 Nov 2026). Axiomic Labs also ran it on their hidden arithmetic sets, and it passed.

- **Code:** [github.com/kash8080/mantis-50m](https://github.com/kash8080/mantis-50m)
- **GGUF (llama.cpp):** [kash8080/Mantis-50M-Base-GGUF](https://huggingface.co/kash8080/Mantis-50M-Base-GGUF)
- **Sibling model:** [kash8080/Mantis-50M](https://huggingface.co/kash8080/Mantis-50M) shares the same first 27,200 training steps

It is a plain base model: no instruction tuning, no chat template. It continues text.

**Synthetic data, in short.** The stable phase of training included 7% synthetic arithmetic from our own generator ("v1"). We removed it for the decay phase, which uses only public data. The model forgot most of what v1 taught during the decay: see [Synthetic arithmetic in training](#synthetic-arithmetic-in-training).

## Results

Zero-shot, with the leaderboard's tools: lm-eval 0.4.12 (`acc_norm`, fp32) for HellaSwag, ARC and PIQA, and the official ArithMark-3 script (default settings). The leaderboard lists the same scores. This is the final checkpoint of the run (step 34,000). It was fixed as the release before any final eval, not picked by score.

| Model | Params | HellaSwag | ARC-E | ARC-C | PIQA | ArithMark | II |
|---|---|---|---|---|---|---|---|
| **Mantis-50M-Base (under-50M #1)** | 49.32M | **34.24** | **44.61** | **26.37** | **65.89** | **35.30** | **18.36** |
| 50M-pro (previous under-50M #1) | 49.59M | 33.25 | 45.62 | 25.51 | 64.80 | 37.40 | 17.93 |
| gibc-v2-lm | 49.82M | 31.87 | 47.85 | 26.11 | 60.28 | 40.20 | 16.13 |

Mantis-50M-Base is ahead of 50M-pro on HellaSwag, ARC-C and PIQA, and behind on ARC-E and ArithMark. The lead is 0.43 of the index. PIQA noise alone is about ±0.6, so treat the two models as close.

ArithMark is mostly near chance. Multiplication (68.9%), time (87.3%), perimeter (60.0%) and area (53.8%) are the only topics well above it. Time needs no computing: in all 55 items the answer is the only multiple of 5, so a preference for round numbers scores well. Per-topic scores are in `evals/arithmark_results.json`.

The raw eval files are in [`evals/`](evals/): `summary.json` (scores, tool versions, ArithMark data and script hashes), `lm_eval_results.json` and the ArithMark results.

## How this model differs from Mantis-50M

| | Mantis-50M | Mantis-50M-Base |
|---|---|---|
| Steps 1 to 27,200 | Same checkpoint, with 7% v1 synthetic arithmetic | Same checkpoint, with 7% v1 synthetic arithmetic |
| Decay, steps 27,200 to 34,000 | 12% synthetic arithmetic from a second generator ("v2") | No synthetic data |
| ArithMark-3 | 85.70 | 35.30 |
| Intelligence Index | 30.05 | 18.36 |
| On the leaderboard | No, not submitted | Yes, #1 under 50M |

We trained the first decay, then checked it on arithmetic outside our generators. It scored 31 to 41% there against 85.7% on ArithMark-3, so that score overstates its arithmetic. This second decay removes the question. We decided to submit its final checkpoint before seeing its score. Because v1 is still in its training history, Axiomic Labs ran it on their hidden arithmetic sets before listing it. It passed.

## Synthetic arithmetic in training

**What was used.** v1 is a generator we wrote: grade-school stories, equations and worked answers, with 27 templates and 50 names. It was 7% of the stable phase, about 1.4B of the tokens read. It was designed with ArithMark in mind: we wanted the skill it tests. It was written from general English, never from benchmark items, and the decay has none of it. [`generator.py`](https://github.com/kash8080/mantis-50m/blob/main/src/slm50m/arith/generator.py) has the code, and the `v1-sample` config of [kash8080/Mantis-50M-arith](https://huggingface.co/datasets/kash8080/Mantis-50M-arith) has a sample of the exact text (99M tokens). This is the only synthetic data in this model.

**It did not raise ArithMark.** Before the decay the model scored 37.3%, about the same as 50M-pro's 37.4%. Addition and subtraction sat at chance on every milestone. We also compared two small models (13.6M params, gate 4 of our plan), trained the same way with v1 (7% stable, 12% decay) and without it:

| Eval | With v1 | Without v1 |
|---|---|---|
| ArithMark-3 | 33.6 | 36.3 |
| Addition / subtraction on ArithMark-3 | 18.8 / 23.1 | 23.4 / 23.9 |
| v1's own templates, 4 choices | 77.8 | 29.8 |

The ArithMark gap is within noise (about 2 points). These runs were much shorter than this model's, so they are a guide, not proof.

**The model forgot it during the decay.** Once v1 left the mix, the skill it taught fell quickly:

| Measure | Before the decay (step 27,000 to 27,200) | Step 27,250 | Final |
|---|---|---|---|
| v1 validation loss | 0.549 | 1.777 | 2.04 |
| v1's own templates, 4 choices (chance 25%) | 98.5 | 48.7 | 35.2 |
| GPT-3 arithmetic, exact match, mean of 10 tasks | 16.6 | not run | 0.6 |

Forgetting here means the v1 templates and v1's habit of giving the bare answer. It does not mean the model cannot add, and we cannot show that v1 left no trace in the weights. See the next section.

## Other math benchmarks

Zero-shot, lm-eval 0.4.12 prompts and metrics. "Before" is the checkpoint at step 27,200, the last one before the decay (not released).

| Benchmark (chance) | Items | Scoring | Before | Mantis-50M-Base |
|---|---|---|---|---|
| [GPT-3 arithmetic](https://huggingface.co/datasets/EleutherAI/arithmetic), mean of 10 tasks (0) | 20,000 | exact match of the greedy answer | 16.6 | **0.6** |
| 2-digit add / subtract | 2,000 each | | 51.9 / 23.8 | 0.7 / 2.5 |
| 3-digit add / subtract | 2,000 each | | 35.4 / 20.8 | 0.1 / 0.3 |
| 2-digit multiply | 2,000 | | 16.0 | 1.9 |
| [MathQA](https://huggingface.co/datasets/allenai/math_qa) (20) | 2,985 | 5 choices, `acc_norm` | 23.4 | 23.5 |
| [MMLU](https://huggingface.co/datasets/cais/mmlu) elementary mathematics (25) | 378 | 4 choices, letters, `acc` | 26.7 | 26.7 |
| MMLU elementary mathematics, answer text (25) | 378 | 4 choices, `acc_norm` | 27.0 | 24.1 |

GPT-3 arithmetic asks "Question: What is 98 plus 45? Answer:" and counts an answer only if the model writes it first and exactly. This model usually writes working first, as the public word-problem data does, so that score understates it. In a separate check of 100 random 2-digit additions with the same prompt, it gave the right sum first on 0, wrote it within 24 tokens on 62, and picked it on 68 when choosing between answer, ±1 and +2. This model's outputs include ones like "59 + 63 = 122" and wrong ones like "we need to subtract 71 from 55". Its arithmetic is weak but present. We did not run this check on the pre-decay checkpoint or on a model that never saw v1, so we cannot say how much of it comes from v1.

To rerun the benchmarks, use [`math_evals.py`](https://github.com/kash8080/mantis-50m/blob/main/src/slm50m/evaluation/math_evals.py) (`modal run modal_app.py::run_math_evals`).

## Held-out arithmetic

Grade-school word problems from [ASDiv](https://huggingface.co/datasets/EleutherAI/asdiv) and [SVAMP](https://huggingface.co/datasets/ChilleD/SVAMP), written by other people and scored the same way as ArithMark (4 choices, mean log-prob per answer token). Orca-Math, one of our training sources, contains most of both sets, so we kept only the 232 problems that share no 8-word span with it, even with numbers masked.

| Prompt / wrong choices | Before (27,200) | Mantis-50M-Base |
|---|---|---|
| "... The answer is N." / N±1, N±2 | 28.9 | 25.9 |
| "... The answer is N." / wrong-operation results | 12.5 | 23.7 |
| "... The answer is N apples." / N±1, N±2 | 27.2 | 29.5 |
| "... The answer is N apples." / wrong-operation results | 18.8 | 28.1 |
| "Question: ... Answer: N" / N±1, N±2 | 27.6 | 28.4 |
| "Question: ... Answer: N" / wrong-operation results | 8.6 | 8.2 |

Chance is 25%, standard error about 3 points. The model is near chance on word problems worded by other people. ArithMark-3's 35.30 and these numbers agree. Scores below chance mean the model prefers numbers that appear in the problem.

To rerun it, use [`heldout_arith.py`](https://github.com/kash8080/mantis-50m/blob/main/src/slm50m/evaluation/heldout_arith.py) (`uv run slm-heldout-arith`).

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("kash8080/Mantis-50M-Base")
model = AutoModelForCausalLM.from_pretrained("kash8080/Mantis-50M-Base")

x = tok("Photosynthesis is the process by which", return_tensors="pt")
out = model.generate(**x, max_new_tokens=48, do_sample=False, repetition_penalty=1.1)
print(tok.decode(out[0], skip_special_tokens=True))
```

Tested with transformers 4.57.6 and 5.20.2 (same logits). The model is a standard `Qwen3ForCausalLM`, so no `trust_remote_code` is needed.

Greedy samples (repetition penalty 1.1), prompt in bold, cut at a sentence end and line breaks removed:

> **Photosynthesis is the process by which** plants convert light energy into chemical energy.

> **Once upon a time, there was a little robot who** lived in a big city called New York. He loved to play with his friends and help them learn new things.

> **Question: Sara has 14 apples and gives away 5. How many apples does she have left?\nAnswer:** Sara starts with 14 apples. She gives away 5 apples, so she is left with 14 - 5 = 9 apples.

## Model

| | |
|---|---|
| Architecture | Dense Qwen3-style decoder (`Qwen3ForCausalLM`) |
| Shape | 25 layers, hidden 384, 6 query heads and 2 KV heads (head dim 64), SwiGLU 1,152 |
| Details | QK-norm, RoPE theta 100,000, RMSNorm, tied input and output embeddings, no biases |
| Parameters | 49,322,240 (tied embedding counted once) |
| Context | 1,024 tokens |
| Tokenizer | Own byte-level BPE, 16,384 tokens, one token per digit. `<|endoftext|>` (id 0) ends documents. No BOS. Its training sample included v1 text at 7% |
| Weights | fp32 safetensors, 197 MB |

## Training

| | |
|---|---|
| Tokens | 8.9B (34,000 steps x 262,144 tokens) |
| Hardware | One H100 on Modal, about 4.4 hours |
| Optimizer | Muon for the block matrices, AdamW for embeddings and norms |
| Schedule | Warmup-stable-decay: 1% warmup, flat to 80%, 1-sqrt decay to 0 |
| Token-Superposition Training | Bags of 6 tokens for the first 30% of steps, then normal next-token training |
| Precision | bf16 autocast, fp32 master weights |
| Cost | About $33 of Modal credits for everything: data prep, tests, ablations, this run, evals, and the first decay (about $4), which is not in this model |

## Data

| Source | Licence | Stable mix | Decay mix |
|---|---|---|---|
| [ClimbMix](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle) | MIT on the reshuffle, CC-BY-NC-4.0 upstream | 45% | 22.7% |
| [TxT360-v2](https://huggingface.co/datasets/IFM/TxT360-v2) web | CC-BY-4.0 | 12% | 6.8% |
| TxT360-v2 QA (`txt360-qa`, `web-high-nltk-qa`) | CC-BY-4.0 | 8% | 15.9% |
| [FineWeb-Edu dedup](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | ODC-BY | 10% | 15.9% |
| [Cosmopedia v2](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | ODC-BY | 5% | 10.2% |
| [Cosmopedia WikiHow](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) | Apache-2.0 | 2% | 10.2% |
| [FineMath-4plus](https://huggingface.co/datasets/HuggingFaceTB/finemath) | ODC-BY | 7% | 9.1% |
| Word problems: [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) (GSM8K-type rows only) and [Orca-Math](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k) | CC-BY-4.0, MIT | 4% | 9.1% |
| Synthetic arithmetic v1, our generator ([sample](https://huggingface.co/datasets/kash8080/Mantis-50M-arith)) | CC-BY-4.0 | 7% | 0% |

Every source is pinned to a Hub commit in the code. Every batch holds every source in its mix share. The decay config is [`configs/base.json`](https://github.com/kash8080/mantis-50m/blob/main/configs/base.json), where the listed weights sum to 0.88 and the loader scales them to 1.

<details>
<summary>Samples of the v1 synthetic data (unedited)</summary>

```
Homework check

7 - 2 = 5

There were 33 tomatoes in a jar, and 23 were removed. How many tomatoes remained in the jar?
Step 1: 33 - 23 = 10. The jar was left with 10 tomatoes.

18 × 13 = 234
```

</details>

## Decontamination

- Every training document that shares a 13-word n-gram with HellaSwag, ARC-Easy, ARC-Challenge, PIQA or ArithMark-3 was dropped before training. Per-source numbers are in the [filter report](https://github.com/kash8080/mantis-50m/blob/main/docs/filter-report.md)
- The v1 generator has zero exact matches and zero 13-gram matches with ArithMark-3, tested over ~54,000 generated problems in every format
- Benchmark items under 13 words have no 13-gram, so the filter cannot catch them. This mostly affects short ARC and PIQA items

## Limitations

- A 50M base model: it makes up facts, drifts off topic and repeats itself. Use it for research, education and on-device experiments, not for anything that needs correct answers
- English only. Context is 1,024 tokens
- Not instruction tuned and not safety tuned. It can produce biased or offensive text that is present in web data
- Weak at arithmetic. It sometimes works out 2-digit sums in its own wording, and it is near chance on word problems worded by other people

## Licence

CC-BY-NC-4.0, because the largest data source (ClimbMix) is non-commercial upstream. The training code is Apache-2.0.

## Citation

```bibtex
@misc{kashyap2026mantis50mbase,
  title  = {Mantis-50M-Base: a 49M parameter language model pretrained from scratch},
  author = {Rahul Kashyap},
  year   = {2026},
  url    = {https://huggingface.co/kash8080/Mantis-50M-Base}
}
```
