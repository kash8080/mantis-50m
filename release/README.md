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

# Mantis-50M

Mantis-50M is a **49.3M parameter** base language model, pretrained from scratch on about **$30 of cloud GPU credits**. With the [Open SLM Leaderboard](https://huggingface.co/spaces/AxiomicLabs/Open_SLM_Leaderboard)'s evaluation tools it scores an Intelligence Index of **30.05**. The best under-50M model listed on 28 Oct 2026 scores 17.93. It is not on the leaderboard yet.

- **Code:** [github.com/kash8080/mantis-50m](https://github.com/kash8080/mantis-50m)
- **GGUF (llama.cpp):** [kash8080/Mantis-50M-GGUF](https://huggingface.co/kash8080/Mantis-50M-GGUF)

It is a plain base model: no instruction tuning, no chat template. It continues text.

## Results

Zero-shot, with the leaderboard's tools: lm-eval 0.4.12 (`acc_norm`, fp32) for HellaSwag, ARC and PIQA, and the official ArithMark-3 script (default settings). This is the final checkpoint of the run (step 34,000). It was fixed as the release before any final eval, not picked by score.

| Model | Params | HellaSwag | ARC-E | ARC-C | PIQA | ArithMark | II |
|---|---|---|---|---|---|---|---|
| **Mantis-50M** | 49.32M | **34.43** | **46.17** | **26.02** | **64.85** | **85.70** | **30.05** |
| 50M-pro (under-50M #1) | 49.59M | 33.25 | 45.62 | 25.51 | 64.80 | 37.40 | 17.93 |
| KLM-1.5 (under-100M #1) | 99.0M | 43.74 | 59.68 | 32.68 | 68.55 | 48.20 | 30.26 |

Mantis-50M leads 50M-pro on all five benchmarks. Most of the gap comes from ArithMark, which our synthetic arithmetic data targets (see Data). Without ArithMark, the other four still lead: 15.64 vs 15.00 of the index. The arithmetic transfers only partly to problems worded differently from our data: see [Held-out arithmetic](#held-out-arithmetic).

ArithMark is uneven across its 17 topics. 14 score above 75%, but two-step subtraction is at 37.5% (32 items) and data at 51.2% (43 items). Per-topic scores are in `evals/arithmark_results.json`.

The raw eval files are in [`evals/`](evals/): `summary.json` (scores, tool versions, ArithMark data and script hashes), `lm_eval_results.json` and the ArithMark results.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("kash8080/Mantis-50M")
model = AutoModelForCausalLM.from_pretrained("kash8080/Mantis-50M")

x = tok("Photosynthesis is the process by which", return_tensors="pt")
out = model.generate(**x, max_new_tokens=48, do_sample=False, repetition_penalty=1.1)
print(tok.decode(out[0], skip_special_tokens=True))
```

Tested with transformers 4.57.6 and 5.20.2 (same logits). The model is a standard `Qwen3ForCausalLM`, so no `trust_remote_code` is needed.

Greedy samples (repetition penalty 1.1), prompt in bold:

> **Photosynthesis is the process by which** plants convert sunlight into chemical energy.

> **Once upon a time, there was a little robot who** loved to play with his friends. One day, he asked his friend, "What's a robot?"

> **Question: Sara has 14 apples and gives away 5. How many apples does she have left?\nAnswer:** Sara has 9 apples.

## Model

| | |
|---|---|
| Architecture | Dense Qwen3-style decoder (`Qwen3ForCausalLM`) |
| Shape | 25 layers, hidden 384, 6 query heads and 2 KV heads (head dim 64), SwiGLU 1,152 |
| Details | QK-norm, RoPE theta 100,000, RMSNorm, tied input and output embeddings, no biases |
| Parameters | 49,322,240 (tied embedding counted once) |
| Context | 1,024 tokens |
| Tokenizer | Own byte-level BPE, 16,384 tokens, one token per digit. `<|endoftext|>` (id 0) ends documents. No BOS |
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
| Cost | About $28 of Modal credits for everything: data prep, tests, ablations, the main run and evals |

## Data

| Source | Licence | Stable mix | Decay mix |
|---|---|---|---|
| [ClimbMix](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle) | MIT on the reshuffle, CC-BY-NC-4.0 upstream | 45% | 20% |
| [TxT360-v2](https://huggingface.co/datasets/IFM/TxT360-v2) web | CC-BY-4.0 | 12% | 6% |
| TxT360-v2 QA (`txt360-qa`, `web-high-nltk-qa`) | CC-BY-4.0 | 8% | 14% |
| [FineWeb-Edu dedup](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | ODC-BY | 10% | 14% |
| [Cosmopedia v2](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | ODC-BY | 5% | 9% |
| [Cosmopedia WikiHow](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) | Apache-2.0 | 2% | 9% |
| [FineMath-4plus](https://huggingface.co/datasets/HuggingFaceTB/finemath) | ODC-BY | 7% | 8% |
| Word problems: [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) (GSM8K-type rows only) and [Orca-Math](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k) | CC-BY-4.0, MIT | 4% | 8% |
| Synthetic arithmetic, our generators ([kash8080/Mantis-50M-arith](https://huggingface.co/datasets/kash8080/Mantis-50M-arith)) | CC-BY-4.0 | 7% (v1) | 12% (v2) |

Every source is pinned to a Hub commit in the code. Every batch holds every source in its mix share.

**Synthetic arithmetic.** Grade-school arithmetic stories, questions and worked answers, made by our own generators ([`src/slm50m/arith/`](https://github.com/kash8080/mantis-50m/tree/main/src/slm50m/arith)). They are written from general English word lists, never from benchmark items. The decay phase uses a second, wider generator with ~500 names, ~200 objects and ten answer formats, because the first one taught the skill only inside its own wording. This data is why ArithMark is high, and we say so plainly. The exact text is published as [kash8080/Mantis-50M-arith](https://huggingface.co/datasets/kash8080/Mantis-50M-arith): all of the decay data and a sample of the stable-phase data.

## Held-out arithmetic

ArithMark's 85.7% overstates the model's general arithmetic. We tested it on grade-school word problems from [ASDiv](https://huggingface.co/datasets/EleutherAI/asdiv) and [SVAMP](https://huggingface.co/datasets/ChilleD/SVAMP), written by other people and scored the same way as ArithMark (4 choices, mean log-prob per answer token). Orca-Math, one of our training sources, contains most of both sets, so we kept only the 232 problems that share no 8-word span with it, even with numbers masked. "Before" is our own checkpoint at step 26,703, before the decay. It is not released, so only the Mantis-50M column can be rerun.

| Prompt / wrong choices | Before | Mantis-50M | ArithMark-3 (before, after) |
|---|---|---|---|
| "... The answer is N." / N±1, N±2 | 25.4 | 35.8 | 37.3, 85.7 |
| "... The answer is N." / wrong-operation results | 15.1 | 32.8 | |
| "... The answer is N apples." / wrong-operation results | 21.0 | 41.1 | |
| "Question: ... Answer: N" / N±1, N±2 | 28.4 | 31.0 | |

Chance is 25%, standard error about 3 points. The decay, with its synthetic arithmetic, taught some arithmetic that carries over to other people's wording (+10 to +20 points in story form), much less than the +48 on ArithMark suggests. Subtraction is the weakest. ASDiv is harder than ArithMark (grades 1 to 6, larger numbers), so these numbers are not an ArithMark estimate.

To rerun it, use [`heldout_arith.py`](https://github.com/kash8080/mantis-50m/blob/main/src/slm50m/evaluation/heldout_arith.py) (`uv run slm-heldout-arith`). It rebuilds the 232 problems from pinned copies of ASDiv, SVAMP and Orca-Math, checks them against the [committed ID list](https://github.com/kash8080/mantis-50m/blob/main/src/slm50m/evaluation/heldout_arith_items.json), and scores any HF model. A rerun from fresh downloads gave the numbers above exactly. It takes about 3 minutes on a laptop CPU.

## Other math benchmarks

To check that the arithmetic is learned and not fitted to ArithMark, we ran three more public math benchmarks: zero-shot, lm-eval 0.4.12 prompts and metrics. "Before" is our checkpoint at step 27,200, the last one before the decay (not released).

| Benchmark (chance) | Items | Scoring | Before | Mantis-50M |
|---|---|---|---|---|
| [GPT-3 arithmetic](https://huggingface.co/datasets/EleutherAI/arithmetic), mean of 10 tasks (0) | 20,000 | exact match of the greedy answer | 16.6 | **36.7** |
| 2-digit add / subtract | 2,000 each | | 51.9 / 23.8 | 93.3 / 50.9 |
| 3-digit add / subtract | 2,000 each | | 35.4 / 20.8 | 80.1 / 46.1 |
| 4-digit add / subtract | 2,000 each | | 8.5 / 5.5 | 42.0 / 33.9 |
| 5-digit add / subtract | 2,000 each | | 0.3 / 0.3 | 3.1 / 4.8 |
| 2-digit multiply | 2,000 | | 16.0 | 4.5 |
| 1-digit compound, e.g. (2+3)*4 | 2,000 | | 3.6 | 8.6 |
| [MathQA](https://huggingface.co/datasets/allenai/math_qa) (20) | 2,985 | 5 choices, `acc_norm` | 23.4 | 24.6 |
| [MMLU](https://huggingface.co/datasets/cais/mmlu) elementary mathematics (25) | 378 | 4 choices, letters, `acc` | 26.7 | 26.5 |
| MMLU elementary mathematics, answer text (25) | 378 | 4 choices, `acc_norm` | 27.0 | 24.1 |

GPT-3 arithmetic asks "Question: What is 98 plus 45? Answer:" and counts an answer only if the model writes it exactly, so there are no choices to guess from. Addition and subtraction up to 4 digits rose 25 to 45 points in the decay. Two-digit multiplication fell, likely because our generators mostly multiply small numbers. MathQA and MMLU stay at chance before and after, as expected at this size: the model has no general math reasoning, and the arithmetic data did not cost it any.

To rerun it, use [`math_evals.py`](https://github.com/kash8080/mantis-50m/blob/main/src/slm50m/evaluation/math_evals.py) (`modal run modal_app.py::run_math_evals`, about 12 minutes on one A10 for both checkpoints). It also reruns the held-out arithmetic above. The final model's numbers there matched exactly.

<details>
<summary>Samples of the synthetic arithmetic data (unedited)</summary>

From the decay-phase generator (`generator_v2.py`):

```
Think and solve
Melody has 9 folders. Ray has twice as many folders. How many folders does Ray have? Multiply: 9 × 2 = 18. So Ray has 18 folders.
The pumpkin measured 8 kilograms. Written in grams, that is 8000 grams.
Question: A class voted on their favorite snack: popcorn 4, carrots 28, crackers 14, pretzels 12. How many votes in all?
Answer: 58 votes.
```

```
Practice sheet
(1) There were 81 pebbles in Bianca's pail. Bianca was given 7 more. Then she put 79 away. She now has 9 pebbles.
(2) 4 × 4 = 16
(3) Audrey came to the book fair with 38 snowballs. Audrey earned 13 more snowballs. Then she gave 29 to William. What is the number of snowballs she has at the end? That left she with 22.
(4) Look at the pattern: 147, 135, 123, 111. What number comes next? Continuing the pattern gives 99.
(5) Megan shared 50 shells equally among 10 friends. How many does each get? 50 ÷ 10 = 5. In each group there are 5 shells.
```

From the stable-phase generator (`generator.py`):

```
Homework check

7 - 2 = 5

There were 33 tomatoes in a jar, and 23 were removed. How many tomatoes remained in the jar?
Step 1: 33 - 23 = 10. The jar was left with 10 tomatoes.

18 × 13 = 234
```

All the decay data, exactly as trained: [kash8080/Mantis-50M-arith](https://huggingface.co/datasets/kash8080/Mantis-50M-arith). Generator code: [`src/slm50m/arith/`](https://github.com/kash8080/mantis-50m/tree/main/src/slm50m/arith).

</details>

## Decontamination

- Every training document that shares a 13-word n-gram with HellaSwag, ARC-Easy, ARC-Challenge, PIQA or ArithMark-3 was dropped before training: 1,474 of 28.9M documents. Per-source numbers are in the [filter report](https://github.com/kash8080/mantis-50m/blob/main/docs/filter-report.md)
- Both arithmetic generators have zero exact matches and zero 13-gram matches with ArithMark-3, tested over ~54,000 (v1) and ~55,000 (v2) generated problems in every format
- Benchmark items under 13 words have no 13-gram, so the filter cannot catch them. This mostly affects short ARC and PIQA items

## Limitations

- A 50M base model: it makes up facts, drifts off topic and repeats itself. Use it for research, education and on-device experiments, not for anything that needs correct answers
- English only. Context is 1,024 tokens
- Not instruction tuned and not safety tuned. It can produce biased or offensive text that is present in web data
- The arithmetic skill is mostly grade-school, two-digit problems in story form. Bare addition and subtraction work up to about 4 digits, multiplication of two 2-digit numbers mostly fails. See [Other math benchmarks](#other-math-benchmarks)
- ArithMark-3's 85.7% overstates the model's arithmetic. On grade-school problems worded by other people it scores 31 to 41% (chance 25%). See [Held-out arithmetic](#held-out-arithmetic)

## Licence

CC-BY-NC-4.0, because the largest data source (ClimbMix) is non-commercial upstream. The training code is Apache-2.0.

## Citation

```bibtex
@misc{kashyap2026mantis50m,
  title  = {Mantis-50M: a 49M parameter language model pretrained from scratch on a $30 budget},
  author = {Rahul Kashyap},
  year   = {2026},
  url    = {https://huggingface.co/kash8080/Mantis-50M}
}
```
