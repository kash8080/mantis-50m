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
---

# Mantis-50M

Mantis-50M is a **49.3M parameter** base language model, pretrained from scratch on about **$30 of cloud GPU credits**. On the [Open SLM Leaderboard](https://huggingface.co/spaces/AxiomicLabs/Open_SLM_Leaderboard) it scores an Intelligence Index of **30.05**. The best under-50M model listed on 28 Oct 2026 scores 17.93.

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

Mantis-50M leads 50M-pro on all five benchmarks. Most of the gap comes from ArithMark, which our synthetic arithmetic data targets (see Data). Without ArithMark, the other four still lead: 15.64 vs 15.00 of the index.

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
| Synthetic arithmetic (our generator) | ours | 7% | 12% |

Every source is pinned to a Hub commit in the code. Every batch holds every source in its mix share.

**Synthetic arithmetic.** Grade-school arithmetic stories, questions and worked answers, made by our own generators ([`src/slm50m/arith/`](https://github.com/kash8080/mantis-50m/tree/main/src/slm50m/arith)). They are written from general English word lists, never from benchmark items. The decay phase uses a second, wider generator with ~500 names, ~200 objects and ten answer formats, because the first one taught the skill only inside its own wording. This data is why ArithMark is high, and we say so plainly.

## Decontamination

- Every training document that shares a 13-word n-gram with HellaSwag, ARC-Easy, ARC-Challenge, PIQA or ArithMark-3 was dropped before training: 1,474 of 28.9M documents. Per-source numbers are in the [filter report](https://github.com/kash8080/mantis-50m/blob/main/docs/filter-report.md)
- Both arithmetic generators have zero exact matches and zero 13-gram matches with ArithMark-3, tested over ~54,000 (v1) and ~55,000 (v2) generated problems in every format
- Benchmark items under 13 words have no 13-gram, so the filter cannot catch them. This mostly affects short ARC and PIQA items

## Limitations

- A 50M base model: it makes up facts, drifts off topic and repeats itself. Use it for research, education and on-device experiments, not for anything that needs correct answers
- English only. Context is 1,024 tokens
- Not instruction tuned and not safety tuned. It can produce biased or offensive text that is present in web data
- The arithmetic skill is mostly grade-school, two-digit problems in story form

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
