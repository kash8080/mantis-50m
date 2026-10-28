---
license: cc-by-nc-4.0
language:
- en
base_model: kash8080/Mantis-50M
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- gguf
- llama.cpp
- small-language-model
---

# Mantis-50M GGUF

[Mantis-50M](https://huggingface.co/kash8080/Mantis-50M) in GGUF format for [llama.cpp](https://github.com/ggml-org/llama.cpp) and the apps built on it. Mantis-50M is a 49.3M parameter base model pretrained from scratch on about $30 of cloud GPU credits. With the [Open SLM Leaderboard](https://huggingface.co/spaces/AxiomicLabs/Open_SLM_Leaderboard)'s evaluation tools it scores an Intelligence Index of 30.05 (not on the leaderboard yet). The [main model card](https://huggingface.co/kash8080/Mantis-50M) has the results, data and training details, and the synthetic arithmetic it was trained on is public as [kash8080/Mantis-50M-arith](https://huggingface.co/datasets/kash8080/Mantis-50M-arith).

| File | Type | Size | Quality vs the original |
|---|---|---|---|
| `Mantis-50M-F16.gguf` | F16 | 99 MB | Same output: mean KL divergence ~1e-6 nats, same top token on every test prompt |

One file on purpose. F16 is effectively lossless here, and smaller types cost too much on a model this small (see below).

## Usage

It is a base model with no chat template: give it the start of a text and it continues.

```bash
llama-completion -m Mantis-50M-F16.gguf -p "Photosynthesis is the process by which" -n 64 --temp 0 --repeat-penalty 1.1
```

```
Photosynthesis is the process by which plants convert sunlight into chemical energy.
```

`llama-server -m Mantis-50M-F16.gguf` serves it on an OpenAI-style completions endpoint. With Ollama: `ollama run hf.co/kash8080/Mantis-50M-GGUF` (tested on Ollama 0.24.0). Ollama samples by default. With temperature 0 and repeat penalty 1.1 it gives the same text as above.

## Speed and memory

Apple M2 Pro, llama.cpp commit `df9beac`, `llama-bench` (prompt 512 tokens, generation 128 tokens):

| Backend | Prompt (tokens/s) | Generation (tokens/s) |
|---|---|---|
| CPU, 6 threads | 3,980 | 620 |
| Metal | 26,460 | 359 |

At this size the CPU generates faster than the GPU. Peak memory on the CPU is ~215 MB with the full 1,024-token context: 99 MB of weights, the rest is the KV cache and compute buffers.

## How it was checked

Against the original fp32 model in transformers, on 7 prompts (prose, arithmetic, digits, tabs, unicode):

| Check | Result |
|---|---|
| Tokenization (`llama-tokenize` vs the HF tokenizer) | Identical ids |
| Last-token logits, CPU, f32 KV cache | Max relative error 1.2e-3, mean KL 5e-6 nats, top token same 7/7 |
| Last-token logits, Metal, defaults | Max relative error 4e-4, mean KL 1e-6 nats, top token same 7/7 |
| Greedy generation (32 tokens) | Same text as transformers |

## Why F16 only

- **F16 vs BF16:** on these weights F16's rounding error is 8x smaller on average (64x in total squared error). The largest weight is 4.6, far inside F16's range
- **K-quants don't fit this shape:** Q4_K, Q5_K and Q6_K need rows that are a multiple of 256. Mantis-50M's rows are 384 and 1,152 wide, so llama.cpp silently swaps in Q5_0 and Q8_0: a "Q4_K_M" file of this model is mostly Q5_0
- At 50M parameters every weight carries more of the model than in a large model, so 4- and 5-bit rounding costs more. A smaller file may come later, with its measured quality loss

## Converting it yourself

The tokenizer is new to llama.cpp, so `convert_hf_to_gguf.py` stops with "BPE pre-tokenizer was not recognized" until its hash is added upstream. [`scripts/setup_llama_cpp.sh`](https://github.com/kash8080/mantis-50m/blob/main/scripts/setup_llama_cpp.sh) in the code repo adds it locally (it uses SmolLM's pre-tokenizer).

## Licence

CC-BY-NC-4.0, same as [Mantis-50M](https://huggingface.co/kash8080/Mantis-50M). Code: [github.com/kash8080/mantis-50m](https://github.com/kash8080/mantis-50m). Arithmetic data: [kash8080/Mantis-50M-arith](https://huggingface.co/datasets/kash8080/Mantis-50M-arith).
