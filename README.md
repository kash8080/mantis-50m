# SLM-50M

A language model with **under 50M parameters**, pretrained from scratch on about **$30 of Modal credits**, aimed at the **#1 spot in the under-50M class** of the [Open SLM Leaderboard](https://huggingface.co/spaces/AxiomicLabs/Open_SLM_Leaderboard).

**Status (24 Oct 2026): gate 0 code written, tests pass on the Mac.** Nothing run on Modal yet. Every design decision is in [docs/decisions.md](docs/decisions.md). Next is gate 1: tokenizer and data.

## Why this can work

- The leaderboard ranks by an **Intelligence Index**: HellaSwag, ARC, PIQA and ArithMark-3 (grade-school arithmetic), each normalized against random guessing
- The under-50M #1 scores **17.93**, and every model in that class sits at 35-40% on ArithMark (chance is 25%)
- A small model trained on good web data plus a lot of our own synthetic arithmetic should reach **~20-25**. Details in [docs/01-goal-and-leaderboard.md](docs/01-goal-and-leaderboard.md)

## The plan in one view

| Area | Plan | Status |
|---|---|---|
| Size | **49,322,240** total params (tied embedding counted once) | Decided |
| Architecture | Dense Qwen3-style decoder, 25 layers x 384, 6 / 2 KV heads, SwiGLU 1,152, QK-norm, RoPE, tied embeddings. Exports as `Qwen3ForCausalLM`, quantizes with llama.cpp | Decided |
| Tokenizer | Own byte-level BPE, 16,384, one token per digit (SmolLM2's pre-tokenizer) | Decided |
| Data | 45% ClimbMix + TxT360-v2, FineWeb-Edu, Cosmopedia, FineMath, word problems, 7% synthetic arithmetic. Shifts toward edu, Q&A and math in the decay. Benchmark 13-gram filter. Mixed in every batch | Decided, weights confirmed at gate 4 |
| Training | Token-Superposition Training (bags of 6) for the first 30%, `torch.optim.Muon` + AdamW, WSD with 1-sqrt decay over the last 20%, context 1,024, batch 262k tokens | Decided, TST and Muon confirmed at gate 4 |
| Licence | CC-BY-NC-4.0 (ClimbMix is NC upstream) | Decided |
| Compute | Modal. ~4.4h on one H100 (~$18.30, $19 budget) for the main run, cheap GPUs for tests and evals | Decided |
| Safety | Staged gates, checkpoints every ~20 min, exact resume, budget hard stop | Decided |
| Monitoring | Per-source val loss, arithmetic proxy, real benchmarks every ~1B tokens, alerts | Decided |
| Release | Final checkpoint at the end of the decay, all eval JSON kept, GGUF 8-bit and 4-bit | Decided |
| Next | ~96M model next month for the under-100M class | Decided |

## Repo layout

One folder per pipeline stage (D-037). Tests mirror the same folders.

```
src/slm50m/
  config.py      model presets (main 49.3M, proxy 13.6M, test), mix weights, RunConfig
  tokenizer_corpus/  build the tokenizer's ~2 GB training corpus in data/tokenizer/
  tokenizer/     train the BPE tokenizer on that corpus
  arith/         synthetic arithmetic generator and the 4-choice proxy eval
  dataprep/      sources, benchmark 13-gram filter, parquet -> token shards + manifests, mixture sampler
  model/         the Qwen3-style decoder and its export to Qwen3ForCausalLM
  training/      loss, TST, Muon + AdamW, WSD schedule, checkpoints, metrics + alerts, the training loop
  evaluation/    lm-eval + ArithMark + Intelligence Index, with versions
scripts/         setup_llama_cpp.sh only. Python commands live in each stage's cli.py (uv run slm-<name>, listed in pyproject.toml)
configs/         run configs: smoke (gate 2), proxy_baseline (gate 4), main (gate 5)
data/            data we download or sample. Gitignored except each folder's README.md and manifest.json
outputs/         what we produce and keep, e.g. outputs/tokenizer/ (the trained tokenizer, committed)
modal_app.py     Modal image, Volumes and functions. Paid calls print the cost and ask first
tests/           gate 0 tests
docs/            design docs, decision log, open questions, experiment log (start at docs/README.md)
```

## Setup

```bash
uv sync
```

```bash
uv run pytest
```

`uv run pytest -m "not network"` skips the one test that downloads HellaSwag. The GGUF test skips until llama.cpp is built with `bash scripts/setup_llama_cpp.sh` (needs `brew install cmake`).

```bash
uv run modal setup
```


## Commands

| What | Command |
|---|---|
| Build the tokenizer corpus (gate 1) | `uv run slm-build-tokenizer-corpus --gb 2` |
| Train the tokenizer (gate 1) | `uv run slm-train-tokenizer` |
| Benchmark items for the filter | `uv run modal run modal_app.py::prep_benchmarks` |
| Prep a source into shards | `uv run modal run modal_app.py::prep --source wikihow --n-files 1` |
| Local data from the tokenizer corpus (Mac) | `uv run slm-prep-local` |
| Train locally (Mac smoke test, ~8 min) | `uv run slm-train configs/mac.json` |
| Train on Modal | `uv run modal run modal_app.py::run_train --config configs/smoke.json --gpu L4 --hours 0.4` |
| Throughput (gate 3) | `uv run modal run modal_app.py::run_benchmark --gpu H100 --data` |
| Run status | `uv run slm-status <run_dir>/metrics.jsonl` |
| Run charts (open `<run_dir>/plots.html`) | `uv run slm-plot <run_dir> --watch 30` |
| Export / evaluate | `uv run slm-export-hf <ckpt> <out>`, `uv run slm-evaluate <ckpt> <out>` |

## Budget

$30 of Modal Starter credits (they reset monthly). Breakdown in [docs/06-infra-modal.md](docs/06-infra-modal.md): ~$16 main run, ~$5-9 for data prep, tests and evals, ~$5-7 reserve. Updated 27 Oct 2026 with $26 left: $19 main run (D-042), ~$7 for gate 4, evals and reserve.

## Next steps

1. Gate 1: build the tokenizer corpus, train the tokenizer, prep the data ([docs/08-roadmap.md](docs/08-roadmap.md))
2. Gates 2 to 6: smoke test, throughput, ablations, main run, release. The GGUF test must pass before gate 5
3. Still open: token budget and GPU type (gate 3), model name and HF account (Rahul). See [docs/open-questions.md](docs/open-questions.md)
