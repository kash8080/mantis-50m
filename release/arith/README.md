---
license: cc-by-4.0
language:
- en
pretty_name: Mantis-50M synthetic arithmetic
size_categories:
- 1M<n<10M
task_categories:
- text-generation
tags:
- synthetic
- arithmetic
- math
configs:
- config_name: v2
  default: true
  data_files:
  - split: train
    path: v2/train-*.parquet
  - split: validation
    path: v2/validation.parquet
- config_name: v1-sample
  data_files:
  - split: train
    path: v1-sample/train-*.parquet
  - split: validation
    path: v1-sample/validation.parquet
---

# Mantis-50M synthetic arithmetic

The synthetic grade-school arithmetic text that [Mantis-50M](https://huggingface.co/kash8080/Mantis-50M) was trained on, published so anyone can check it. It is decoded from the exact token shards used in training, not regenerated.

| Config | What | Train docs | Validation docs | Train tokens |
|---|---|---|---|---|
| `v2` (default) | All of the decay-phase data (12% of the decay mix) | 1,129,331 | 2,000 | 299,478,229 |
| `v1-sample` | 1 of the 20 stable-phase shards (7% of the stable mix) | 305,306 | 2,000 | 99,362,627 |

The decay drew about 214M tokens from `v2`, under one pass. The full v1 data is 20 shards like `v1-sample` (6.1M documents, 2.0B tokens). Validation documents were held out of training. Tokens are Mantis-50M tokens, counting one `<|endoftext|>` per document.

```python
from datasets import load_dataset

v2 = load_dataset("kash8080/Mantis-50M-arith", "v2", split="train")
print(v2[0]["text"])
```

## Fields

| Field | Meaning |
|---|---|
| `text` | One training document, exactly as trained (without the `<|endoftext|>` that followed it) |
| `generator` | `arith_v2` or `arith` (v1) |
| `shard` | Training shard index. Each shard is one seeded run of the generator |
| `doc` | Position in the shard, in training order |

## How it was made

Both generators are plain Python in [`src/slm50m/arith/`](https://github.com/kash8080/mantis-50m/tree/main/src/slm50m/arith): `generator.py` (v1) and `generator_v2.py` with `lexicon.py` (v2). They fill templates from general English word lists: names, objects, containers, gain and loss verbs, lead phrases. Every number is computed, so every answer is correct. Documents are worksheets, drills, answer keys, dialogues and stories with one or more problems each.

- **v1** covers addition, subtraction, multiplication, division, sequences, money, time, measurement and tallies, with 50 names and a small set of fixed wordings
- **v2** was written after v1's skill turned out to work only in v1's own wording. It has ~500 names with pronoun agreement, ~200 objects, ~90 verbs, ~110 lead phrases and ten answer formats (story, question and answer, worked steps, equation, dialogue, fill-in and others). It covers the usual grade-school topics, which overlap with ArithMark-3's
- This dataset was decoded from the training shards on the Modal Volume. Each shard's document and token counts match the training manifests, and every 100th document re-encodes to the same token ids

## Regenerating all of v1

The generators are deterministic, so the 19 v1 shards that are not published here can be rebuilt exactly. Shard *k* of v1 comes from `random.Random(k)`, for *k* = 0 to 19. Shard *k* of v2 comes from `random.Random(1000 + k)`. Each shard draws documents one at a time from its RNG. In shard 0, the first 2,000 documents are the validation split and the rest are train. Regenerating seeds 0 and 1 gave the published documents, from the first one to the last one of v1 shard 0.

```python
import json
import random

from slm50m.arith.generator import generate_document  # v2: slm50m.arith.generator_v2

def v1_shard(k: int) -> list[str]:
    n_docs = json.load(open("manifests/arith.json"))["files"][k]["docs_kept"]
    rng = random.Random(k)
    return [generate_document(rng) for _ in range(n_docs)]
```

Run it from a clone of the [code repo](https://github.com/kash8080/mantis-50m) after `uv sync`. [`manifests/arith.json`](https://github.com/kash8080/mantis-50m/blob/main/manifests/arith.json) has each shard's document and token counts.

## Decontamination

- No benchmark item was used to write the generators
- Zero exact matches and zero 13-gram matches with ArithMark-3 over ~54,000 (v1) and ~55,000 (v2) generated problems in every format. The tests are `test_no_overlap_with_arithmark` and `test_v2_no_overlap_with_arithmark` in [`tests/arith/test_arith.py`](https://github.com/kash8080/mantis-50m/blob/main/tests/arith/test_arith.py)
- Every `v2` document also went through the 13-gram filter against HellaSwag, ARC-Easy, ARC-Challenge, PIQA and ArithMark-3 before training: zero documents dropped

## What it does and does not teach

The decay, with `v2` as 12% of its mix, raised Mantis-50M's ArithMark-3 score from 37.3% to 85.7%. There was no decay without `v2` to compare against, so part of that rise may come from the rest of the decay mix. On grade-school word problems written by other people (ASDiv and SVAMP, 232 problems that no training data paraphrases) the gain is much smaller: 31 to 41% against 25% chance, up 10 to 20 points. Details in [the model card](https://huggingface.co/kash8080/Mantis-50M#held-out-arithmetic). Wider wording does not by itself give general arithmetic.

## Known issues

The text is unedited generator output. It has occasional grammar slips ("That left she with 22") and odd pairings of objects and containers (horses shared into bags). The arithmetic itself is always correct.

## Licence

CC-BY-4.0. The generator code is Apache-2.0.
