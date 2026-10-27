"""Held-out 4-choice arithmetic proxies, scored like ArithMark (D-014, D-024, D-045). Monitoring only.

Two item sets:
- make_items: v1 problems from an unseen seed. Same wording as the v1 training data, so it measures recall
  of that wording (it reached 98.5% while ArithMark addition stayed at chance).
- make_transfer_items: v2 problems built only from the held-out word lists (arith/lexicon.py), with
  ArithMark-style distractors (answer -2..+2, never the smallest or largest). It measures whether the
  skill fires on wording the model never saw in arithmetic text.

Context and each ending are tokenized separately (no special tokens), the ending is scored by its
mean log-prob per token, and the best-scoring ending is the prediction. Chance is 25%.
"""

import random

import torch
import torch.nn.functional as F

from slm50m.arith import generator_v2
from slm50m.arith.generator import random_problem

PROXY_SEED = 1_000_003  # far from any training seed (prep uses small seeds)
TRANSFER_SEED = 2_000_003


def make_items(n: int, seed: int = PROXY_SEED) -> list[dict]:
    """n items: {"ctx", "endings" (4, leading space), "label"}. Distractors are near misses of the answer."""
    rng = random.Random(seed)
    items = []
    while len(items) < n:
        p = random_problem(rng)
        wrong = sorted({p.value + d for d in (-10, -2, -1, 1, 2, 10) if p.value + d >= 0} - {p.value})
        values = [p.value] + rng.sample(wrong, 3)
        rng.shuffle(values)
        items.append({"ctx": f"{p.stem} {p.lead}", "endings": [f" {p.render(v)}." for v in values],
                      "label": values.index(p.value)})
    return items


def middle_distractors(value: int, rng: random.Random) -> list[int]:
    """Three wrong values within 2 of the answer, with the answer never the smallest or largest (as in ArithMark)."""
    if value < 2:
        return [value + 1, value + 2, value + 3]
    return [value - 1, value + 1, rng.choice([value - 2, value + 2])]


def make_transfer_items(n: int, seed: int = TRANSFER_SEED) -> list[dict]:
    """n items from held-out wording only, with a "topic" field for per-topic accuracy."""
    rng = random.Random(seed)
    items = []
    while len(items) < n:
        p = generator_v2.random_problem(rng, "heldout")
        values = [p.value] + middle_distractors(p.value, rng)
        endings = [f" {p.render(v)}." for v in values]
        if len(set(endings)) < 4:  # a render that hides the value, e.g. a fixed phrase
            continue
        rng.shuffle(values)
        items.append({"ctx": generator_v2.sentence_caps(f"{p.stem} {p.lead}"),
                      "endings": [f" {p.render(v)}." for v in values], "label": values.index(p.value), "topic": p.topic})
    return items


@torch.no_grad()
def score_items(model, tokenizer, items: list[dict], device: str | torch.device) -> float:
    """Accuracy in [0, 1]. `tokenizer` is the raw tokenizers.Tokenizer. One forward of 4 rows per item."""
    hits = item_hits(model, tokenizer, items, device)
    return sum(hits) / len(items)


@torch.no_grad()
def score_by_topic(model, tokenizer, items: list[dict], device: str | torch.device) -> dict[str, float]:
    """Accuracy per item topic, plus "all"."""
    hits = item_hits(model, tokenizer, items, device)
    out: dict[str, list[int]] = {}
    for item, hit in zip(items, hits):
        out.setdefault(item.get("topic", "all"), []).append(hit)
    scores = {t: sum(h) / len(h) for t, h in sorted(out.items())}
    scores["all"] = sum(hits) / len(hits)
    return scores


@torch.no_grad()
def item_hits(model, tokenizer, items: list[dict], device: str | torch.device) -> list[int]:
    """1 where the model picks the right ending, else 0. One forward of 4 rows per item."""
    model.eval()
    hits = []
    for item in items:
        ctx = tokenizer.encode(item["ctx"], add_special_tokens=False).ids
        endings = [tokenizer.encode(e, add_special_tokens=False).ids for e in item["endings"]]
        rows = [ctx + e for e in endings]
        width = max(map(len, rows))
        ids = torch.zeros(len(rows), width, dtype=torch.long)  # pad with 0 (eos), never scored
        for i, row in enumerate(rows):
            ids[i, :len(row)] = torch.tensor(row)
        logprobs = F.log_softmax(model(ids.to(device)).float(), dim=-1)  # (4, W, V)
        scores = []
        for i, e in enumerate(endings):
            # ending token j sits at position len(ctx)+j and is predicted from position len(ctx)+j-1
            pos = torch.arange(len(ctx) - 1, len(ctx) - 1 + len(e), device=logprobs.device)
            target = torch.tensor(e, device=logprobs.device)
            scores.append(logprobs[i, pos, target].mean().item())
        hits.append(int(max(range(len(scores)), key=scores.__getitem__) == item["label"]))
    return hits
