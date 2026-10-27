"""Held-out 4-choice arithmetic proxy, scored like ArithMark (D-014, D-024). Monitoring only.

Context and each ending are tokenized separately (no special tokens), the ending is scored by its
mean log-prob per token, and the best-scoring ending is the prediction. Chance is 25%.
"""

import random

import torch
import torch.nn.functional as F

from slm50m.arith.generator import random_problem

PROXY_SEED = 1_000_003  # far from any training seed (prep uses small seeds)


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


@torch.no_grad()
def score_items(model, tokenizer, items: list[dict], device: str | torch.device) -> float:
    """Accuracy in [0, 1]. `tokenizer` is the raw tokenizers.Tokenizer. One forward of 4 rows per item."""
    model.eval()
    correct = 0
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
        correct += int(max(range(len(scores)), key=scores.__getitem__) == item["label"])
    return correct / len(items)
