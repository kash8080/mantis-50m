"""The benchmark items the 13-gram filter checks against (D-025).

Same splits and text the leaderboard scores: HellaSwag validation, ARC-Easy and ARC-Challenge test,
PIQA validation (dataset paths as in lm_eval 0.4.12's task files), and all 1,000 ArithMark-3 items.
Each item gives its context joined with every choice, so n-grams that cross the boundary count too.
"""

import json
from collections.abc import Iterator
from pathlib import Path

from slm50m.config import BUILD_DIR

ARITHMARK_URL = "https://huggingface.co/datasets/AxiomicLabs/ArithMark-3.0/resolve/main/arithmark-3.jsonl"
ARITHMARK_PATH = BUILD_DIR / "cache" / "arithmark-3.jsonl"


def arithmark_path() -> Path:
    """Local copy of arithmark-3.jsonl, downloaded once (~0.8 MB)."""
    if not ARITHMARK_PATH.exists():
        import urllib.request

        ARITHMARK_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(ARITHMARK_URL, ARITHMARK_PATH)
    return ARITHMARK_PATH


def load_arithmark(path: str | Path | None = None) -> list[dict]:
    with open(path or arithmark_path()) as f:
        return [json.loads(line) for line in f]


def iter_benchmark_texts() -> Iterator[tuple[str, str]]:
    """(benchmark, text) pairs. Downloads the eval sets with `datasets` (gate 1, Modal CPU)."""
    from datasets import load_dataset

    for row in load_dataset("Rowan/hellaswag", split="validation"):
        ctx = f"{row['activity_label']}: {row['ctx_a']} {row['ctx_b']}"
        for ending in row["endings"]:
            yield "hellaswag", f"{ctx} {ending}"
    for subset in ("ARC-Easy", "ARC-Challenge"):
        for row in load_dataset("allenai/ai2_arc", subset, split="test"):
            for choice in row["choices"]["text"]:
                yield subset.lower(), f"{row['question']} {choice}"
    for row in load_dataset("baber/piqa", split="validation"):
        for sol in (row["sol1"], row["sol2"]):
            yield "piqa", f"{row['goal']} {sol}"
    for row in load_arithmark():
        for ending in row["endings"]:
            yield "arithmark", row["ctx"] + ending
