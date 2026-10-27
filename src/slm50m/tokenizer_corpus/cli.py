"""Download and write the tokenizer's training corpus to data/tokenizer/ (gate 1).

    uv run slm-build-tokenizer-corpus --gb 2

Then train on it with `uv run slm-train-tokenizer`.
"""

import argparse

from slm50m.config import TOKENIZER_CORPUS_DIR
from slm50m.tokenizer_corpus.build import build_corpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gb", type=float, default=2.0, help="corpus size in GB of text")
    parser.add_argument("--out", default=str(TOKENIZER_CORPUS_DIR))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    manifest = build_corpus(args.out, args.gb * 1e9, seed=args.seed)
    total = sum(s["bytes"] for s in manifest["sources"].values())
    print(f"wrote {total / 1e9:.2f} GB to {args.out}")
