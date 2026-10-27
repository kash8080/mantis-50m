"""Tokenize a slice of the tokenizer corpus into a local data folder, for Mac smoke tests.

    uv run slm-prep-local                     # 10 MB of text per source -> _build/data_local
    uv run slm-train configs/mac.json

Same layout as the Modal prep (tokens/<source>/train_00000.bin + val.bin), so slm-train reads it unchanged.
No download and no benchmark filter: the real training data is prepped on Modal (modal_app.py::prep).
"""

import argparse

from slm50m.config import BUILD_DIR, STABLE_MIX, TOKENIZER_CORPUS_DIR, TOKENIZER_DIR
from slm50m.dataprep.prep import prep_corpus_source
from slm50m.tokenizer.train_tokenizer import load_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", default=str(TOKENIZER_CORPUS_DIR))
    parser.add_argument("--tokenizer-dir", default=str(TOKENIZER_DIR))
    parser.add_argument("--out", default=str(BUILD_DIR / "data_local"))
    parser.add_argument("--mb", type=float, default=10.0, help="MB of text per source")
    parser.add_argument("--val-docs", type=int, default=200)
    args = parser.parse_args()

    tok = load_tokenizer(args.tokenizer_dir)
    for source in STABLE_MIX:
        stats = prep_corpus_source(args.corpus_dir, source, args.out, tok, args.mb * 1e6, args.val_docs)
        print(f"{source:15s} {stats['tokens_train']:>12,} train  {stats['tokens_val']:>9,} val tokens")
    print(f"wrote {args.out}")
