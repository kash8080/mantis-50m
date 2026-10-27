"""Train the 16,384 tokenizer on the saved corpus in data/tokenizer/ (gate 1).

    uv run slm-build-tokenizer-corpus --gb 2      # first, once
    uv run slm-train-tokenizer                    # writes outputs/tokenizer/

After gate 1 the result is frozen and committed to git (D-018). --vocab-size 8192 / 32768 gives the
comparison tokenizers for the write-up. Write those under _build/, not over outputs/tokenizer.
"""

import argparse

from slm50m.config import TOKENIZER_CORPUS_DIR, TOKENIZER_DIR
from slm50m.tokenizer.train_tokenizer import VOCAB_SIZE, save_tokenizer, train_tokenizer
from slm50m.tokenizer_corpus.build import iter_corpus_texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", default=str(TOKENIZER_CORPUS_DIR))
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--out", default=str(TOKENIZER_DIR))
    args = parser.parse_args()

    tok = train_tokenizer(iter_corpus_texts(args.corpus_dir), args.vocab_size)
    print(f"saved to {save_tokenizer(tok, args.out)} (vocab {tok.get_vocab_size()})")
