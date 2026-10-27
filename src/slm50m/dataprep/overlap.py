"""Cross-source exact-duplicate check (D-025).

Hash the normalized text (as the benchmark filter normalizes it) of the first n documents of each source's train
shards, then count how many hashes each pair of sources shares. Measured only: we act if a pair overlaps by more
than 5%. Works from the token shards, so it needs no download.
"""

import hashlib
from collections.abc import Iterator
from itertools import combinations
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from slm50m.config import EOS_ID
from slm50m.dataprep.decontam import normalize_words

READ_TOKENS = 1 << 26  # 64M tokens (128 MB) per read


def iter_documents(shards: list[Path]) -> Iterator[np.ndarray]:
    """Token ids of each document, in shard order, without the <|endoftext|> that ends it."""
    for shard in shards:
        ids = np.memmap(shard, dtype=np.uint16, mode="r")
        carry = np.empty(0, dtype=np.uint16)
        for start in range(0, len(ids), READ_TOKENS):
            block = np.concatenate([carry, ids[start:start + READ_TOKENS]])
            ends = np.flatnonzero(block == EOS_ID)
            begin = 0
            for end in ends:
                yield block[begin:end]
                begin = end + 1
            carry = block[begin:]
        # every document ends with <|endoftext|>, so nothing is left at the end of a shard


def doc_hash(text: str) -> int:
    """64-bit hash of the normalized text, stable across processes (unlike Python's hash)."""
    return int.from_bytes(hashlib.blake2b(" ".join(normalize_words(text)).encode(), digest_size=8).digest(), "little")


def source_hashes(data_dir: str | Path, source: str, tokenizer: Tokenizer, n_docs: int,
                  batch_docs: int = 4_096) -> np.ndarray:
    """Hashes of the first n_docs train documents of a source, in order (duplicates kept)."""
    shards = sorted((Path(data_dir) / "tokens" / source).glob("train_*.bin"))
    hashes: list[int] = []
    batch: list[list[int]] = []

    def flush():
        hashes.extend(doc_hash(t) for t in tokenizer.decode_batch(batch, skip_special_tokens=False))
        batch.clear()

    for doc in iter_documents(shards):
        batch.append(doc.tolist())
        if len(hashes) + len(batch) == n_docs:
            break
        if len(batch) == batch_docs:
            flush()
    flush()
    return np.array(hashes, dtype=np.uint64)


def overlap_report(hashes: dict[str, np.ndarray]) -> dict:
    """Per source: documents hashed and exact duplicates inside it. Per pair: distinct documents both share,
    as a share of each source's distinct documents."""
    unique = {s: np.unique(h) for s, h in hashes.items()}
    report = {"sources": {s: {"docs": int(len(h)), "distinct": int(len(unique[s])),
                              "dup_within": round(1 - len(unique[s]) / max(len(h), 1), 6)}
                          for s, h in hashes.items()},
              "pairs": []}
    for a, b in combinations(sorted(unique), 2):
        shared = int(len(np.intersect1d(unique[a], unique[b], assume_unique=True)))
        report["pairs"].append({"a": a, "b": b, "shared": shared,
                                "of_a": round(shared / max(len(unique[a]), 1), 6),
                                "of_b": round(shared / max(len(unique[b]), 1), 6)})
    report["max_pair_share"] = max((max(p["of_a"], p["of_b"]) for p in report["pairs"]), default=0.0)
    return report
