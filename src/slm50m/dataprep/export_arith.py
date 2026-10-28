"""Decode the synthetic arithmetic shards we trained on back to text, as parquet for the HF dataset (D-050).

The shards hold uint16 token ids, documents back to back, each ending with <|endoftext|>. Decoding them gives the
exact training text. Runs in a Modal container next to the data Volume (modal_app.py::upload_arith_dataset), so
nothing big passes through the Mac. Every shard is checked against manifests/<source>.json (documents and tokens),
and a sample of documents must re-encode to the same ids.
"""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from slm50m.config import EOS_ID
from slm50m.dataprep.prep import shard_index, shard_path, val_path

# dataset config name -> (source, shard indices to export, None = all)
CONFIGS = {"v2": ("arith_v2", None), "v1-sample": ("arith", [0])}
CHECK_EVERY = 100  # re-encode every 100th document


def split_docs(ids: np.ndarray) -> list[list[int]]:
    """The documents of a shard, without their <|endoftext|>."""
    ends = np.flatnonzero(ids == EOS_ID)
    if len(ids) and (not len(ends) or ends[-1] != len(ids) - 1):
        raise ValueError("shard does not end with <|endoftext|>")
    starts = np.concatenate([[0], ends[:-1] + 1])
    return [ids[s:e].tolist() for s, e in zip(starts, ends, strict=True)]


def decode_file(path: Path, tokenizer: Tokenizer) -> tuple[list[str], int]:
    """(texts, tokens in the file). Stops if a sampled document does not re-encode to its ids."""
    ids = np.fromfile(path, dtype=np.uint16)
    docs = split_docs(ids)
    texts = tokenizer.decode_batch(docs, skip_special_tokens=False)
    for doc, text in list(zip(docs, texts, strict=True))[::CHECK_EVERY]:
        if tokenizer.encode(text, add_special_tokens=False).ids != doc:
            raise ValueError(f"{path}: a document does not round-trip through the tokenizer")
    return texts, len(ids)


def write(path: Path, texts: list[str], source: str, shard: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"text": texts, "generator": [source] * len(texts),
                      "shard": pa.array([shard] * len(texts), pa.int16()), "doc": pa.array(range(len(texts)), pa.int32())})
    pq.write_table(table, path, compression="zstd", row_group_size=20_000)


def export(data_dir: Path, out_dir: Path, tokenizer: Tokenizer, manifests: dict[str, dict]) -> dict:
    """Write <config>/train-<shard>.parquet and <config>/validation.parquet for every config. Returns counts."""
    summary = {}
    for name, (source, shards) in CONFIGS.items():
        entries = [e for e in manifests[source]["files"] if shards is None or shard_index(e["shard"]) in shards]
        counts = {"source": source, "shards": [], "train_docs": 0, "validation_docs": 0, "train_tokens": 0,
                  "validation_tokens": 0}
        for e in entries:
            index = shard_index(e["shard"])
            n_val = e.get("val_docs", 0)
            texts, tokens = decode_file(shard_path(data_dir, source, index), tokenizer)
            if (len(texts), tokens) != (e["docs_kept"] - n_val, e["tokens_train"]):
                raise ValueError(f"{source} {e['shard']}: {len(texts):,} docs / {tokens:,} tokens on the Volume, "
                                 f"manifest says {e['docs_kept'] - n_val:,} / {e['tokens_train']:,}")
            write(out_dir / name / f"train-{index:05d}.parquet", texts, source, index)
            counts["train_docs"] += len(texts)
            counts["train_tokens"] += tokens
            if n_val:
                val, val_tokens = decode_file(val_path(data_dir, source), tokenizer)
                if (len(val), val_tokens) != (n_val, e["tokens_val"]):
                    raise ValueError(f"{source} val.bin: {len(val):,} docs / {val_tokens:,} tokens, "
                                     f"manifest says {n_val:,} / {e['tokens_val']:,}")
                write(out_dir / name / "validation.parquet", val, source, index)
                counts["validation_docs"] += len(val)
                counts["validation_tokens"] += val_tokens
            counts["shards"].append(e["shard"])
        summary[name] = counts
    return summary
