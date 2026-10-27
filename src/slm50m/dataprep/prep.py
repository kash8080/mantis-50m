"""Text -> uint16 token shards, with the benchmark filter on, plus manifest entries (D-025, D-036).

On Modal each CPU container runs one parquet file: download, filter, tokenize, write the shard,
delete the parquet. Peak disk is about one parquet file plus its shard.

Layout in the data folder:
  tokens/<source>/train_00000.bin   uint16 token ids, documents back to back, each ends with <|endoftext|>
  tokens/<source>/val.bin           the first `val_docs` kept documents of the source's first file
  tokens/<source>/train_00000.json  that shard's full manifest entry, so a manifest lost in a crash can be rebuilt
  manifests/<source>.json           a copy of the repo's manifests/<source>.json (modal_app.py keeps them in sync)
"""

import json
import os
import random
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from slm50m.arith import generator_v2
from slm50m.arith.generator import generate_document
from slm50m.config import EOS_ID, MANIFEST_DIR
from slm50m.dataprep.decontam import BenchmarkFilter
from slm50m.dataprep.sources import SOURCES, download_parquet, iter_texts


def _write_docs(f, tokenizer: Tokenizer, docs: list[str]) -> int:
    """Tokenize a batch of documents, add <|endoftext|> after each, append to f. Returns tokens written."""
    if not docs:
        return 0
    encoded = tokenizer.encode_batch(docs, add_special_tokens=False)
    ids = np.fromiter((i for enc in encoded for i in (*enc.ids, EOS_ID)), dtype=np.uint16)
    f.write(ids.tobytes())
    return len(ids)


def prep_documents(texts: Iterable[str], tokenizer: Tokenizer, out_path: Path,
                   bench_filter: BenchmarkFilter | None = None, val_path: Path | None = None,
                   val_docs: int = 0, batch_docs: int = 1_024) -> dict:
    """Filter and tokenize documents into out_path. The first val_docs kept documents go to val_path instead.

    Files are written under a .tmp name and renamed at the end, so a crash never leaves a shard that looks done.
    """
    assert tokenizer.get_vocab_size() <= 2 ** 16, "uint16 shards need vocab <= 65,536"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {"docs_kept": 0, "docs_dropped": 0, "tokens_train": 0, "tokens_val": 0, "val_docs": 0}
    tmp_train = out_path.with_suffix(".tmp")
    tmp_val = val_path.with_suffix(".tmp") if val_path else None
    with open(tmp_train, "wb") as train_f, open(tmp_val or os.devnull, "wb") as val_f:
        batch: list[str] = []

        def flush():
            n_val = min(len(batch), val_docs - stats["val_docs"]) if val_path else 0
            stats["tokens_val"] += _write_docs(val_f, tokenizer, batch[:n_val])
            stats["val_docs"] += n_val
            stats["tokens_train"] += _write_docs(train_f, tokenizer, batch[n_val:])
            batch.clear()

        for text in texts:
            if bench_filter is not None and not bench_filter.keep(text):
                stats["docs_dropped"] += 1
                continue
            stats["docs_kept"] += 1
            batch.append(text)
            if len(batch) == batch_docs:
                flush()
        flush()
    os.replace(tmp_train, out_path)
    if tmp_val:
        os.replace(tmp_val, val_path)
    if bench_filter is not None:
        stats["dropped_by_benchmark"] = dict(bench_filter.dropped)
    return stats


def _save_entry(shard: Path, entry: dict) -> dict:
    """Written after the shard, so a shard with this file next to it is finished and traceable to its source file."""
    shard.with_suffix(".json").write_text(json.dumps(entry, indent=2))
    return entry


def shard_path(data_dir: str | Path, source: str, index: int) -> Path:
    return Path(data_dir) / "tokens" / source / f"train_{index:05d}.bin"


def val_path(data_dir: str | Path, source: str) -> Path:
    return Path(data_dir) / "tokens" / source / "val.bin"


def prep_hf_file(source: str, part_index: int, filename: str, shard_index: int, data_dir: str | Path,
                 tokenizer: Tokenizer, bench_filter: BenchmarkFilter, cache_dir: str | Path,
                 val_docs: int = 0) -> dict:
    """One parquet file from the Hub -> one shard. Returns the manifest entry for that file."""
    part = SOURCES[source].parts[part_index]
    # a fresh filter count per file, so the entry only reports this file's drops
    bench_filter.dropped.clear()
    local = download_parquet(part, filename, str(cache_dir))
    try:
        out = shard_path(data_dir, source, shard_index)
        stats = prep_documents(iter_texts(local, part), tokenizer, out, bench_filter,
                               val_path(data_dir, source) if val_docs else None, val_docs)
    finally:
        os.remove(local)
    # hf_hub_download paths look like .../snapshots/<commit>/<filename>: record the commit we really read
    parts = Path(local).parts
    commit = parts[parts.index("snapshots") + 1] if "snapshots" in parts else part.revision
    return _save_entry(out, {"repo": part.repo, "revision": commit, "file": filename,
                             "role": "val+train" if val_docs else "train", "shard": out.name, **stats})


ARITH_GENERATORS = {"arith": ("src/slm50m/arith/generator.py", generate_document, 0),
                    "arith_v2": ("src/slm50m/arith/generator_v2.py", generator_v2.generate_document, 1_000)}


def prep_arith(data_dir: str | Path, tokenizer: Tokenizer, shard_index: int, n_tokens: int, val_docs: int = 0,
               source: str = "arith", bench_filter: BenchmarkFilter | None = None) -> dict:
    """Synthetic arithmetic -> one shard of about n_tokens. Seeded by the shard index (plus an offset per
    generator), so shards never repeat. arith_v2 runs the benchmark filter too (D-045): its wording is wide
    enough that a 13-gram hit is possible, and any hit document is dropped and counted."""
    revision, generate, seed_offset = ARITH_GENERATORS[source]
    rng = random.Random(seed_offset + shard_index)
    if bench_filter is not None:
        bench_filter.dropped.clear()

    def docs():
        written = 0
        while written < n_tokens:
            doc = generate(rng)
            written += len(tokenizer.encode(doc, add_special_tokens=False).ids) + 1  # + <|endoftext|>
            yield doc

    out = shard_path(data_dir, source, shard_index)
    stats = prep_documents(docs(), tokenizer, out, bench_filter, val_path(data_dir, source) if val_docs else None,
                           val_docs)
    return _save_entry(out, {"repo": "generated", "revision": revision,
                             "file": f"seed {shard_index}", "role": "val+train" if val_docs else "train",
                             "shard": out.name, **stats})


def prep_corpus_source(corpus_dir: str | Path, source: str, data_dir: str | Path, tokenizer: Tokenizer,
                       max_bytes: float, val_docs: int) -> dict:
    """The first max_bytes of text of one source in the tokenizer corpus -> train_00000.bin + val.bin.

    For Mac smoke tests only: no download, no benchmark filter, no manifest. Never train a released model on it.
    """
    def texts():
        read = 0
        with open(Path(corpus_dir) / "corpus" / f"{source}.jsonl") as f:
            for line in f:
                if read >= max_bytes:
                    return
                text = json.loads(line)["text"]
                read += len(text.encode())
                yield text

    return prep_documents(texts(), tokenizer, shard_path(data_dir, source, 0), None,
                          val_path(data_dir, source), val_docs)


def shard_index(name: str) -> int:
    """train_00012.bin -> 12"""
    return int(name[6:11])


def merge_entries(*copies: list[dict]) -> list[dict]:
    """One entry per shard from several copies of a manifest. Stops if two copies disagree on a shard's file."""
    merged: dict[str, dict] = {}
    for entries in copies:
        for e in entries:
            old = merged.setdefault(e["shard"], e)
            if old["file"] != e["file"]:
                raise ValueError(f"{e['shard']} is {old['file']} in one manifest and {e['file']} in another")
    return sorted(merged.values(), key=lambda e: e["shard"])


def unlisted_shards(entries: list[dict], shard_names: Iterable[str]) -> list[str]:
    """Shards in the data folder that no manifest entry lists."""
    listed = {e["shard"] for e in entries}
    return sorted(n for n in shard_names if n not in listed)


def read_manifest(source: str, manifest_dir: str | Path = MANIFEST_DIR) -> list[dict]:
    path = Path(manifest_dir) / f"{source}.json"
    return json.loads(path.read_text())["files"] if path.exists() else []


def write_manifest(source: str, entries: list[dict], manifest_dir: str | Path = MANIFEST_DIR) -> Path:
    """manifests/<source>.json: every file we used and what it became. Committed to git (D-036).

    New entries are added to what is already there (same shard name replaces), so an extension only appends.
    """
    path = Path(manifest_dir) / f"{source}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = list({e["shard"]: e for e in read_manifest(source, manifest_dir) + entries}.values())
    totals = {k: sum(e[k] for e in entries) for k in ("docs_kept", "docs_dropped", "tokens_train", "tokens_val")}
    path.write_text(json.dumps({"source": source, "licence": SOURCES[source].licence, "totals": totals,
                                "files": sorted(entries, key=lambda e: e["shard"])}, indent=2))
    return path
