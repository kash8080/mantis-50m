"""Gate 0: the benchmark filter in prep (D-025) and stratified mixing in every batch (D-008)."""

import json

import numpy as np
import pytest

from slm50m.dataprep.decontam import BenchmarkFilter
from slm50m.dataprep.mixture import MixtureLoader, stratified_counts
from slm50m.dataprep.prep import (merge_entries, prep_arith, prep_corpus_source, prep_documents, unlisted_shards,
                                  write_manifest)
from slm50m.tokenizer.train_tokenizer import load_tokenizer
from tests.conftest import SOURCES

# a made-up item in HellaSwag's style (activity label, context, ending)
PLANTED = ("Making a sandwich: A man lays two slices of bread on a wooden board. He spreads butter on each slice "
           "with a knife and then places cheese on top.")


def test_prep_drops_planted_benchmark_text(tmp_path, tokenizer_dir):
    tok = load_tokenizer(tokenizer_dir)
    bench = BenchmarkFilter([("hellaswag", PLANTED)])
    docs = [
        "A clean document about the weather in spring.",
        "Some blog intro. " + PLANTED.upper() + " And then the post goes on.",  # case and context don't hide it
        "Another clean one: 27 + 8 = 35.",
    ]
    out, val = tmp_path / "train_00000.bin", tmp_path / "val.bin"
    stats = prep_documents(docs, tok, out, bench, val_path=val, val_docs=1)

    assert stats["docs_dropped"] == 1 and stats["dropped_by_benchmark"] == {"hellaswag": 1}
    assert stats["docs_kept"] == 2 and stats["val_docs"] == 1
    train_ids = np.fromfile(out, dtype=np.uint16).tolist()
    val_ids = np.fromfile(val, dtype=np.uint16).tolist()
    assert tok.decode(val_ids, skip_special_tokens=False) == docs[0] + "<|endoftext|>"
    assert tok.decode(train_ids, skip_special_tokens=False) == docs[2] + "<|endoftext|>"
    assert stats["tokens_train"] == len(train_ids)


def test_stratified_counts_every_batch():
    weights = np.array([0.45, 0.12, 0.10, 0.05, 0.02, 0.07, 0.04, 0.04, 0.04, 0.07])
    rng = np.random.default_rng(0)
    n, total = 32, np.zeros(len(weights))
    for _ in range(4_000):
        counts = stratified_counts(weights, n, rng)
        assert counts.sum() == n
        assert np.all(counts >= np.floor(weights * n)) and np.all(counts <= np.ceil(weights * n))
        total += counts
    assert np.allclose(total / total.sum(), weights, atol=2e-3)  # exact in expectation, including the 2% source


def test_loader_batches_have_planned_sources(data_dir):
    mix = {"alpha": 0.5, "beta": 0.25, "gamma": 0.25}
    loader = MixtureLoader(data_dir, SOURCES, seed=0)
    windows, counts = loader.batch(mix, step=5, micro=1, n_seqs=8, length=65)
    assert counts == {"alpha": 4, "beta": 2, "gamma": 2}
    # the fixture puts source i's tokens in [100 i, 100 i + 100), so each row names its source
    row_sources = [SOURCES[int(row[0]) // 100] for row in windows]
    assert row_sources == ["alpha"] * 4 + ["beta"] * 2 + ["gamma"] * 2
    assert all(len({int(t) // 100 for t in row}) == 1 for row in windows)
    # a pure function of (seed, step, micro)
    again, _ = MixtureLoader(data_dir, SOURCES, seed=0).batch(mix, step=5, micro=1, n_seqs=8, length=65)
    assert (again == windows).all()


def test_prep_corpus_source_slices_the_tokenizer_corpus(tmp_path, tokenizer_dir):
    """Mac smoke data: first docs to val, the rest to train, stop once max_bytes of text is read."""
    tok = load_tokenizer(tokenizer_dir)
    docs = [f"Document number {i} about apples." for i in range(10)]
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "alpha.jsonl").write_text("".join(json.dumps({"text": d}) + "\n" for d in docs))
    stats = prep_corpus_source(tmp_path, "alpha", tmp_path / "data", tok, max_bytes=len(docs[0]) * 4, val_docs=1)

    assert stats["docs_kept"] == 4 and stats["val_docs"] == 1
    val_ids = np.fromfile(tmp_path / "data" / "tokens" / "alpha" / "val.bin", dtype=np.uint16).tolist()
    train_ids = np.fromfile(tmp_path / "data" / "tokens" / "alpha" / "train_00000.bin", dtype=np.uint16).tolist()
    assert tok.decode(val_ids, skip_special_tokens=False) == docs[0] + "<|endoftext|>"
    assert tok.decode(train_ids, skip_special_tokens=False) == "".join(d + "<|endoftext|>" for d in docs[1:4])


def test_shard_entry_file_rebuilds_the_manifest(tmp_path, tokenizer_dir):
    """Each shard's .json holds its full manifest entry, so a run that died before writing the manifest is
    recoverable (D-036). Manifest copies merge by shard, and a shard no copy lists is found."""
    tok = load_tokenizer(tokenizer_dir)
    entries = [prep_arith(tmp_path, tok, i, n_tokens=200, val_docs=2 if i == 0 else 0) for i in range(3)]
    for e in entries:
        on_disk = json.loads((tmp_path / "tokens" / "arith" / e["shard"]).with_suffix(".json").read_text())
        assert on_disk == e

    mac, volume = entries[:1], entries[:2]  # the Mac's copy is behind, shard 2 is in neither
    merged = merge_entries(mac, volume)
    assert [e["shard"] for e in merged] == ["train_00000.bin", "train_00001.bin"]
    shards = [p.name for p in (tmp_path / "tokens" / "arith").glob("train_*.bin")]
    assert unlisted_shards(merged, shards) == ["train_00002.bin"]

    with pytest.raises(ValueError, match="train_00001.bin"):
        merge_entries(volume, [{**entries[1], "file": "seed 99"}])

    path = write_manifest("arith", merged, tmp_path / "manifests")
    totals = json.loads(path.read_text())["totals"]
    assert totals["tokens_train"] == sum(e["tokens_train"] for e in merged)
