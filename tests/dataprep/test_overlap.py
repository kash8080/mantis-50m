"""Gate 1: the cross-source exact-duplicate check (D-025)."""

import numpy as np

from slm50m.dataprep.overlap import iter_documents, overlap_report, source_hashes
from slm50m.dataprep.prep import prep_documents
from slm50m.tokenizer.train_tokenizer import load_tokenizer

SHARED = "The farmer counts 1,204 sheep before the storm."


def test_overlap_finds_normalized_duplicates_across_sources(tmp_path, tokenizer_dir):
    tok = load_tokenizer(tokenizer_dir)
    docs = {
        "alpha": [SHARED, "Only alpha has this one.", "Alpha again, twice.", "Alpha again, twice."],
        "beta": ["the FARMER counts 1204 sheep before the storm", "Beta's own text.", "More beta."],
    }
    for source, texts in docs.items():
        # a tiny second shard checks that documents run across shard files
        prep_documents(texts[:2], tok, tmp_path / "tokens" / source / "train_00000.bin")
        prep_documents(texts[2:], tok, tmp_path / "tokens" / source / "train_00001.bin")

    shards = sorted((tmp_path / "tokens" / "alpha").glob("train_*.bin"))
    assert [tok.decode(d.tolist()) for d in iter_documents(shards)] == docs["alpha"]

    hashes = {s: source_hashes(tmp_path, s, tok, n_docs=10) for s in docs}
    report = overlap_report(hashes)
    assert report["sources"]["alpha"] == {"docs": 4, "distinct": 3, "dup_within": 0.25}
    (pair,) = report["pairs"]
    assert (pair["a"], pair["b"], pair["shared"]) == ("alpha", "beta", 1)  # case and punctuation don't hide it
    assert pair["of_a"] == round(1 / 3, 6) and pair["of_b"] == round(1 / 3, 6)

    assert len(source_hashes(tmp_path, "alpha", tok, n_docs=2)) == 2
    assert np.array_equal(source_hashes(tmp_path, "beta", tok, n_docs=10), hashes["beta"])  # stable across calls
