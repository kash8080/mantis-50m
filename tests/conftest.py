"""Shared fixtures: a small trained tokenizer, the ArithMark items, a tiny token data folder."""

import random

import numpy as np
import pytest

from slm50m.arith.generator import generate_documents
from slm50m.dataprep.benchmarks import load_arithmark

ENGLISH = [
    "The quick brown fox jumps over the lazy dog while the farmer counts his 1,204 sheep.",
    "In 1969, astronauts landed on the Moon. It took about 76 hours to get there.",
    "Café owners in Zürich paid €3.50 for a coffee — prices rose by 12% in 2023!",
    "def add(a, b):\n    return a + b\n\n\nprint(add(2, 3))",
    "Photosynthesis turns light, water and carbon dioxide into sugar and oxygen.",
]


@pytest.fixture(scope="session")
def arithmark() -> list[dict]:
    try:
        return load_arithmark()
    except OSError as e:  # first run downloads ~0.8 MB
        pytest.skip(f"ArithMark data not available: {e}")


@pytest.fixture(scope="session")
def tokenizer_dir(tmp_path_factory, arithmark):
    """A 1,024-token tokenizer with our real pipeline, trained in a second on mixed text."""
    from slm50m.tokenizer.train_tokenizer import save_tokenizer, train_tokenizer

    texts = ENGLISH * 20 + generate_documents(seed=0, n=300) + [r["ctx"] + e for r in arithmark[:300] for e in r["endings"]]
    out = tmp_path_factory.mktemp("tokenizer")
    save_tokenizer(train_tokenizer(texts, vocab_size=1_024), out)
    return out


SOURCES = ["alpha", "beta", "gamma"]


@pytest.fixture
def data_dir(tmp_path):
    """tokens/<source>/train_00000.bin + val.bin for three sources. Every token of source i is in its own
    id range [100 i, 100 i + 100), so a test can tell which source a window came from."""
    rng = np.random.default_rng(0)
    for i, source in enumerate(SOURCES):
        d = tmp_path / "tokens" / source
        d.mkdir(parents=True)
        for name, n in (("train_00000.bin", 50_000), ("train_00001.bin", 20_000), ("val.bin", 5_000)):
            rng.integers(100 * i, 100 * i + 100, n, dtype=np.uint16).tofile(d / name)
    return tmp_path


@pytest.fixture
def rng():
    return random.Random(0)
