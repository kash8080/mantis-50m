"""Where each training source lives on the HF Hub and how a parquet row becomes text (D-022).

Mix weights are in config.py (STABLE_MIX, DECAY_MIX). Source names here are the keys there.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import BinaryIO

import pyarrow.parquet as pq


def text_column(row: dict) -> str | None:
    return row["text"]


# Checked 24 Oct 2026. The other values are "math" and "augmented_math", dropped (D-022)
OMI2_GSM8K_SOURCES = {"gsm8k", "augmented_gsm8k"}


def omi2_word_problem(row: dict) -> str | None:
    """OpenMathInstruct-2: keep only the GSM8K-derived rows, written as Question / Answer."""
    if row["problem_source"] not in OMI2_GSM8K_SOURCES:
        return None
    return f"Question: {row['problem']}\nAnswer: {row['generated_solution']}"


def orca_word_problem(row: dict) -> str | None:
    return f"Question: {row['question']}\nAnswer: {row['answer']}"


@dataclass(frozen=True)
class Part:
    """One HF dataset folder. A source can have several parts (word problems has two)."""

    repo: str
    prefix: str  # parquet files under this path in the repo
    columns: tuple[str, ...] = ("text",)
    to_text: Callable[[dict], str | None] = text_column  # None means skip the row
    revision: str | None = None  # HF commit hash (D-022)


@dataclass(frozen=True)
class Source:
    name: str
    licence: str
    parts: tuple[Part, ...] = ()  # empty for the synthetic arithmetic, which is generated


# HF commits pinned 24 Oct 2026 (D-022). Prefixes checked against each repo's file list.
CLIMBMIX_REV = "915333b4f8b8684f39aeaafea600fea6f43fb703"
TXT360_REV = "a86bdfb101ebaaf71b445f95fb0f9b9bc2a47511"
SMOLLM_CORPUS_REV = "3ba9d605774198c5868892d7a8deda78031a781f"
COSMOPEDIA_REV = "0ae6ec63f91742bd2d1eaef4f02232c55d719385"
FINEMATH_REV = "e92b25a616738fe95dc186b64dfb19f9c8525594"
OMI2_REV = "469216e3f46f4dacf476b382e192485ea51a143e"
ORCA_REV = "29255d1770cc4eac66e5e7fa378cba542c026350"

SOURCES: dict[str, Source] = {s.name: s for s in [
    Source("climbmix", "MIT (reshuffle), upstream CC-BY-NC-4.0",
           (Part("karpathy/climbmix-400b-shuffle", "", revision=CLIMBMIX_REV),)),
    Source("txt360_web", "CC-BY-4.0", (Part("IFM/TxT360-v2", "web-high-medium/", revision=TXT360_REV),)),
    Source("txt360_qa", "CC-BY-4.0", (Part("IFM/TxT360-v2", "txt360-qa/", revision=TXT360_REV),)),
    Source("txt360_nltk_qa", "CC-BY-4.0", (Part("IFM/TxT360-v2", "web-high-nltk-qa/", revision=TXT360_REV),)),
    Source("fineweb_edu", "ODC-BY",
           (Part("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup/", revision=SMOLLM_CORPUS_REV),)),
    Source("cosmopedia_v2", "ODC-BY",
           (Part("HuggingFaceTB/smollm-corpus", "cosmopedia-v2/", revision=SMOLLM_CORPUS_REV),)),
    Source("wikihow", "Apache-2.0", (Part("HuggingFaceTB/cosmopedia", "data/wikihow/", revision=COSMOPEDIA_REV),)),
    Source("finemath", "ODC-BY", (Part("HuggingFaceTB/finemath", "finemath-4plus/", revision=FINEMATH_REV),)),
    Source("word_problems", "CC-BY-4.0 and MIT", (
        # "data/train-" is the full split only. data/ also holds train_1M/2M/5M, which are subsets of it
        Part("nvidia/OpenMathInstruct-2", "data/train-", ("problem", "generated_solution", "problem_source"),
             omi2_word_problem, OMI2_REV),
        Part("microsoft/orca-math-word-problems-200k", "data/", ("question", "answer"), orca_word_problem,
             ORCA_REV),
    )),
    Source("arith", "ours (src/slm50m/arith)"),
]}


def list_parquet_files(part: Part) -> list[str]:
    """Parquet files of one part on the Hub, sorted, so "the first N files" is stable."""
    from huggingface_hub import list_repo_files

    files = list_repo_files(part.repo, repo_type="dataset", revision=part.revision)
    return sorted(f for f in files if f.startswith(part.prefix) and f.endswith(".parquet"))


def download_parquet(part: Part, filename: str, cache_dir: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(part.repo, filename, repo_type="dataset", revision=part.revision, cache_dir=cache_dir)


def iter_texts(parquet: str | BinaryIO, part: Part, batch_rows: int = 2_048) -> Iterator[str]:
    """Stream the documents of one parquet file (a local path or an open handle), one row group batch at a time."""
    pf = pq.ParquetFile(parquet)
    for batch in pf.iter_batches(batch_size=batch_rows, columns=list(part.columns)):
        for row in batch.to_pylist():
            text = part.to_text(row)
            if text:
                yield text
