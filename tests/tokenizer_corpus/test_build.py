"""Gate 1: the tokenizer corpus stops at its byte budget, records what it read, and keeps the row filter."""

import pyarrow as pa
import pyarrow.parquet as pq

from slm50m.dataprep.sources import SOURCES
from slm50m.tokenizer_corpus.build import take_part


def test_take_part_stops_at_budget_and_filters(tmp_path):
    omi2 = SOURCES["word_problems"].parts[0]
    for i in range(2):
        pq.write_table(pa.table({
            "problem": [f"p{i}{j}" for j in range(100)],
            "generated_solution": ["s" * 20] * 100,
            "problem_source": ["gsm8k", "math"] * 50,  # half the rows are dropped
        }), tmp_path / f"f{i}.parquet", row_group_size=10)

    files_used: list[dict] = []
    docs = list(take_part(omi2, 3_000, ["f0.parquet", "f1.parquet"], lambda _, f: pa.OSFile(str(tmp_path / f)),
                          files_used))

    assert all(d.startswith("Question: p") and "\nAnswer: s" in d for d in docs)
    assert [f["file"] for f in files_used] == ["f0.parquet", "f1.parquet"]  # f0's 50 kept rows aren't enough
    assert files_used[0]["rows"] == 50
    assert sum(f["bytes"] for f in files_used) == sum(len(d.encode()) for d in docs)
    assert 3_000 <= sum(f["bytes"] for f in files_used) < 3_000 + len(docs[-1].encode())
