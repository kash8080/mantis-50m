"""Benchmark 13-gram filter (D-025).

Normalize (lowercase, drop punctuation, split on whitespace), then drop any training document that
shares a 13-word n-gram with a benchmark item. Each prep container builds the n-gram table once.
"""

import re
from collections import Counter
from collections.abc import Iterable, Iterator

NGRAM = 13
_PUNCT = re.compile(r"[^\w\s]")


def normalize_words(text: str) -> list[str]:
    return _PUNCT.sub("", text.lower()).split()


def ngrams(words: list[str], n: int = NGRAM) -> Iterator[str]:
    for i in range(len(words) - n + 1):
        yield " ".join(words[i:i + n])


class BenchmarkFilter:
    """Lookup table from benchmark 13-grams to the benchmark they came from."""

    def __init__(self, items: Iterable[tuple[str, str]], n: int = NGRAM):
        # items: (benchmark name, text). Store hashes, not strings, to keep the table small
        self.n = n
        self.table: dict[int, str] = {}
        for bench, text in items:
            for gram in ngrams(normalize_words(text), n):
                self.table.setdefault(hash(gram), bench)
        self.dropped: Counter[str] = Counter()  # documents dropped per benchmark, for the filter report

    def hit(self, text: str) -> str | None:
        """The benchmark this document overlaps with, or None if it is clean."""
        for gram in ngrams(normalize_words(text), self.n):
            bench = self.table.get(hash(gram))
            if bench is not None:
                return bench
        return None

    def keep(self, text: str) -> bool:
        bench = self.hit(text)
        if bench is not None:
            self.dropped[bench] += 1
        return bench is None
