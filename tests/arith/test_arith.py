"""Gate 0: generated answers are right, and nothing overlaps ArithMark-3 (D-011)."""

import random

from slm50m.arith.generator import FORMATS, TEMPLATES
from slm50m.dataprep.decontam import BenchmarkFilter

PER_TEMPLATE = 2_000


def test_answers_are_correct(rng):
    for template, _ in TEMPLATES:
        for _ in range(200):
            p = template(rng)
            for expr, expected in p.checks:
                assert eval(expr) == expected, (template.__name__, expr)
            for fmt in ("sentence", "qa", "steps"):
                assert p.answer in p.write(fmt), (template.__name__, fmt)


def test_no_overlap_with_arithmark(arithmark):
    """Zero exact context matches and zero 13-gram hits, on a large sample from every template and format."""
    contexts = {row["ctx"] for row in arithmark}
    bench = BenchmarkFilter(("arithmark", row["ctx"] + e) for row in arithmark for e in row["endings"])
    rng = random.Random(123)
    for template, _ in TEMPLATES:
        for _ in range(PER_TEMPLATE):
            p = template(rng)
            assert f"{p.stem} {p.lead}" not in contexts
            for fmt in FORMATS:
                text = p.write(fmt)
                assert bench.hit(text) is None, (template.__name__, text)
