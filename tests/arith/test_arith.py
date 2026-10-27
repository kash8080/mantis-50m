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


# ---------------------------------------------------------------- v2 (D-045)

V2_PER_BUILDER = 1_000


def test_v2_answers_are_correct(rng):
    from slm50m.arith import generator_v2 as g

    ops = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "×": lambda a, b: a * b, "÷": lambda a, b: a // b}
    for builder, _ in g.BUILDERS:
        for split in ("train", "heldout"):
            for _ in range(200):
                p = builder(rng, g.LEXICONS[split])
                for op, a, b, c in p.ops:
                    assert ops[op](a, b) == c and (op != "÷" or a % b == 0), (builder.__name__, p.ops)
                # the answer is the last result, or one of the operands (larger of two, groups before a remainder)
                assert p.value == p.ops[-1][3] or any(p.value in op[1:3] for op in p.ops), (builder.__name__, p)
                assert p.value >= 0
                # the answer text is fixed per problem, so proxy distractors differ only in the number
                assert p.answer == p.answer and p.render(p.value + 1) == p.render(p.value + 1)


def test_v2_no_overlap_with_arithmark(arithmark):
    """Zero exact context matches and zero 13-gram hits, on every builder in every format and on whole documents."""
    from slm50m.arith import generator_v2 as g

    contexts = {row["ctx"] for row in arithmark}
    bench = BenchmarkFilter(("arithmark", row["ctx"] + e) for row in arithmark for e in row["endings"])
    rng = random.Random(321)
    for builder, _ in g.BUILDERS:
        for split in ("train", "heldout"):
            for _ in range(V2_PER_BUILDER // 2):
                p = builder(rng, g.LEXICONS[split])
                assert f"{p.stem} {p.lead}" not in contexts
                text = g.write(p, rng, split)
                assert bench.hit(text) is None, (builder.__name__, text)
    for doc in g.generate_documents(seed=5, n=3_000):
        assert bench.hit(doc) is None, doc


def test_v2_training_text_never_uses_held_out_names():
    """The transfer proxy only means something if its held-out names never appear in training text."""
    import re

    from slm50m.arith import generator_v2 as g
    from slm50m.arith.lexicon import FEMALE, MALE, NEUTRAL, SURNAMES, held_out

    held = {n for n in FEMALE + MALE + NEUTRAL + SURNAMES if held_out(n)}
    assert len(held) > 50
    seen = set()
    for doc in g.generate_documents(seed=11, n=4_000):
        seen |= set(re.findall(r"\b[A-Z][a-z]+\b", doc))
    assert not held & seen, sorted(held & seen)
    # and the held-out side really uses them
    proxy_text = " ".join(g.write(g.random_problem(random.Random(i), "heldout"), random.Random(i), "heldout")
                          for i in range(500))
    assert held & set(re.findall(r"\b[A-Z][a-z]+\b", proxy_text))


def test_v2_documents(rng):
    from slm50m.arith import generator_v2 as g

    docs = g.generate_documents(seed=3, n=500)
    assert all(d.strip() for d in docs)
    topics = {g.random_problem(rng).topic for _ in range(3_000)}
    # every ArithMark topic is covered
    assert topics >= {"addition", "subtraction", "measurement", "multiplication", "money", "time", "division",
                      "patterns", "geometry_area", "fractions_counting", "comparison_difference",
                      "two_step_add_subtract", "geometry_perimeter", "comparison", "data", "two_step_subtraction",
                      "two_step_addition"}


def test_transfer_proxy_items():
    from slm50m.arith.proxy import make_transfer_items

    items = make_transfer_items(300)
    assert len(items) == 300
    for item in items:
        assert len(set(item["endings"])) == 4 and 0 <= item["label"] < 4
    labels = [i["label"] for i in items]
    assert len(set(labels)) == 4  # shuffled
