"""Held-out arithmetic check: the Orca-Math filter and the distractors (model card, "Held-out arithmetic")."""

import json

from slm50m.evaluation.heldout_arith import ITEMS_FILE, clean_texts, masked_ngrams, op_distractors


def test_masked_ngrams_ignore_numbers():
    a = "Sam has 12 apples and buys 3 more at the shop. How many now?"
    b = "Sam has 40 apples and buys 7 more at the shop. How many now?"
    assert masked_ngrams(a) == masked_ngrams(b)
    assert ("sam", "has", "#", "apples", "and", "buys", "#", "more") in masked_ngrams(a)


def test_clean_texts_drops_paraphrases():
    orca = ["Sam has 40 apples and buys 7 more at the shop. How many apples does he have?"]
    kept = "A train carries 30 people. 12 get off at the first stop. How many are left?"
    para = "Sam has 12 apples and buys 3 more at the shop. How many now?"
    assert clean_texts([kept, para], orca) == {kept}


def test_op_distractors_skip_answer():
    # 12 + 3 = 15: wrong operations give 9, 36, 4 and the operands themselves
    assert op_distractors([12.0, 3.0], 15) == [3, 4, 9, 12, 36]


def test_items_file():
    ids = json.loads(ITEMS_FILE.read_text())
    assert (len(ids["asdiv"]), len(ids["svamp"])) == (224, 8)
