"""Synthetic grade 1-5 arithmetic, v2: wide wording for the decay (D-045).

The v1 generator (arith/generator.py) taught the model to compute, but only inside its own wording: 50 names,
a few verbs, one lead phrase per template. On other wording the skill stays off (the ArithMark diagnosis,
27 Oct 2026). v2 keeps the same idea (a Problem written in several formats) and widens every surface:
~500 names plus roles and titled surnames, ~200 things, ~90 gain and loss verbs, ~110 lead phrases, ten answer formats,
worksheet, drill and lesson documents, and place-value working. It covers every ArithMark topic.

Every word list is split into train and held-out parts (arith/lexicon.py). Training text uses only the
train part. The transfer proxy uses only the held-out part, so it measures transfer, not recall.
A slice of the documents still uses v1 problems, so the format the model already knows stays in the data.

Never copy ArithMark wording. tests/arith checks zero exact matches and zero 13-gram hits against it.
"""

import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from slm50m.arith import generator as v1
from slm50m.arith.lexicon import Lexicon

LEXICONS = {split: Lexicon(split) for split in ("train", "heldout")}

# operand sizes: v2 leans toward 2 digits, where ArithMark's grade 1-2 items sit (G6 in the diagnosis)
ADD = {1: 0.25, 2: 0.55, 3: 0.15, 4: 0.05}
SMALL = {1: 0.45, 2: 0.55}

# ---------------------------------------------------------------- text helpers


def num(rng: random.Random, weights: dict[int, float] = ADD, low: int = 1) -> int:
    digits = rng.choices(list(weights), weights=list(weights.values()))[0]
    lo = max(low, 10 ** (digits - 1))
    return rng.randint(lo, max(lo, 10 ** digits - 1))


def plural(word: str) -> str:
    if word.endswith(("s", "x", "ch", "sh")):
        return word + "es"
    if word.endswith("f"):
        return word[:-1] + "ves"
    if word.endswith("y") and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


def count(n: int, thing: tuple[str, str]) -> str:
    return f"{n} {thing[0] if n == 1 else thing[1]}"


def article(word: str) -> str:
    return ("an " if word[0] in "aeiou" else "a ") + word


def cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def clock(minutes: int) -> str:
    h, m = divmod(minutes % (12 * 60), 60)
    return f"{h or 12}:{m:02d}"


PLURAL_SUBJECTS = {"they", "the twins", "the scouts"}


def is_plural(subject: str) -> bool:
    return subject.lower() in PLURAL_SUBJECTS or " and " in subject


def fill(template: str, s: str = "", **kw) -> str:
    """Format a lexicon template. {has}, {had}, {was}, {does} agree with the subject {s}."""
    pl = is_plural(s)
    return cap(template.format(s=s, has="have" if pl else "has", was="were" if pl else "was",
                               does="do" if pl else "does", **kw))


@dataclass
class Agent:
    name: str
    pron: str  # she, he or they

    def ref(self, rng: random.Random) -> str:
        """The name or the pronoun, for a mention after the first."""
        return self.name if rng.random() < 0.4 else self.pron


def person(rng: random.Random, lex: Lexicon, avoid: tuple[Agent, ...] = ()) -> Agent:
    taken = {a.name for a in avoid}
    while True:
        r = rng.random()
        if r < 0.86:
            pool = rng.choices([(lex.female, "she"), (lex.male, "he"), (lex.neutral, "they")], weights=[45, 45, 10])[0]
            a = Agent(rng.choice(pool[0]), pool[1])
        elif r < 0.94:
            name, pron = rng.choice(lex.roles)
            a = Agent(name, pron)
        else:
            pron = rng.choice(["she", "he"])
            a = Agent(f"{rng.choice(v1_titles[pron])} {rng.choice(lex.surnames)}", pron)
        if a.name not in taken:
            return a


v1_titles = {"she": ["Ms.", "Mrs.", "Dr.", "Aunt", "Coach"], "he": ["Mr.", "Dr.", "Uncle", "Coach"]}


def thing(rng: random.Random, lex: Lexicon, kinds: tuple[str, ...] | None = None) -> tuple[str, tuple[str, str]]:
    kind = rng.choice(kinds or tuple(lex.things))
    s, p = rng.choice(lex.things[kind])
    return kind, (s, p or plural(s))


PACK = ("school", "toys", "craft", "food", "nature", "kitchen", "things")  # not animals


def holders(lex: Lexicon) -> list[tuple[str, str]]:
    """Containers you put things in (not a shelf or a bench)."""
    return [c for c in lex.containers if c[1] == "in"]


def tailed(unit: str | tuple[str, str], tail: str) -> Callable[[int], str]:
    """An answer like "107 centimeters long". The tail is fixed per problem, so distractors read the same."""
    if isinstance(unit, str):
        return lambda v: f"{v} {unit}{tail}"
    return lambda v: f"{count(v, unit)}{tail}"


def counted(thing_: tuple[str, str], rng: random.Random, extra: tuple[str, ...] = ()) -> Callable[[int], str]:
    """How the answer is written: mostly "62 beads", sometimes a bare number or a tail like "in all"."""
    r = rng.random()
    if r < 0.82:
        return lambda v: count(v, thing_)
    if r < 0.90:
        return str
    tail = rng.choice(("in all", "altogether", "in total") + extra)
    return lambda v: f"{count(v, thing_)} {tail}"


# ---------------------------------------------------------------- problems


@dataclass
class Problem:
    topic: str
    stem: str  # the situation, as full sentences
    lead: str  # the words the answer completes
    question: str  # the same thing asked as a question
    value: int
    render: Callable[[int], str]  # value -> answer words
    ops: list[tuple[str, int, int, int]]  # (op, a, b, result), op in + - × ÷. The tests recompute them
    steps: list[str] = field(default_factory=list)  # worked lines when ops alone read badly (clock times)

    @property
    def answer(self) -> str:
        return self.render(self.value)


def ask_have(subject: str, x: str, rng: random.Random, tail: str = "now") -> str:
    does = "do" if is_plural(subject) else "does"
    return rng.choice([f"How many {x} {does} {subject} have {tail}?", f"How many {x} {does} {subject} have?",
                       f"What is the number of {x} {subject} {'have' if is_plural(subject) else 'has'} {tail}?"])


ANIMAL_GAIN = ["bought {n} more", "got {n} more", "was given {n} more", "adopted {n} more", "raised {n} more",
               "got {n} more from {o}", "took in {n} more"]
ANIMAL_LOSE = ["sold {n}", "gave away {n}", "gave {n} to {o}", "lent {n} to {o}", "traded away {n}", "let {o} keep {n}"]


def gain_phrase(rng, lex, kind, n, o, x) -> str:
    verb = rng.choice(ANIMAL_GAIN if kind == "animals" else lex.gain + lex.gain_by_kind.get(kind, []))
    text = verb.format(n=n, o=o)
    if verb.endswith(("{n} more", "another {n}")) and rng.random() < 0.5:
        text += f" {x}"
    return text


def lose_phrase(rng, lex, kind, n, o, x) -> str:
    verb = rng.choice(ANIMAL_LOSE if kind == "animals" else lex.lose + lex.lose_by_kind.get(kind, []))
    text = verb.format(n=n, o=o)
    if verb.endswith("{n}"):
        text += rng.choice(["", "", " of them", f" {x}"])
    return text


def time_word(rng, lex) -> str:
    return rng.choice(["Then", "Later", "After that", "The next day", "Soon", "That afternoon", f"On {rng.choice(DAYS)}",
                       rng.choice(lex.times_of_day)])


DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def opening(rng, a: Agent, n: int, t, lex, kind: str = "") -> str:
    c, p = rng.choice(lex.containers)
    if kind == "animals":
        return rng.choice([f"{a.name} had {count(n, t)}.", f"{a.name} has {count(n, t)}.",
                           f"At first, {a.name} had {count(n, t)} on the farm.", f"{a.name} looked after {count(n, t)}."])
    return rng.choice([
        f"{a.name} had {count(n, t)}.", f"{a.name} has {count(n, t)}.", f"{a.name} started with {count(n, t)}.",
        f"{a.name} owned {count(n, t)}.", f"At first, {a.name} had {count(n, t)}.",
        f"{a.name} kept {count(n, t)} {p} {article(c)}.", f"{rng.choice(lex.times_of_day)}, {a.name} had {count(n, t)}.",
        f"{a.name} counted {count(n, t)}.", f"{a.name}'s collection had {count(n, t)}.",
        f"There were {count(n, t)} in {a.name}'s {c}.", f"{a.name} came to {rng.choice(lex.places)} with {count(n, t)}.",
    ])


# --- addition


def add_gain(rng, lex):
    a, (kind, t) = person(rng, lex), thing(rng, lex)
    o = person(rng, lex, (a,)).name
    n1, n2 = num(rng), num(rng)
    s = a.ref(rng)
    event = f"{time_word(rng, lex)}, {s} {gain_phrase(rng, lex, kind, n2, o, t[1])}." if rng.random() < 0.4 else \
        f"{cap(s)} {gain_phrase(rng, lex, kind, n2, o, t[1])}."
    if event.split()[1:2] == ["was"] or " was " in event[:len(s) + 6]:
        event = event.replace(" was ", " were ", 1) if is_plural(s) else event
    s2 = a.ref(rng)
    return Problem("addition", f"{opening(rng, a, n1, t, lex, kind)} {event}", fill(rng.choice(lex.lead_own), s2, x=t[1]),
                   ask_have(s2, t[1], rng), n1 + n2, counted(t, rng), [("+", n1, n2, n1 + n2)])


def add_two(rng, lex):
    a = person(rng, lex)
    b = person(rng, lex, (a,))
    _, t = thing(rng, lex)
    n1, n2 = num(rng), num(rng)
    c, p = rng.choice(lex.containers)
    stem = rng.choice([
        f"{a.name} has {count(n1, t)} and {b.name} has {count(n2, t)}.",
        f"{a.name} collected {count(n1, t)}. {b.name} collected {count(n2, t)}.",
        f"{a.name} brought {count(n1, t)} to {rng.choice(lex.places)}, and {b.name} brought {n2}.",
        f"There are {count(n1, t)} {p} {a.name}'s {c} and {count(n2, t)} {p} {b.name}'s {c}.",
        f"{a.name} made {count(n1, t)}. {b.name} made {count(n2, t)}.",
        f"{a.name} owns {count(n1, t)}, and {b.name} owns {count(n2, t)}.",
        f"{a.name} found {count(n1, t)} and {b.name} found {n2} more.",
        f"{a.name} is holding {count(n1, t)}. {b.name} is holding {count(n2, t)}.",
    ])
    q = rng.choice([f"How many {t[1]} do they have in all?", f"How many {t[1]} are there altogether?",
                    f"How many {t[1]} do {a.name} and {b.name} have together?", f"What is the total number of {t[1]}?"])
    return Problem("addition", stem, fill(rng.choice(lex.lead_both)), q, n1 + n2, counted(t, rng),
                   [("+", n1, n2, n1 + n2)])


def add_box(rng, lex):
    a, (kind, t) = person(rng, lex), thing(rng, lex, PACK)
    c, p = rng.choice(lex.containers)
    n1, n2 = num(rng), num(rng)
    start = rng.choice([f"There were {count(n1, t)} {p} the {c}.", f"{cap(article(c))} held {count(n1, t)}.",
                        f"{a.name}'s {c} had {count(n1, t)} {p} it.", f"The {c} had {count(n1, t)} {p} it.",
                        f"{cap(count(n1, t))} sat {p} the {c}.", f"{a.name} looked {p} the {c} and saw {count(n1, t)}."])
    if rng.random() < 0.75:
        add = fill(rng.choice(lex.put_in), a.ref(rng) if a.name in start else a.name, n=n2, p=p)
    else:
        add = fill(rng.choice(lex.put_in_passive), n=n2, p=p)
    q = rng.choice([f"How many {t[1]} are {p} the {c} now?", f"How many {t[1]} does the {c} hold now?",
                    f"How many are {p} the {c} after that?"])
    return Problem("addition", f"{start} {add}.", fill(rng.choice(lex.lead_box), c=c, p=p), q, n1 + n2,
                   counted(t, rng), [("+", n1, n2, n1 + n2)])


def add_kinds(rng, lex):
    a, (kind, t) = person(rng, lex), thing(rng, lex)
    c1, c2 = rng.sample(lex.colors, 2)
    n1, n2 = num(rng), num(rng)
    stem = rng.choice([f"{a.name} has {n1} {c1} {t[1]} and {n2} {c2} {t[1]}.",
                       f"In {a.name}'s room there are {n1} {c1} {t[1]} and {n2} {c2} ones.",
                       f"{a.name} sorted {t[1]} into two piles: {n1} {c1} and {n2} {c2}.",
                       f"{a.name} bought {n1} {c1} {t[1]} and {n2} {c2} {t[1]}."])
    s = a.ref(rng)
    lead = rng.choice([fill(rng.choice(lex.lead_total)), fill("In all, {s} {has}", s), fill("Altogether {s} {has}", s),
                       fill("Counting both colors, {s} {has}", s)])
    return Problem("addition", stem, lead, f"How many {t[1]} are there in all?", n1 + n2, counted(t, rng),
                   [("+", n1, n2, n1 + n2)])


DAY_VERBS = ["collected", "found", "bought", "got", "gathered", "picked up", "counted", "sold", "gave out", "packed",
             "sorted", "handed out", "made", "used"]


def add_days(rng, lex):
    a, (kind, t) = person(rng, lex), thing(rng, lex)
    d = rng.randrange(6)
    verb = rng.choice(DAY_VERBS)
    n1, n2 = num(rng), num(rng)
    s = a.ref(rng)
    first = rng.choice([f"On {DAYS[d]}, {a.name} {verb} {count(n1, t)}.", f"{a.name} {verb} {count(n1, t)} on {DAYS[d]}.",
                        f"In the morning {a.name} {verb} {count(n1, t)}."])
    second = rng.choice([f"On {DAYS[d + 1]}, {s} {verb} {n2} more.", f"{cap(s)} {verb} {count(n2, t)} the next day.",
                         f"In the afternoon {s} {verb} {count(n2, t)}."])
    lead = rng.choice([f"Over both days, {s} {verb}", f"In all, {s} {verb}", f"Altogether {s} {verb}",
                       f"The two days together come to", f"Added up, {s} {verb}"])
    return Problem("addition", f"{first} {second}", cap(lead), f"How many {t[1]} is that in all?", n1 + n2,
                   counted(t, rng), [("+", n1, n2, n1 + n2)])


def add_crowd(rng, lex):
    noun, arrive, _, where = rng.choice(lex.crowds)
    n1, n2 = num(rng, SMALL), num(rng, SMALL)
    stem = rng.choice([f"There were {n1} {noun} {where}. Then {n2} more {arrive}.",
                       f"{cap(str(n1))} {noun} were {where}. {cap(str(n2))} more {arrive}.",
                       f"At first, {n1} {noun} were {where}. Soon {n2} more {arrive}."])
    lead = rng.choice(["Now there are", f"Now the number of {noun} {where} is", "That makes", f"Now {where} there are",
                       "Altogether there are now", "After that there were"])
    return Problem("addition", stem, cap(lead), f"How many {noun} are there now?", n1 + n2,
                   lambda v: f"{v} {noun}", [("+", n1, n2, n1 + n2)])


# --- subtraction


def sub_lose(rng, lex):
    a, (kind, t) = person(rng, lex), thing(rng, lex)
    o = person(rng, lex, (a,)).name
    n1 = num(rng, low=2)
    n2 = rng.randint(1, n1 - 1)
    s = a.ref(rng)
    act = lose_phrase(rng, lex, kind, n2, o, t[1])
    event = f"{time_word(rng, lex)}, {s} {act}." if rng.random() < 0.4 else f"{cap(s)} {act}."
    s2 = a.ref(rng)
    lead = fill(rng.choice(lex.lead_left), s2)
    q = rng.choice([ask_have(s2, t[1], rng), f"How many {t[1]} are left?", f"How many {t[1]} {fill('{does}', s2).lower()}"
                    f" {s2} have left?"])
    render = counted(t, rng, ("left",)) if not lead.endswith(("only", "kept")) else (lambda v: count(v, t))
    return Problem("subtraction", f"{opening(rng, a, n1, t, lex, kind)} {event}", lead, q, n1 - n2, render,
                   [("-", n1, n2, n1 - n2)])


def sub_box(rng, lex):
    a, (kind, t) = person(rng, lex), thing(rng, lex, PACK)
    o = person(rng, lex, (a,)).name
    c, p = rng.choice(lex.containers)
    c2, _ = rng.choice(lex.containers)
    n1 = num(rng, low=2)
    n2 = rng.randint(1, n1 - 1)
    start = rng.choice([f"There were {count(n1, t)} {p} the {c}.", f"{cap(article(c))} held {count(n1, t)}.",
                        f"The {c} had {count(n1, t)} {p} it.", f"{a.name} filled {article(c)} with {count(n1, t)}."])
    if rng.random() < 0.7:
        take = fill(rng.choice(lex.take_out), a.ref(rng) if a.name in start else a.name, n=n2, c=c, c2=c2, o=o)
    else:
        take = fill(rng.choice(lex.take_out_passive), n=n2)
    return Problem("subtraction", f"{start} {take}.", fill(rng.choice(lex.lead_box), c=c, p=p),
                   f"How many {t[1]} are {p} the {c} now?", n1 - n2, counted(t, rng, ("left",)), [("-", n1, n2, n1 - n2)])


def sub_need(rng, lex):
    a, (kind, t) = person(rng, lex), thing(rng, lex)
    c, p = rng.choice(lex.containers)
    goal = num(rng, low=3)
    have = rng.randint(1, goal - 1)
    s = a.ref(rng)
    stem = rng.choice([f"{a.name} wants to collect {count(goal, t)}. {cap(s)} already {fill('{has}', s).lower()} {have}.",
                       f"{a.name} needs {count(goal, t)} for a project and has {have} so far.",
                       f"{cap(article(c))} fits {count(goal, t)}. Right now it holds {have}.",
                       f"The goal is {count(goal, t)}. {a.name} has gathered {have}."])
    lead = rng.choice([f"{cap(s)} still {'need' if is_plural(s) else 'needs'}", "The number still needed is",
                       f"To reach the goal, {s} must get", "The amount still missing is", f"{cap(s)} must find"])
    if stem.startswith(cap(article(c))):
        lead = rng.choice(["It can still fit", "There is still room for", "The number of empty spots is"])
    return Problem("subtraction", stem, lead, f"How many more {t[1]} are needed?", goal - have,
                   rng.choice([lambda v: count(v, t), lambda v: f"{v} more"]), [("-", goal, have, goal - have)])


def sub_crowd(rng, lex):
    noun, _, leave, where = rng.choice(lex.crowds)
    n1 = num(rng, SMALL, low=3)
    n2 = rng.randint(1, n1 - 1)
    stem = rng.choice([f"There were {n1} {noun} {where}. Then {n2} {leave}.",
                       f"{cap(str(n1))} {noun} were {where}. {cap(str(n2))} of them {leave}.",
                       f"At first, {n1} {noun} were {where}. After a while, {n2} {leave}."])
    lead = rng.choice(["Now there are", "There are still", f"Now {where} there are", "That leaves", "The ones left number",
                       "Still there were"])
    return Problem("subtraction", stem, cap(lead), f"How many {noun} are left?", n1 - n2, lambda v: f"{v} {noun}",
                   [("-", n1, n2, n1 - n2)])


def sub_rest(rng, lex):
    a, (kind, t) = person(rng, lex), thing(rng, lex)
    c1, c2 = rng.sample(lex.colors, 2)
    n1 = num(rng, low=3)
    n2 = rng.randint(1, n1 - 1)
    stem = rng.choice([f"{a.name} has {count(n1, t)}. {n2} of them are {c1} and the rest are {c2}.",
                       f"Of {a.name}'s {count(n1, t)}, {n2} are {c1}. The others are {c2}.",
                       f"A pile has {count(n1, t)}: {n2} {c1} ones and some {c2} ones."])
    lead = rng.choice([f"The number of {c2} {t[1]} is", f"So the {c2} ones number", f"There are this many {c2} ones:",
                       f"The {c2} {t[1]} come to"])
    return Problem("subtraction", stem, lead, f"How many {t[1]} are {c2}?", n1 - n2, lambda v: count(v, t),
                   [("-", n1, n2, n1 - n2)])


# --- comparison


def cmp_more_than(rng, lex):
    a = person(rng, lex)
    b = person(rng, lex, (a,))
    _, t = thing(rng, lex)
    n = num(rng)
    more = rng.random() < 0.6
    d = num(rng, SMALL) if more else rng.randint(1, max(1, n - 1))
    if not more and d >= n:
        n += d
    v = n + d if more else n - d
    word = "more" if more else "fewer"
    stem = rng.choice([f"{a.name} has {count(n, t)}. {b.name} has {d} {word} {t[1]} than {a.name}.",
                       f"{b.name} has {d} {word} {t[1]} than {a.name}, who has {n}.",
                       f"{a.name} collected {count(n, t)}, and {b.name} collected {d} {word} than that."])
    s = b.ref(rng)
    lead = rng.choice([fill("So {s} {has}", s), fill("That means {s} {has}", s), fill("{s} {has}", s),
                       f"The number {b.name} has is"])
    return Problem("comparison", stem, lead, f"How many {t[1]} does {b.name} have?", v, lambda val: count(val, t),
                   [("+" if more else "-", n, d, v)])


def cmp_which(rng, lex):
    a = person(rng, lex)
    b = person(rng, lex, (a,))
    _, t = thing(rng, lex)
    n1, n2 = rng.sample(range(2, 200 if rng.random() < 0.7 else 1000), 2)
    big = rng.random() < 0.6
    v = max(n1, n2) if big else min(n1, n2)
    stem = rng.choice([f"{a.name} has {count(n1, t)}. {b.name} has {count(n2, t)}.",
                       f"{a.name} counted {count(n1, t)} and {b.name} counted {count(n2, t)}.",
                       f"One {rng.choice(lex.containers)[0]} has {count(n1, t)} and another has {count(n2, t)}."])
    lead = rng.choice(["The larger amount is", f"The one with more {t[1]} has", "The bigger number is",
                       "The greater of the two is", "The most anyone has is"] if big else
                      ["The smaller amount is", f"The one with fewer {t[1]} has", "The smaller number is",
                       "The lesser of the two is", "The fewest anyone has is"])
    q = "Which amount is larger?" if big else "Which amount is smaller?"
    op = ("-", max(n1, n2), min(n1, n2), max(n1, n2) - min(n1, n2))
    return Problem("comparison", stem, lead, q, v, lambda val: count(val, t), [op])


def cmp_numbers(rng, lex):
    n1, n2 = rng.sample(range(10, 10_000), 2)
    big = rng.random() < 0.6
    v = max(n1, n2) if big else min(n1, n2)
    stem = rng.choice([f"Compare {n1} and {n2}.", f"Look at the numbers {n1} and {n2}.",
                       f"Two numbers are written on the board: {n1} and {n2}."])
    lead = rng.choice(["The greater number is", "The larger one is", "Of the two, the bigger is"] if big else
                      ["The smaller number is", "The lesser one is", "Of the two, the smaller is"])
    op = ("-", max(n1, n2), min(n1, n2), max(n1, n2) - min(n1, n2))
    return Problem("comparison", stem, lead, "Which number is " + ("greater?" if big else "smaller?"), v, str, [op])


def cmp_difference(rng, lex):
    a = person(rng, lex)
    b = person(rng, lex, (a,))
    _, t = thing(rng, lex)
    n1, n2 = num(rng), num(rng)
    if n1 == n2:
        n1 += rng.randint(1, 9)
    hi, lo = (a, b) if n1 > n2 else (b, a)
    d = abs(n1 - n2)
    stem = rng.choice([f"{a.name} has {count(n1, t)} and {b.name} has {count(n2, t)}.",
                       f"{a.name} scored {n1} points and {b.name} scored {n2} points.",
                       f"{a.name} read {n1} pages. {b.name} read {n2} pages.",
                       f"{a.name} picked {count(n1, t)}. {b.name} picked {count(n2, t)}."])
    unit = t if "points" not in stem and "pages" not in stem else \
        (("point", "points") if "points" in stem else ("page", "pages"))
    kind = rng.random()
    if kind < 0.4:
        lead = fill(rng.choice(lex.lead_diff), a=hi.name, b=lo.name)
        render = (lambda v: f"{v} more {unit[1]}") if lead.endswith(("has", "need")) else (lambda v: count(v, unit))
        q = f"How many more {unit[1]} does {hi.name} have than {lo.name}?"
    elif kind < 0.7:
        lead = rng.choice([f"{lo.name} would need", f"To catch up, {lo.name} needs", f"{lo.name} is short by"])
        render = lambda v: (f"{v} more" if lead.endswith("need") or lead.endswith("needs") else count(v, unit))
        q = f"How many more does {lo.name} need to match {hi.name}?"
    else:
        lead = rng.choice([f"{lo.name} has fewer by", f"Compared with {hi.name}, {lo.name} has", "The gap between them is"])
        render = (lambda v: f"{v} fewer") if lead.endswith("has") else (lambda v: count(v, unit))
        q = f"How many fewer does {lo.name} have?"
    return Problem("comparison_difference", stem, lead, q, d, render, [("-", max(n1, n2), min(n1, n2), d)])


# --- multiplication and division

LEGS = [("spider", "legs", 8), ("dog", "legs", 4), ("bird", "legs", 2), ("ant", "legs", 6), ("cat", "legs", 4),
        ("duck", "legs", 2), ("horse", "legs", 4), ("beetle", "legs", 6), ("octopus", "arms", 8), ("starfish", "arms", 5),
        ("bicycle", "wheels", 2), ("tricycle", "wheels", 3), ("car", "wheels", 4), ("wagon", "wheels", 4),
        ("skateboard", "wheels", 4), ("triangle", "sides", 3), ("square", "sides", 4), ("pentagon", "sides", 5),
        ("hexagon", "sides", 6), ("hand", "fingers", 5), ("week", "days", 7), ("dozen eggs", "eggs", 12),
        ("chair", "legs", 4), ("stool", "legs", 3), ("clover", "leaves", 3), ("table", "legs", 4)]


def mul_groups(rng, lex):
    a, (kind, t) = person(rng, lex), thing(rng, lex, PACK)
    c, p = rng.choice(holders(lex))
    g, k = rng.randint(2, 10), rng.randint(2, 12)
    if rng.random() < 0.2:
        k = rng.randint(11, 25)
    stem = rng.choice([f"{a.name} has {g} {plural(c)} with {count(k, t)} {p} each.",
                       f"There are {g} {plural(c)}. Each {c} has {count(k, t)}.",
                       f"{a.name} filled {g} {plural(c)}, putting {k} {t[1]} {p} each one.",
                       f"Each {c} holds {count(k, t)}, and {a.name} has {g} of them.",
                       f"{a.name} bought {g} packs of {t[1]}. Every pack has {k}."])
    return Problem("multiplication", stem, fill(rng.choice(lex.lead_total)), f"How many {t[1]} are there in all?",
                   g * k, counted(t, rng), [("×", g, k, g * k)])


def mul_rows(rng, lex):
    _, t = thing(rng, lex)
    r, k = rng.randint(2, 12), rng.randint(2, 12)
    place = rng.choice(lex.places)
    stem = rng.choice([f"At {place}, {t[1]} are set out in {r} rows with {k} in each row.",
                       f"There are {r} rows of {t[1]}, and each row has {k}.",
                       f"{cap(t[1])} are lined up in {r} rows of {k}."])
    return Problem("multiplication", stem, fill(rng.choice(lex.lead_total)), f"How many {t[1]} are there?", r * k,
                   lambda v: count(v, t), [("×", r, k, r * k)])


def mul_legs(rng, lex):
    item, part, each = rng.choice(LEGS)
    n = rng.randint(2, 12)
    items = plural(item) if item != "dozen eggs" else "dozens of eggs"
    stem = rng.choice([f"There are {n} {items}. Each {item} has {each} {part}.",
                       f"{person(rng, lex).name} counted {n} {items}. One {item} has {each} {part}.",
                       f"{cap(article(item))} has {each} {part}. Think of {n} {items}."])
    lead = rng.choice([f"Altogether there are", f"The total number of {part} is", f"In all, the {items} have",
                       "That makes", f"Counting every one, there are"])
    return Problem("multiplication", stem, lead, f"How many {part} are there in all?", n * each,
                   lambda v: f"{v} {part}", [("×", n, each, n * each)])


def mul_times_as_many(rng, lex):
    a = person(rng, lex)
    b = person(rng, lex, (a,))
    _, t = thing(rng, lex)
    n, k = rng.randint(2, 15), rng.choice([2, 2, 3, 3, 4, 5, 10])
    word = {2: rng.choice(["twice", "two times", "double"]), 3: rng.choice(["three times", "triple"]),
            4: "four times", 5: "five times", 10: "ten times"}[k]
    as_many = "as many" if word not in ("double", "triple") else "the number of"
    stem = (f"{a.name} has {count(n, t)}. {b.name} has {word} {as_many} {t[1]}." if as_many == "as many" else
            f"{a.name} has {count(n, t)}. {b.name} has {word} that number.")
    s = b.ref(rng)
    return Problem("multiplication", stem, rng.choice([fill("So {s} {has}", s), fill("That means {s} {has}", s),
                                                       f"The number {b.name} has is"]),
                   f"How many {t[1]} does {b.name} have?", n * k, lambda v: count(v, t), [("×", n, k, n * k)])


ROUTINES = [("reads", "pages", "read"), ("swims", "laps", "swam"), ("does", "push-ups", "did"),
            ("drinks", "cups of water", "drank"), ("walks", "blocks", "walked"), ("writes", "words", "wrote"),
            ("bakes", "loaves of bread", "baked"), ("plants", "seeds", "planted"), ("folds", "paper cranes", "folded"),
            ("practices", "songs", "practiced"), ("feeds", "birds", "fed"), ("sells", "cups of lemonade", "sold")]


def mul_routine(rng, lex):
    a = person(rng, lex)
    verb, things, past = rng.choice(ROUTINES)
    k, d = rng.randint(2, 15), rng.randint(2, 9)
    s = a.ref(rng)
    stem = rng.choice([f"{a.name} {verb} {k} {things} every day.", f"Each day, {a.name} {verb} {k} {things}."])
    lead = rng.choice([f"In {d} days, {s} {past}", f"After {d} days, {s} has {past}", f"Over {d} days that adds up to"])
    return Problem("multiplication", stem, cap(lead), f"How many {things} in {d} days?", k * d,
                   lambda v: f"{v} {things}", [("×", k, d, k * d)])


def div_share(rng, lex):
    a, (kind, t) = person(rng, lex), thing(rng, lex)
    k, q = rng.randint(2, 10), rng.randint(2, 12)
    n = k * q
    who = rng.choice(["friends", "children", "classmates", "cousins", "teams", "plates", "bags", "tables", "groups"])
    stem = rng.choice([f"{a.name} shared {count(n, t)} equally among {k} {who}.",
                       f"{cap(count(n, t))} are split evenly between {k} {who}.",
                       f"{a.name} has {count(n, t)} to share fairly with {k} {who}.",
                       f"{a.name} divided {count(n, t)} into {k} equal groups."])
    lead = rng.choice(lex.lead_each) if who in ("friends", "children", "classmates", "cousins", "teams", "groups") else \
        rng.choice([f"Each one of the {who} gets", f"On each of the {who} there are", "Each share is"])
    return Problem("division", stem, cap(lead), f"How many does each get?", q, lambda v: count(v, t), [("÷", n, k, q)])


def div_groups(rng, lex):
    a, (kind, t) = person(rng, lex), thing(rng, lex, PACK)
    c, _ = rng.choice(holders(lex))
    k, q = rng.randint(2, 10), rng.randint(2, 12)
    n = k * q
    stem = rng.choice([f"{a.name} has {count(n, t)} and puts {k} in each {c}.",
                       f"{cap(count(n, t))} are packed into {plural(c)}, {k} to {article(c)}.",
                       f"{a.name} sorts {count(n, t)} into groups of {k}."])
    unit = (c, plural(c)) if "groups" not in stem else ("group", "groups")
    lead = rng.choice([f"That fills", f"The number of {unit[1]} is", f"{a.name} ends up with", "So there are"])
    return Problem("division", stem, lead, f"How many {unit[1]} are there?", q, lambda v: count(v, unit),
                   [("÷", n, k, q)])


def div_leftover(rng, lex):
    a, (kind, t) = person(rng, lex), thing(rng, lex)
    k, q = rng.randint(2, 9), rng.randint(2, 12)
    r = rng.randint(1, k - 1)
    n = k * q + r
    stem = rng.choice([f"{a.name} shares {count(n, t)} equally among {k} friends and keeps the ones left over.",
                       f"{cap(count(n, t))} are put into bags of {k}. Only full bags count."])
    if rng.random() < 0.5:
        lead = rng.choice(["The number left over is", "Left over there are", f"{a.name} keeps"])
        return Problem("division", stem, lead, "How many are left over?", r, lambda v: count(v, t),
                       [("×", q, k, q * k), ("-", n, q * k, r)])
    lead = rng.choice(["Each friend gets", "The number of full bags is"]) if "friends" in stem else \
        "The number of full bags is"
    return Problem("division", stem, lead, "How many does each get?", q, str, [("×", q, k, q * k), ("-", n, q * k, r)])


# --- money (cents)

ITEMS = ["pencil", "eraser", "apple", "sticker", "juice box", "bookmark", "banana", "balloon", "postcard", "marble",
         "notebook", "ruler", "cookie", "pear", "toy car", "ribbon", "candle", "comb", "whistle", "button", "orange",
         "muffin", "pack of seeds", "lollipop", "carton of milk", "yo-yo", "bracelet", "magnet", "crayon box", "badge"]
COINS = [("quarter", 25), ("dime", 10), ("nickel", 5), ("penny", 1)]


def cents(v: int) -> str:
    return f"{v} cents" if v != 1 else "1 cent"


def money_total(rng, lex):
    a = person(rng, lex)
    i1, i2 = rng.sample(ITEMS, 2)
    p1, p2 = rng.randint(5, 95), rng.randint(5, 95)
    stem = rng.choice([f"{cap(article(i1))} costs {cents(p1)} and {article(i2)} costs {cents(p2)}.",
                       f"{a.name} buys {article(i1)} for {cents(p1)} and {article(i2)} for {cents(p2)}.",
                       f"At {rng.choice(lex.places)}, {article(i1)} is {cents(p1)} and {article(i2)} is {cents(p2)}."])
    lead = rng.choice(["Together they cost", "The total cost is", f"{a.name} pays", "Both items come to", "The bill is"])
    return Problem("money", stem, lead, "How much do they cost together?", p1 + p2, cents, [("+", p1, p2, p1 + p2)])


def money_change(rng, lex):
    a = person(rng, lex)
    item = rng.choice(ITEMS)
    paid = rng.choice([50, 100, 100, 100])
    price = rng.randint(5, paid - 1)
    paid_words = {50: rng.choice(["50 cents", "two quarters"]), 100: rng.choice(["one dollar", "a dollar", "100 cents"])}
    stem = rng.choice([f"{a.name} buys {article(item)} for {cents(price)} and pays with {paid_words[paid]}.",
                       f"{cap(article(item))} costs {cents(price)}. {a.name} hands over {paid_words[paid]}."])
    lead = rng.choice([f"{a.name} gets back", "The change is", f"The cashier gives {a.name}", "The change comes to"])
    return Problem("money", stem, lead, "How much change is given?", paid - price, cents, [("-", paid, price, paid - price)])


def money_coins(rng, lex):
    a = person(rng, lex)
    kinds = sorted(rng.sample(COINS, rng.randint(1, 3)), key=lambda c: -c[1])
    counts = [rng.randint(1, 6) for _ in kinds]
    parts = [f"{n} {coin if n == 1 else ('pennies' if coin == 'penny' else coin + 's')}" for (coin, _), n in zip(kinds, counts)]
    listing = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    values = [n * v for (_, v), n in zip(kinds, counts)]
    total = sum(values)
    ops = [("×", n, v, n * v) for (_, v), n in zip(kinds, counts)]
    running = values[0]
    for v in values[1:]:
        ops.append(("+", running, v, running + v))
        running += v
    stem = rng.choice([f"{a.name} has {listing}.", f"In {a.name}'s pocket there are {listing}.",
                       f"{a.name} found {listing} under the couch."])
    lead = rng.choice([f"{a.name} has", "That adds up to", "In all, the coins make", "The coins are worth"])
    return Problem("money", stem, lead, "How much money is that?", total, cents, ops)


def money_several(rng, lex):
    a = person(rng, lex)
    item = rng.choice(ITEMS)
    k, p = rng.randint(2, 9), rng.randint(5, 60)
    stem = rng.choice([f"{a.name} buys {k} {plural(item)} at {cents(p)} each.",
                       f"One {item} costs {cents(p)}. {a.name} wants {k} of them."])
    lead = rng.choice(["The total cost is", f"{a.name} pays", f"{k} of them cost", "That comes to"])
    return Problem("money", stem, lead, "What is the total cost?", k * p, cents, [("×", k, p, k * p)])


def money_save(rng, lex):
    a = person(rng, lex)
    n1 = rng.randint(10, 95)
    if rng.random() < 0.5:
        n2 = rng.randint(5, 95)
        stem = f"{a.name} had {cents(n1)} and earned {cents(n2)} more by {rng.choice(['doing chores', 'selling lemonade', 'walking a dog', 'raking leaves'])}."
        return Problem("money", stem, rng.choice([f"Now {a.name} has", "That makes", "In all there is"]),
                       "How much money is there now?", n1 + n2, cents, [("+", n1, n2, n1 + n2)])
    n2 = rng.randint(1, n1 - 1)
    stem = f"{a.name} had {cents(n1)} and spent {cents(n2)} on {article(rng.choice(ITEMS))}."
    return Problem("money", stem, rng.choice([f"{a.name} has", "That leaves", f"Now {a.name} has", "What is left is"]),
                   "How much money is left?", n1 - n2, cents, [("-", n1, n2, n1 - n2)])


# --- time (minutes)

ACTIVITIES = ["reading", "drawing", "piano practice", "soccer practice", "a bike ride", "baking", "homework", "a nap",
              "a movie", "swim class", "art club", "a walk", "the game", "music class", "the puppet show", "a hike",
              "gardening", "cleaning up", "a spelling test", "chess club", "the bus ride", "a board game"]


def time_elapsed(rng, lex):
    a = person(rng, lex)
    start = rng.randint(7 * 60, 17 * 60)
    length = rng.randint(5, 150) if rng.random() < 0.6 else rng.randint(1, 30) * 5
    end = start + length
    act = rng.choice(ACTIVITIES)
    stem = rng.choice([f"{cap(act)} started at {clock(start)} and ended at {clock(end)}.",
                       f"{a.name} began {act} at {clock(start)} and finished at {clock(end)}.",
                       f"From {clock(start)} to {clock(end)}, {a.name} was busy with {act}."])
    lead = rng.choice([f"{cap(act)} lasted", "That is", "It took", f"{a.name} spent", "The time it took was"])
    return Problem("time", stem, lead, "How long did it take?", length, lambda v: f"{v} minutes",
                   [("-", end, start, length)], [f"{clock(end)} - {clock(start)} = {length} minutes"])


def time_sum(rng, lex):
    a = person(rng, lex)
    act1, act2 = rng.sample(ACTIVITIES, 2)
    m1, m2 = rng.randint(5, 60), rng.randint(5, 60)
    stem = rng.choice([f"{a.name} spent {m1} minutes on {act1} and {m2} minutes on {act2}.",
                       f"{cap(act1)} took {m1} minutes. Then {act2} took {m2} minutes."])
    lead = rng.choice([f"Altogether, {a.name} spent", "In all that is", "The total time is", "Together they took"])
    return Problem("time", stem, lead, "How many minutes in all?", m1 + m2, lambda v: f"{v} minutes",
                   [("+", m1, m2, m1 + m2)])


def time_left(rng, lex):
    a = person(rng, lex)
    total = rng.randint(20, 120)
    done = rng.randint(1, total - 1)
    act = rng.choice(ACTIVITIES)
    stem = rng.choice([f"{cap(act)} lasts {total} minutes. {a.name} has done {done} minutes so far.",
                       f"The plan was {total} minutes of {act}. After {done} minutes, {a.name} took a break."])
    lead = rng.choice(["The time left is", f"{a.name} still has", "There are still", "What remains is"])
    return Problem("time", stem, lead, "How many minutes are left?", total - done, lambda v: f"{v} minutes",
                   [("-", total, done, total - done)])


def time_hours(rng, lex):
    h, m = rng.randint(1, 3), rng.randint(1, 59)
    act = rng.choice(ACTIVITIES)
    stem = rng.choice([f"{cap(act)} took {h} hour{'s' if h > 1 else ''} and {m} minutes.",
                       f"{person(rng, lex).name} spent {h} hour{'s' if h > 1 else ''} and {m} minutes on {act}."])
    lead = rng.choice(["In minutes, that is", "That is the same as", "Written in minutes, it is"])
    return Problem("time", stem, lead, "How many minutes is that?", 60 * h + m, lambda v: f"{v} minutes",
                   [("×", h, 60, 60 * h), ("+", 60 * h, m, 60 * h + m)])


def time_end(rng, lex):
    start = rng.randint(7 * 60, 17 * 60)
    length = rng.randint(5, 95)
    act = rng.choice(ACTIVITIES)
    stem = f"{cap(act)} starts at {clock(start)} and lasts {length} minutes."
    lead = rng.choice(["It ends at", "It is over at", "The clock will read"])
    return Problem("time", stem, lead, "When does it end?", start + length, clock, [("+", start, length, start + length)],
                   [f"{clock(start)} + {length} minutes = {clock(start + length)}"])


def time_daily(rng, lex):
    a = person(rng, lex)
    m, d = rng.randint(5, 30), rng.randint(2, 7)
    act = rng.choice(ACTIVITIES)
    stem = f"{a.name} does {m} minutes of {act} each day for {d} days."
    lead = rng.choice(["In all, that is", f"{a.name} spends", "The total time is"])
    return Problem("time", stem, lead, "How many minutes in all?", m * d, lambda v: f"{v} minutes", [("×", m, d, m * d)])


# --- measurement (centimeters)

LONG_THINGS = ["ribbon", "piece of string", "rope", "stick", "board", "pencil", "crayon", "worm", "scarf", "paper strip",
               "garden hose", "piece of yarn", "snake", "branch", "wire", "chain", "line of blocks", "paper chain"]


def meas_join(rng, lex):
    a = person(rng, lex)
    t1, t2 = rng.sample(LONG_THINGS, 2)
    l1, l2 = rng.randint(3, 99), rng.randint(3, 99)
    stem = rng.choice([f"{a.name} lays {article(t1)} that is {l1} centimeters long next to {article(t2)} that is "
                       f"{l2} centimeters long, end to end.",
                       f"One {t1} is {l1} centimeters. Another is {l2} centimeters. They are joined."])
    lead = rng.choice(["Together they measure", "The total length is", "End to end they reach", "Joined, they are"])
    return Problem("measurement", stem, lead, "How long are they together?", l1 + l2,
                   tailed("centimeters", rng.choice(["", "", " long"])), [("+", l1, l2, l1 + l2)])


def meas_cut(rng, lex):
    a = person(rng, lex)
    t = rng.choice(LONG_THINGS)
    l1 = rng.randint(10, 150)
    l2 = rng.randint(1, l1 - 1)
    stem = rng.choice([f"{a.name} had {article(t)} {l1} centimeters long and cut off {l2} centimeters.",
                       f"{cap(article(t))} was {l1} centimeters long. {a.name} cut {l2} centimeters off it."])
    lead = rng.choice(["Now it is", "What is left measures", "The piece left is", "Now the length is"])
    return Problem("measurement", stem, lead, "How long is it now?", l1 - l2, lambda v: f"{v} centimeters",
                   [("-", l1, l2, l1 - l2)])


def meas_compare(rng, lex):
    a = person(rng, lex)
    b = person(rng, lex, (a,))
    what = rng.choice(["plant", "tower of blocks", "sunflower", "paper chain", "drawing", "snowman", "sandcastle"])
    h1, h2 = rng.sample(range(5, 150), 2)
    hi, lo = (a, b) if h1 > h2 else (b, a)
    stem = f"{a.name}'s {what} is {h1} centimeters tall. {b.name}'s {what} is {h2} centimeters tall."
    lead = rng.choice([f"{hi.name}'s {what} is taller by", "The difference in height is",
                       f"{lo.name}'s {what} is shorter by", "The taller one is ahead by"])
    return Problem("measurement", stem, lead, "How much taller is one than the other?", abs(h1 - h2),
                   lambda v: f"{v} centimeters", [("-", max(h1, h2), min(h1, h2), abs(h1 - h2))])


def meas_grow(rng, lex):
    a = person(rng, lex)
    what = rng.choice(["bean plant", "tomato plant", "sunflower", "puppy", "tree", "icicle", "stack of books"])
    h, g = rng.randint(5, 90), rng.randint(2, 40)
    stem = f"{a.name}'s {what} was {h} centimeters tall. Over the month it grew {g} centimeters."
    lead = rng.choice(["Now it is", "Its height now is", "At the end of the month it measures"])
    return Problem("measurement", stem, lead, "How tall is it now?", h + g, tailed("centimeters", rng.choice(["", " tall"])),
                   [("+", h, g, h + g)])


def meas_convert(rng, lex):
    t = rng.choice(LONG_THINGS)
    if rng.random() < 0.6:
        m = rng.randint(1, 9)
        extra = rng.choice([0, rng.randint(1, 99)])
        stem = f"{cap(article(t))} is {m} meter{'s' if m > 1 else ''}" + (f" and {extra} centimeters long." if extra else " long.")
        ops = [("×", m, 100, 100 * m)] + ([("+", 100 * m, extra, 100 * m + extra)] if extra else [])
        return Problem("measurement", stem, rng.choice(["In centimeters, that is", "That is the same as"]),
                       "How many centimeters is that?", 100 * m + extra, lambda v: f"{v} centimeters", ops)
    cm = rng.randint(2, 30)
    stem = f"{cap(article(t))} is {cm} centimeters long."
    return Problem("measurement", stem, rng.choice(["In millimeters, that is", "That is the same as"]),
                   "How many millimeters is that?", cm * 10, lambda v: f"{v} millimeters", [("×", cm, 10, cm * 10)])


def meas_pieces(rng, lex):
    a = person(rng, lex)
    t = rng.choice(LONG_THINGS)
    k, each = rng.randint(2, 9), rng.randint(2, 20)
    if rng.random() < 0.5:
        stem = f"{a.name} cuts {article(t)} into {k} pieces, each {each} centimeters long."
        return Problem("measurement", stem, rng.choice(["Before cutting, it was", "The whole length was"]),
                       "How long was it?", k * each, lambda v: f"{v} centimeters", [("×", k, each, k * each)])
    stem = f"{a.name} cuts {article(t)} {k * each} centimeters long into {k} equal pieces."
    return Problem("measurement", stem, rng.choice(["Each piece is", "One piece measures"]), "How long is each piece?",
                   each, lambda v: f"{v} centimeters", [("÷", k * each, k, each)])


# --- geometry

SHAPES = ["rectangle drawn on grid paper", "garden", "picture frame", "rug", "sandbox", "poster", "field", "table top",
          "photo", "playground", "pool", "classroom floor", "patio", "flower bed", "board", "window", "pen for goats"]
LENGTH_UNITS = ["units", "units", "units", "centimeters", "meters", "feet", "inches", "yards"]


def peri_rect(rng, lex):
    shape, unit = rng.choice(SHAPES), rng.choice(LENGTH_UNITS)
    l, w = rng.randint(2, 30), rng.randint(1, 20)
    stem = rng.choice([f"A {shape} is {l} {unit} long and {w} {unit} wide.",
                       f"The sides of a rectangular {shape} are {l} {unit} and {w} {unit}.",
                       f"{cap(article(shape))} has length {l} {unit} and width {w} {unit}."])
    lead = rng.choice(["The perimeter is", "The distance around it is", "Around the edge it measures",
                       "Its perimeter comes to", "Walking all the way around is", "The border needed is"])
    ops = [("+", l, w, l + w), ("×", 2, l + w, 2 * (l + w))]
    return Problem("geometry_perimeter", stem, lead, "What is the perimeter?", 2 * (l + w), lambda v: f"{v} {unit}", ops)


def peri_poly(rng, lex):
    unit = rng.choice(LENGTH_UNITS)
    kind = rng.random()
    if kind < 0.35:
        s = rng.randint(2, 25)
        n, name = rng.choice([(4, "square"), (3, "triangle with equal sides"), (5, "pentagon with equal sides"),
                              (6, "hexagon with equal sides"), (8, "octagon with equal sides")])
        stem = f"Each side of a {name} is {s} {unit}."
        ops = [("×", n, s, n * s)]
        v = n * s
    else:
        sides = [rng.randint(2, 30) for _ in range(rng.randint(3, 5))]
        listing = ", ".join(map(str, sides[:-1])) + f" and {sides[-1]}"
        name = {3: "triangle", 4: "four-sided shape", 5: "five-sided shape"}[len(sides)]
        stem = rng.choice([f"A {name} has sides of {listing} {unit}.", f"The sides of a {name} measure {listing} {unit}."])
        ops, run = [], sides[0]
        for s in sides[1:]:
            ops.append(("+", run, s, run + s))
            run += s
        v = run
    lead = rng.choice(["The perimeter is", "The distance around it is", "All the sides add up to", "Its perimeter is"])
    return Problem("geometry_perimeter", stem, lead, "What is the perimeter?", v, lambda val: f"{val} {unit}", ops)


AREA_UNITS = ["square units", "square units", "square centimeters", "square meters", "square feet"]


def area_rect(rng, lex):
    shape, unit = rng.choice(SHAPES), rng.choice(AREA_UNITS)
    l, w = rng.randint(2, 15), rng.randint(2, 12)
    lin = unit.replace("square ", "")
    stem = rng.choice([f"A {shape} is {l} {lin} long and {w} {lin} wide.", f"A rectangle measures {l} {lin} by {w} {lin}.",
                       f"On grid paper, a rectangle covers {l} squares across and {w} squares down."])
    if "grid" in stem:
        unit = "square units"
    lead = rng.choice(["The area is", "It covers", "Its area comes to", "The space inside is"])
    return Problem("geometry_area", stem, lead, "What is the area?", l * w, lambda v: f"{v} {unit}", [("×", l, w, l * w)])


def area_tiles(rng, lex):
    a = person(rng, lex)
    r, c = rng.randint(2, 12), rng.randint(2, 12)
    where = rng.choice(["kitchen floor", "bathroom wall", "patio", "hallway", "table top", "craft board"])
    stem = rng.choice([f"{a.name} covers a {where} with {r} rows of tiles, {c} tiles in each row.",
                       f"A {where} is {r} tiles long and {c} tiles wide."])
    lead = rng.choice(["The number of tiles is", "That takes", f"The {where} uses", "In all, there are"])
    return Problem("geometry_area", stem, lead, "How many tiles are used?", r * c, lambda v: f"{v} tiles",
                   [("×", r, c, r * c)])


def area_two(rng, lex):
    l1, w1, l2, w2 = rng.randint(2, 9), rng.randint(2, 9), rng.randint(2, 9), rng.randint(2, 9)
    stem = (f"A shape is made of two rectangles. One is {l1} units by {w1} units. "
            f"The other is {l2} units by {w2} units.")
    ops = [("×", l1, w1, l1 * w1), ("×", l2, w2, l2 * w2), ("+", l1 * w1, l2 * w2, l1 * w1 + l2 * w2)]
    return Problem("geometry_area", stem, rng.choice(["The total area is", "Together they cover"]),
                   "What is the area of the shape?", l1 * w1 + l2 * w2, lambda v: f"{v} square units", ops)


# --- fractions and pieces

WHOLES = ["pizza", "cake", "pie", "chocolate bar", "sandwich", "loaf of bread", "orange", "pan of brownies",
          "watermelon", "quiche", "tart", "block of cheese", "sheet of paper", "ribbon", "cookie cake"]
FRACTIONS = [(1, 2, ["half", "one half", "1/2"]), (1, 3, ["one third", "a third", "1/3"]),
             (1, 4, ["one fourth", "a quarter", "1/4"]), (2, 3, ["two thirds", "2/3"]), (3, 4, ["three fourths", "3/4"]),
             (1, 5, ["one fifth", "1/5"]), (2, 5, ["two fifths", "2/5"]), (1, 6, ["one sixth", "1/6"]),
             (1, 8, ["one eighth", "1/8"]), (3, 8, ["three eighths", "3/8"]), (1, 10, ["one tenth", "1/10"])]


def frac_eat(rng, lex):
    a = person(rng, lex)
    whole = rng.choice(WHOLES)
    n = rng.choice([4, 6, 8, 10, 12, 16])
    k = rng.randint(1, n - 1)
    verb = rng.choice(["ate", "took", "gave away", "shared", "used", "packed"])
    stem = rng.choice([f"{a.name} cut {article(whole)} into {n} equal pieces and {verb} {k} of them.",
                       f"{cap(article(whole))} was cut into {n} pieces. {a.name} {verb} {k} pieces."])
    lead = rng.choice(["Now there are", "The pieces left number", "That leaves", "What is left is"])
    return Problem("fractions_counting", stem, lead, "How many pieces are left?", n - k,
                   tailed(("piece", "pieces"), rng.choice(["", " left"])), [("-", n, k, n - k)])


def frac_of_set(rng, lex):
    a, (kind, t) = person(rng, lex), thing(rng, lex)
    k, den, words = rng.choice(FRACTIONS)
    total = den * rng.randint(1, 12)
    part_ = total * k // den
    word = rng.choice(words)
    stem = rng.choice([f"{a.name} has {count(total, t)}. {cap(word)} of them are {rng.choice(lex.colors)}.",
                       f"There are {count(total, t)}, and {a.name} gives away {word} of them.",
                       f"{a.name} uses {word} of {count(total, t)}."])
    lead = rng.choice(["That is", "The number is", f"{cap(word)} of {total} is", "So the count is"])
    ops = [("÷", total, den, total // den)] + ([("×", total // den, k, part_)] if k > 1 else [])
    return Problem("fractions_counting", stem, lead, f"How many is {word} of {total}?", part_, lambda v: count(v, t), ops)


def frac_pieces(rng, lex):
    whole = rng.choice(WHOLES)
    k, den, words = rng.choice([f for f in FRACTIONS if f[1] in (2, 3, 4, 5, 6, 8, 10)])
    m = rng.randint(1, 3)
    n = den * m
    word = rng.choice(words)
    if rng.random() < 0.5:
        stem = f"{cap(article(whole))} is cut into {n} equal pieces. {cap(word)} of the {whole} is eaten."
        v = n * k // den
        ops = [("÷", n, den, n // den)] + ([("×", n // den, k, v)] if k > 1 else [])
        lead = rng.choice(["The pieces eaten are", "That is", "The number of pieces eaten is"])
    else:
        w = rng.randint(2, 6)
        stem = f"There are {w} {plural(whole) if whole[-1] != 'h' else whole + 'es'}, each cut into {n} pieces."
        v = w * n
        ops = [("×", w, n, v)]
        lead = rng.choice(["In all there are", "That makes", "Altogether there are"])
    return Problem("fractions_counting", stem, lead, "How many pieces?", v,
                   lambda val: f"{val} piece" + ("" if val == 1 else "s"), ops)


# --- patterns


def pattern(rng, lex):
    kind = rng.choices(["add", "sub", "double", "skip", "missing", "objects"], weights=[30, 18, 8, 14, 15, 15])[0]
    step = rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 25, 50, 100])
    n_terms = rng.randint(3, 6)
    if kind == "double":
        start = rng.randint(1, 12)
        terms = [start * 2 ** i for i in range(n_terms)]
        nxt, op = terms[-1] * 2, ("×", terms[-1], 2, terms[-1] * 2)
    elif kind == "sub":
        start = rng.randint(n_terms * step, n_terms * step + 200)
        terms = [start - i * step for i in range(n_terms)]
        nxt, op = terms[-1] - step, ("-", terms[-1], step, terms[-1] - step)
    else:
        start = rng.randint(0, 200)
        terms = [start + i * step for i in range(n_terms)]
        nxt, op = terms[-1] + step, ("+", terms[-1], step, terms[-1] + step)
    seq = ", ".join(map(str, terms))
    if kind == "missing" and n_terms >= 4:
        i = rng.randint(1, n_terms - 2)
        shown = ", ".join("__" if j == i else str(x) for j, x in enumerate(terms))
        prev = terms[i - 1]
        return Problem("patterns", f"Find the missing number: {shown}.", rng.choice(["The missing number is", "The blank is"]),
                       "What number is missing?", terms[i], str, [("+", prev, step, terms[i])])
    if kind == "skip":
        stem = f"Count by {step}s starting at {start}: {seq}."
    elif kind == "objects":
        rows = rng.choice(["Row 1", "Day 1", "Step 1", "Week 1"])
        label = rows.split()[0]
        word = rng.choice(["chairs", "blocks", "dots", "stars", "beads", "cups"])
        listing = ", ".join(f"{label.lower() if j else label} {j + 1} has {x}" for j, x in enumerate(terms))
        return Problem("patterns", f"{listing}.", f"Following the pattern, {label.lower()} {n_terms + 1} has",
                       "What comes next?", nxt, lambda v: f"{v} {word}", [op])
    else:
        stem = rng.choice([f"Look at the pattern: {seq}.", f"A number pattern goes {seq}.", f"{seq}, ...",
                           f"The numbers {seq} follow a rule."])
    lead = rng.choice(["The next number is", "Next comes", f"The number after {terms[-1]} is", "Continuing the pattern gives",
                       "The pattern continues with"])
    return Problem("patterns", stem, lead, "What number comes next?", nxt, str, [op])


# --- data

SURVEYS = [("favorite color", ["red", "blue", "green", "yellow", "purple", "orange"], "votes"),
           ("favorite fruit", ["apples", "bananas", "grapes", "mangoes", "pears", "cherries"], "votes"),
           ("favorite pet", ["dogs", "cats", "fish", "birds", "rabbits", "turtles"], "votes"),
           ("favorite sport", ["soccer", "basketball", "swimming", "tennis", "running", "baseball"], "votes"),
           ("favorite season", ["spring", "summer", "fall", "winter"], "votes"),
           ("favorite snack", ["popcorn", "pretzels", "crackers", "yogurt", "carrots", "cheese"], "votes"),
           ("ways to get to school", ["bus", "car", "bike", "walking", "train"], "students"),
           ("books read by each group", ["Group A", "Group B", "Group C", "Group D"], "books")]


def data_table(rng, lex):
    topic, labels, unit = rng.choice(SURVEYS)
    labels = rng.sample(labels, rng.randint(3, min(5, len(labels))))
    counts = [rng.randint(2, 40) for _ in labels]
    who = rng.choice(["A class", "Some students", "A club", "The third grade", "A group of friends", "A school"])
    style = rng.random()
    if style < 0.3:
        intro = f"{who} voted on their {topic}" if unit == "votes" else f"{who} counted {topic}"
        shown = f"{intro}: " + ", ".join(f"{l} {n}" for l, n in zip(labels, counts)) + "."
    elif style < 0.55:
        shown = f"{who} made a tally of {topic}.\n" + "\n".join(f"{cap(l)}: {n}" for l, n in zip(labels, counts))
    elif style < 0.75:
        shown = (f"Survey of {topic}\n| Choice | {cap(unit)} |\n" +
                 "\n".join(f"| {cap(l)} | {n} |" for l, n in zip(labels, counts)))
    else:
        shown = f"A bar graph of {topic} shows " + ", ".join(f"{n} for {l}" for l, n in zip(labels, counts)) + "."
    q = rng.random()
    if q < 0.35:
        total, ops, run = sum(counts), [], counts[0]
        for n in counts[1:]:
            ops.append(("+", run, n, run + n))
            run += n
        lead = rng.choice([f"The total number of {unit} is", f"In all there were", f"Altogether the {unit} add up to",
                           f"Counting every choice gives"])
        return Problem("data", shown, lead, f"How many {unit} in all?", total, lambda v: f"{v} {unit}", ops)
    i, j = rng.sample(range(len(labels)), 2)
    if q < 0.7:
        if counts[i] == counts[j]:
            counts[i] += 1
            return data_table(rng, lex)
        hi, lo = (i, j) if counts[i] > counts[j] else (j, i)
        d = counts[hi] - counts[lo]
        lead = rng.choice([f"{cap(labels[hi])} got more {unit} than {labels[lo]} by", f"The difference between {labels[hi]} "
                           f"and {labels[lo]} is", f"{cap(labels[hi])} beat {labels[lo]} by"])
        return Problem("data", shown, lead, f"How many more {unit} did {labels[hi]} get than {labels[lo]}?", d,
                       lambda v: f"{v} {unit}", [("-", counts[hi], counts[lo], d)])
    if q < 0.88:
        lead = rng.choice([f"{cap(labels[i])} and {labels[j]} together got", f"Combined, {labels[i]} and {labels[j]} have"])
        return Problem("data", shown, lead, f"How many {unit} for {labels[i]} and {labels[j]}?", counts[i] + counts[j],
                       lambda v: f"{v} {unit}", [("+", counts[i], counts[j], counts[i] + counts[j])])
    best = max(counts) if rng.random() < 0.5 else min(counts)
    word = "most" if best == max(counts) else "fewest"
    lead = f"The choice with the {word} {unit} got"
    return Problem("data", shown, lead, f"How many {unit} did the choice with the {word} get?", best, lambda v: f"{v} {unit}",
                   [("-", max(counts), min(counts), max(counts) - min(counts))])


# --- two steps


def two_step(rng, lex):
    a, (kind, t) = person(rng, lex), thing(rng, lex)
    o = person(rng, lex, (a,)).name
    n1 = rng.randint(10, 99)
    signs = rng.choice(["++", "--", "+-", "-+"])
    v, ops, sentences = n1, [], []
    for sign in signs:
        if sign == "+":
            k = num(rng, SMALL)
            sentences.append(f"{cap(a.ref(rng))} {gain_phrase(rng, lex, kind, k, o, t[1])}.")
            ops.append(("+", v, k, v + k))
            v += k
        else:
            if v < 2:
                return two_step(rng, lex)
            k = rng.randint(1, v - 1)
            sentences.append(f"{cap(a.ref(rng))} {lose_phrase(rng, lex, kind, k, o, t[1])}.")
            ops.append(("-", v, k, v - k))
            v -= k
    sentences = [s.replace("They was ", "They were ").replace("they was ", "they were ") for s in sentences]
    topic = {"++": "two_step_addition", "--": "two_step_subtraction"}.get(signs, "two_step_add_subtract")
    joiner = rng.choice([" ", " Then ", " Later, "])
    second = sentences[1]
    if joiner != " " and second.split()[0] in ("She", "He", "They", "The", "My", "Our"):
        second = second[0].lower() + second[1:]  # "Then she ..." but never "Then nolan ..."
    body = sentences[0] + joiner + second
    s = a.ref(rng)
    return Problem(topic, f"{opening(rng, a, n1, t, lex, kind)} {body}", fill(rng.choice(lex.lead_own), s, x=t[1]),
                   ask_have(s, t[1], rng, "at the end"), v, counted(t, rng), ops)


def two_step_box(rng, lex):
    a, (kind, t) = person(rng, lex), thing(rng, lex, PACK)
    c, p = rng.choice(lex.containers)
    n1 = rng.randint(10, 99)
    k1 = rng.randint(1, n1 - 1)
    k2 = num(rng, SMALL)
    out = fill(rng.choice(lex.take_out_passive), n=k1)
    put = fill(rng.choice(lex.put_in_passive), n=k2, p=p)
    first, second, ops = (out, put, [("-", n1, k1, n1 - k1), ("+", n1 - k1, k2, n1 - k1 + k2)]) if rng.random() < 0.5 else \
        (put, out, [("+", n1, k2, n1 + k2), ("-", n1 + k2, k1, n1 + k2 - k1)])
    v = n1 - k1 + k2
    stem = f"The {c} had {count(n1, t)} {p} it. {first}. Then {second[0].lower() + second[1:]}."
    return Problem("two_step_add_subtract", stem, fill(rng.choice(lex.lead_box), c=c, p=p),
                   f"How many {t[1]} are {p} the {c} now?", v, counted(t, rng), ops)


def two_step_crowd(rng, lex):
    noun, arrive, leave, where = rng.choice(lex.crowds)
    n1 = rng.randint(10, 60)
    k1 = rng.randint(1, n1 - 1)
    k2 = rng.randint(1, 30)
    stem = f"There were {n1} {noun} {where}. First {k1} {leave}. Then {k2} more {arrive}."
    v = n1 - k1 + k2
    return Problem("two_step_add_subtract", stem, rng.choice(["Now there are", "In the end there were", "That leaves"]),
                   f"How many {noun} are there now?", v, lambda val: f"{val} {noun}",
                   [("-", n1, k1, n1 - k1), ("+", n1 - k1, k2, v)])


def two_step_compare(rng, lex):
    a = person(rng, lex)
    b = person(rng, lex, (a,))
    _, t = thing(rng, lex)
    n, d = num(rng, SMALL), num(rng, SMALL)
    stem = f"{a.name} has {count(n, t)}. {b.name} has {d} more than {a.name}."
    return Problem("two_step_addition", stem, fill(rng.choice(lex.lead_both)), "How many do they have together?",
                   2 * n + d, counted(t, rng), [("+", n, d, n + d), ("+", n, n + d, 2 * n + d)])


def two_step_packs(rng, lex):
    a, (kind, t) = person(rng, lex), thing(rng, lex)
    g, k, e = rng.randint(2, 8), rng.randint(2, 10), rng.randint(1, 20)
    stem = f"{a.name} has {g} packs of {t[1]} with {k} in each pack, plus {count(e, t)} loose."
    return Problem("two_step_addition", stem, fill(rng.choice(lex.lead_own), a.ref(rng), x=t[1]),
                   f"How many {t[1]} in all?", g * k + e, counted(t, rng), [("×", g, k, g * k), ("+", g * k, e, g * k + e)])


# weights follow ArithMark's topic mix, with extra on the topics where the model sat at chance
BUILDERS: list[tuple[Callable[[random.Random, Lexicon], Problem], float]] = [
    (add_gain, 3.0), (add_two, 2.0), (add_box, 2.0), (add_kinds, 1.0), (add_days, 1.0), (add_crowd, 1.0),
    (sub_lose, 3.5), (sub_box, 2.0), (sub_need, 1.0), (sub_crowd, 1.0), (sub_rest, 0.8),
    (cmp_more_than, 1.2), (cmp_which, 1.2), (cmp_numbers, 0.5), (cmp_difference, 2.5),
    (mul_groups, 1.5), (mul_rows, 0.8), (mul_legs, 1.0), (mul_times_as_many, 0.8), (mul_routine, 0.7),
    (div_share, 1.5), (div_groups, 1.0), (div_leftover, 0.7),
    (money_total, 1.0), (money_change, 1.0), (money_coins, 0.8), (money_several, 0.8), (money_save, 0.8),
    (time_elapsed, 1.0), (time_sum, 0.8), (time_left, 0.6), (time_hours, 0.4), (time_end, 0.4), (time_daily, 0.4),
    (meas_join, 1.0), (meas_cut, 0.8), (meas_compare, 0.8), (meas_grow, 0.6), (meas_convert, 0.5), (meas_pieces, 0.5),
    (peri_rect, 1.2), (peri_poly, 1.0), (area_rect, 1.0), (area_tiles, 1.0), (area_two, 0.4),
    (frac_eat, 1.2), (frac_of_set, 1.0), (frac_pieces, 0.8), (pattern, 2.5), (data_table, 2.5),
    (two_step, 3.0), (two_step_box, 1.0), (two_step_crowd, 0.8), (two_step_compare, 0.6), (two_step_packs, 0.6),
]
V1_SHARE = 0.08  # problems still drawn from the v1 generator


def random_problem(rng: random.Random, split: str = "train") -> Problem:
    lex = LEXICONS[split]
    builder = rng.choices([b for b, _ in BUILDERS], weights=[w for _, w in BUILDERS])[0]
    return builder(rng, lex)


# ---------------------------------------------------------------- writing

SIGNS = {"+": ["+"], "-": ["-", "-", "−"], "×": ["×", "x", "*"], "÷": ["÷", "/"]}
OP_WORDS = {
    "+": ["{a} plus {b} is {c}", "{a} plus {b} equals {c}", "{a} and {b} make {c}", "the sum of {a} and {b} is {c}",
          "{b} more than {a} is {c}", "adding {a} and {b} gives {c}", "{a} add {b} is {c}", "{a} and {b} more is {c}"],
    "-": ["{a} minus {b} is {c}", "{a} take away {b} is {c}", "{a} take away {b} leaves {c}",
          "the difference between {a} and {b} is {c}", "{b} less than {a} is {c}", "subtracting {b} from {a} gives {c}",
          "{a} minus {b} equals {c}", "{b} fewer than {a} is {c}"],
    "×": ["{a} times {b} is {c}", "{a} groups of {b} make {c}", "{a} multiplied by {b} is {c}",
          "the product of {a} and {b} is {c}", "{a} lots of {b} is {c}", "{a} times {b} equals {c}"],
    "÷": ["{a} divided by {b} is {c}", "{a} shared by {b} is {c} each", "{a} divided by {b} equals {c}",
          "{a} split into {b} equal groups is {c} in each group"],
}
WORKED = {
    "+": ["We add {a} and {b}: {a} + {b} = {c}.", "Add: {a} + {b} = {c}.", "Put the two amounts together: {a} + {b} = {c}.",
          "To find the total, add {a} and {b}, which gives {c}.", "{a} + {b} = {c}.", "Adding gives {a} + {b} = {c}."],
    "-": ["We subtract: {a} - {b} = {c}.", "Take {b} away from {a}: {a} - {b} = {c}.", "Find the difference: {a} - {b} = {c}.",
          "Subtract {b} from {a} to get {c}.", "{a} - {b} = {c}.", "Taking away gives {a} - {b} = {c}."],
    "×": ["Multiply: {a} × {b} = {c}.", "{a} groups of {b} is {c}.", "We multiply {a} by {b} and get {c}.",
          "{a} × {b} = {c}."],
    "÷": ["Divide: {a} ÷ {b} = {c}.", "Split {a} into {b} equal parts: {a} ÷ {b} = {c}.", "We divide {a} by {b} to get {c}.",
          "{a} ÷ {b} = {c}."],
}


def equation(op: tuple[str, int, int, int], rng) -> str:
    s, a, b, c = op
    return f"{a} {rng.choice(SIGNS[s])} {b} = {c}"


def place_value(op: tuple[str, int, int, int], rng) -> str | None:
    """Tens-and-ones working for 2-digit addition and subtraction, or None."""
    s, a, b, c = op
    if s not in "+-" or not (10 <= a < 100 and 1 <= b < 100):
        return None
    at, ao, bt, bo = a // 10, a % 10, b // 10, b % 10
    if rng.random() < 0.5:  # split into tens and ones
        if s == "+":
            return (f"Split into tens and ones: {at * 10} + {bt * 10} = {(at + bt) * 10} and {ao} + {bo} = {ao + bo}. "
                    f"Then {(at + bt) * 10} + {ao + bo} = {c}.")
        if bt == 0:
            return f"Count back {bo} from {a}: {a} - {bo} = {c}."
        return f"Take away the tens first: {a} - {bt * 10} = {a - bt * 10}. Then take away the ones: {a - bt * 10} - {bo} = {c}."
    if s == "+":
        if ao + bo >= 10:
            return (f"Add the ones: {ao} + {bo} = {ao + bo}. Write {(ao + bo) % 10} and carry 1. "
                    f"Add the tens: {at} + {bt} + 1 = {at + bt + 1}. So {a} + {b} = {c}.")
        return f"Add the ones: {ao} + {bo} = {ao + bo}. Add the tens: {at} + {bt} = {at + bt}. So {a} + {b} = {c}."
    if ao < bo:
        return (f"The ones: {ao} is less than {bo}, so borrow a ten: {ao + 10} - {bo} = {ao + 10 - bo}. "
                f"The tens: {at} - 1 - {bt} = {at - 1 - bt}. So {a} - {b} = {c}.")
    return f"The ones: {ao} - {bo} = {ao - bo}. The tens: {at} - {bt} = {at - bt}. So {a} - {b} = {c}."


FORMATS = {"story": 0.28, "ask_then_say": 0.12, "qa": 0.14, "answer_intro": 0.06, "worked": 0.12, "place_value": 0.05,
           "equation": 0.07, "words": 0.06, "dialogue": 0.05, "blank": 0.05}


SENTENCE_START = re.compile(r"(^|[.?!]\s+|\n)([a-z])")


def sentence_caps(text: str) -> str:
    """Capitalize each sentence start ("my uncle kept" -> "My uncle kept")."""
    return SENTENCE_START.sub(lambda m: m.group(1) + m.group(2).upper(), text)


def write(p: Problem, rng: random.Random, split: str = "train") -> str:
    return sentence_caps(_write(p, rng, split))


def _write(p: Problem, rng: random.Random, split: str) -> str:
    lex = LEXICONS[split]
    sep = "\n" if "\n" in p.stem else " "  # a table or tally ends on its own line
    fmt = rng.choices(list(FORMATS), weights=list(FORMATS.values()))[0]
    final = f"{p.lead} {p.answer}."
    if fmt == "story":
        return f"{p.stem}{sep}{final}"
    if fmt == "ask_then_say":
        return f"{p.stem}{sep}{p.question} {final}"
    if fmt == "qa":
        q, a = rng.choice(lex.qa_labels)
        return f"{q} {p.stem} {p.question}\n{a} {rng.choice([p.answer + '.', final])}"
    if fmt == "answer_intro":
        return f"{p.stem} {p.question} {rng.choice(lex.answer_intros)} {p.answer}."
    if fmt == "dialogue":
        t, s = rng.choice(lex.speakers)
        return f"{t}: {p.stem} {p.question}\n{s}: {rng.choice([final, p.answer + '.'])}"
    if fmt == "blank":
        return f"{p.stem}{sep}{p.lead} ____.\n{rng.choice(['Answer:', 'Fill in:', 'Missing:'])} {p.answer}."
    if fmt == "equation":
        return p.steps[-1] if p.steps else ", ".join(equation(op, rng) for op in p.ops)
    if fmt == "words":
        return " ".join(cap(rng.choice(OP_WORDS[op[0]]).format(a=op[1], b=op[2], c=op[3])) + "." for op in p.ops)
    if fmt == "place_value" and len(p.ops) == 1 and (pv := place_value(p.ops[0], rng)):
        return f"{p.stem} {p.question} {pv} {final}"
    worked = " ".join(p.steps) if p.steps else " ".join(rng.choice(WORKED[op[0]]).format(a=op[1], b=op[2], c=op[3])
                                                        for op in p.ops)
    return f"{p.stem} {p.question} {worked} {final}"


# ---------------------------------------------------------------- documents

LESSONS = {
    "addition": ["When we add, we put groups together to find how many there are in all.",
                 "Adding means joining amounts. The answer is called the sum."],
    "subtraction": ["When we subtract, we take some away and find how many are left.",
                    "Subtraction tells us what remains, or how far apart two amounts are."],
    "multiplication": ["Multiplying is a fast way to add equal groups.", "When every group has the same number, multiply."],
    "division": ["Dividing means sharing into equal groups.", "Division tells us how many are in each group."],
    "money": ["A quarter is 25 cents, a dime is 10 cents, a nickel is 5 cents and a penny is 1 cent.",
              "To find change, subtract the price from the money paid."],
    "time": ["One hour has 60 minutes. To find how long something took, count the minutes from start to end."],
    "measurement": ["One meter is 100 centimeters. Lengths add up when things are laid end to end."],
    "geometry_perimeter": ["The perimeter is the distance around a shape. Add up the lengths of all the sides."],
    "geometry_area": ["Area is the space inside a shape. For a rectangle, multiply the length by the width."],
    "fractions_counting": ["A fraction names equal parts of a whole. Half of something is one of two equal parts."],
    "patterns": ["A number pattern follows a rule. Find the rule, then use it to get the next number."],
    "data": ["A table or chart shows counts. We can add them, compare them, or find the biggest."],
    "comparison": ["To compare two amounts, see which is bigger and by how much."],
    "comparison_difference": ["The difference between two numbers is how much bigger one is than the other."],
}
# topic -> the builders that make it, for lesson pages. A lesson on "comparison" also takes comparison_difference
TOPIC_BUILDERS = {t: [b for b, _ in BUILDERS if b(random.Random(0), LEXICONS["train"]).topic.startswith(t)]
                  for t in LESSONS}
NUMBERING = ["{i}. ", "{i}) ", "Problem {i}: ", "#{i} ", "({i}) ", "", ""]


def problems(rng: random.Random, n: int, split: str) -> list[tuple[Problem | None, str]]:
    out = []
    for _ in range(n):
        if split == "train" and rng.random() < V1_SHARE:
            p1 = v1.random_problem(rng)
            out.append((None, p1.write(rng.choices(list(v1.FORMATS), weights=list(v1.FORMATS.values()))[0])))
        else:
            p = random_problem(rng, split)
            out.append((p, write(p, rng, split)))
    return out


def drill(rng: random.Random) -> str:
    """Bare facts, in symbols or words: a page of sums, differences, products and quotients."""
    lines = []
    for _ in range(rng.randint(6, 20)):
        s = rng.choices("+-×÷", weights=[4, 4, 2, 1])[0]
        if s in "+-":
            a, b = num(rng), num(rng)
            if s == "-" and b > a:
                a, b = b, a
            op = (s, a, b, a + b if s == "+" else a - b)
        else:
            a, b = rng.randint(2, 12), rng.randint(2, 12)
            op = ("×", a, b, a * b) if s == "×" else ("÷", a * b, b, a)
        lines.append(equation(op, rng) if rng.random() < 0.6 else cap(rng.choice(OP_WORDS[op[0]]).format(
            a=op[1], b=op[2], c=op[3])))
    head = rng.choice(["Number facts", "Quick practice", "Facts to know", "Drill", "Mental math", ""])
    joiner = rng.choice(["\n", "\n", "; "])
    return (head + "\n" if head else "") + joiner.join(lines)


def generate_document(rng: random.Random, split: str = "train") -> str:
    """One training document: a worksheet (sometimes with an answer key), a lesson, or a page of drills."""
    lex = LEXICONS[split]
    kind = rng.choices(["worksheet", "answer_key", "lesson", "drill"], weights=[0.62, 0.12, 0.14, 0.12])[0]
    if kind == "drill":
        return drill(rng)
    if kind == "lesson":
        topic = rng.choice(list(LESSONS))
        builders = TOPIC_BUILDERS[topic]
        intro = rng.choice(LESSONS[topic])
        examples = [write(rng.choice(builders)(rng, lex), rng, split) for _ in range(rng.randint(2, 5))]
        return "\n".join([rng.choice(lex.headings), intro, *(f"Example {i}: {e}" for i, e in enumerate(examples, 1))])
    heading = rng.choice(lex.headings + [""])
    numbering = rng.choice(NUMBERING)
    items = problems(rng, rng.randint(3, 12), split)
    if kind == "answer_key":
        qs = [p for p, _ in items if p is not None]
        if len(qs) >= 2:
            body = [numbering.format(i=i) + sentence_caps(p.stem + ("\n" if "\n" in p.stem else " ") + p.question)
                    for i, p in enumerate(qs, 1)]
            key = ", ".join(f"{i}. {p.answer}" for i, p in enumerate(qs, 1))
            return "\n".join(([heading] if heading else []) + body + [f"Answers: {key}."])
    body = [numbering.format(i=i) + text for i, (_, text) in enumerate(items, 1)]
    return rng.choice(["\n", "\n\n"]).join(([heading] if heading else []) + body)


def generate_documents(seed: int, n: int, split: str = "train") -> list[str]:
    rng = random.Random(seed)
    return [generate_document(rng, split) for _ in range(n)]
