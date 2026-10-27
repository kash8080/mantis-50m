"""Synthetic grade 1-5 arithmetic in our own wording (D-011, D-024).

Each template builds one Problem from a seeded RNG. A Problem can be written in four formats
(continuation sentence, Q&A, equation, worked steps), and the same Problem also gives a 4-choice
item for the proxy eval (arith/proxy.py). Documents bundle 5-15 problems.

Never copy ArithMark wording. tests/arith checks zero exact matches and zero 13-gram hits against it.
"""

import random
from collections.abc import Callable
from dataclasses import dataclass, field

# ---------------------------------------------------------------- word lists

NAMES = [
    "Aria", "Ben", "Chloe", "Dev", "Elena", "Farah", "Gus", "Hana", "Isaac", "Jonah", "Kira", "Luis", "Mila",
    "Noor", "Omar", "Priya", "Quinn", "Rosa", "Sanjay", "Tess", "Umar", "Vera", "Wes", "Ximena", "Yusuf", "Zara",
    "Amir", "Bella", "Carlos", "Dina", "Ezra", "Fiona", "Gita", "Hugo", "Ines", "Jamal", "Keiko", "Lena", "Marco",
    "Nadia", "Oscar", "Pablo", "Rhea", "Soren", "Tariq", "Uma", "Viktor", "Wren", "Yara", "Zeke",
]
# (singular, plural)
THINGS = [
    ("apple", "apples"), ("crayon", "crayons"), ("sticker", "stickers"), ("seashell", "seashells"),
    ("pencil", "pencils"), ("card", "cards"), ("stamp", "stamps"), ("bead", "beads"), ("cookie", "cookies"),
    ("muffin", "muffins"), ("rock", "rocks"), ("coin", "coins"), ("toy car", "toy cars"), ("book", "books"),
    ("acorn", "acorns"), ("feather", "feathers"), ("grape", "grapes"), ("pebble", "pebbles"), ("cup", "cups"),
    ("brick", "bricks"), ("tomato", "tomatoes"), ("peach", "peaches"), ("ticket", "tickets"), ("block", "blocks"),
    ("puzzle piece", "puzzle pieces"), ("paper clip", "paper clips"), ("carrot", "carrots"), ("egg", "eggs"),
]
CONTAINERS = ["box", "basket", "jar", "crate", "drawer", "bin", "tray", "bucket"]
PLACES = ["the library", "the farm stand", "the bakery", "the school fair", "the toy shop", "the garden club"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
HEADINGS = ["Practice problems", "Math warm-up", "Word problems", "Number practice", "Quick sums",
            "Arithmetic drills", "Homework check", ""]

# ---------------------------------------------------------------- numbers


def number(rng: random.Random, digit_weights: dict[int, float], low: int = 1) -> int:
    """A number whose digit count is drawn from digit_weights (weighted toward the easy end)."""
    digits = rng.choices(list(digit_weights), weights=list(digit_weights.values()))[0]
    return rng.randint(max(low, 10 ** (digits - 1)), 10 ** digits - 1)


ADD_DIGITS = {1: 0.30, 2: 0.45, 3: 0.18, 4: 0.07}


def count(n: int, thing: tuple[str, str]) -> str:
    return f"{n} {thing[0] if n == 1 else thing[1]}"


def dollars(cents: int) -> str:
    return f"${cents // 100}.{cents % 100:02d}"


def clock(minutes: int) -> str:
    """Minutes after midnight -> 12-hour clock time like 3:05."""
    h, m = divmod(minutes % (12 * 60), 60)
    return f"{h or 12}:{m:02d}"


# ---------------------------------------------------------------- problems


@dataclass
class Problem:
    topic: str
    stem: str  # the situation, as full sentences
    lead: str  # the words the answer completes ("Now the jar holds")
    question: str  # the same thing asked as a question
    value: int  # the answer as a number (used for near-miss distractors)
    render: Callable[[int], str]  # value -> answer words ("35 jars")
    steps: list[str]  # worked equations, e.g. ["27 + 8 = 35"]
    checks: list[tuple[str, int]] = field(default_factory=list)  # python expressions the tests evaluate

    @property
    def answer(self) -> str:
        return self.render(self.value)

    def write(self, fmt: str) -> str:
        if fmt == "sentence":
            return f"{self.stem} {self.lead} {self.answer}."
        if fmt == "qa":
            return f"Question: {self.stem} {self.question}\nAnswer: {self.answer}."
        if fmt == "equation":
            return self.steps[-1]
        if fmt == "steps":
            worked = " ".join(f"Step {i}: {s}." for i, s in enumerate(self.steps, 1))
            return f"{self.stem} {self.question}\n{worked} {self.lead} {self.answer}."
        raise ValueError(fmt)


def mul_sign(rng):
    return rng.choice(["×", "x", "*"])


def div_sign(rng):
    return rng.choice(["÷", "/"])


# Each template: rng -> Problem. Grouped by topic, same topics as ArithMark (D-024)


def add_into_container(rng):
    name, thing, box = rng.choice(NAMES), rng.choice(THINGS), rng.choice(CONTAINERS)
    a, b = number(rng, ADD_DIGITS), number(rng, ADD_DIGITS)
    c = a + b
    return Problem("addition", f"A {box} held {count(a, thing)}. {name} put {b} more into the {box}.",
                   f"Now the {box} holds", f"How many {thing[1]} are in the {box} now?",
                   c, lambda v: count(v, thing), [f"{a} + {b} = {c}"], [(f"{a} + {b}", c)])


def add_two_people(rng):
    n1, n2 = rng.sample(NAMES, 2)
    thing = rng.choice(THINGS)
    a, b = number(rng, ADD_DIGITS), number(rng, ADD_DIGITS)
    c = a + b
    return Problem("addition", f"{n1} collected {count(a, thing)} and {n2} collected {count(b, thing)}.",
                   f"Between them, {n1} and {n2} collected", f"How many {thing[1]} did they collect in all?",
                   c, lambda v: count(v, thing), [f"{a} + {b} = {c}"], [(f"{a} + {b}", c)])


def add_two_days(rng):
    place, thing = rng.choice(PLACES), rng.choice(THINGS)
    d = rng.randrange(6)
    a, b = number(rng, ADD_DIGITS), number(rng, ADD_DIGITS)
    c = a + b
    return Problem("addition", f"On {DAYS[d]}, {place} gave out {count(a, thing)}. On {DAYS[d + 1]}, it gave out {b}.",
                   "Over the two days the total came to", f"How many {thing[1]} were given out over both days?",
                   c, lambda v: count(v, thing), [f"{a} + {b} = {c}"], [(f"{a} + {b}", c)])


def sub_hand_out(rng):
    name, thing = rng.choice(NAMES), rng.choice(THINGS)
    a = number(rng, ADD_DIGITS, low=2)
    b = rng.randint(1, a)
    c = a - b
    return Problem("subtraction", f"{name} began the week with {count(a, thing)} and handed {b} of them to classmates.",
                   f"The number {name} kept was", f"How many {thing[1]} did {name} keep?",
                   c, lambda v: count(v, thing), [f"{a} - {b} = {c}"], [(f"{a} - {b}", c)])


def sub_still_needs(rng):
    name, thing = rng.choice(NAMES), rng.choice(THINGS)
    goal = number(rng, ADD_DIGITS, low=2)
    have = rng.randint(1, goal)
    c = goal - have
    return Problem("subtraction", f"{name} wants a set of {count(goal, thing)} and owns {have} so far.",
                   f"To finish the set, {name} needs", f"How many more {thing[1]} does {name} need?",
                   c, lambda v: f"{v} more", [f"{goal} - {have} = {c}"], [(f"{goal} - {have}", c)])


def sub_taken_away(rng):
    thing, box = rng.choice(THINGS), rng.choice(CONTAINERS)
    a = number(rng, ADD_DIGITS, low=2)
    b = rng.randint(1, a)
    c = a - b
    return Problem("subtraction", f"There were {count(a, thing)} in a {box}, and {b} were removed.",
                   f"The {box} was left with", f"How many {thing[1]} remained in the {box}?",
                   c, lambda v: count(v, thing), [f"{a} - {b} = {c}"], [(f"{a} - {b}", c)])


def mul_groups(rng):
    thing, box = rng.choice(THINGS), rng.choice(CONTAINERS)
    kind = rng.choices(["1x1", "2x1", "2x2"], weights=[0.6, 0.3, 0.1])[0]
    a = rng.randint(2, 12) if kind != "2x2" else rng.randint(10, 99)
    b = rng.randint(2, 9) if kind == "1x1" else rng.randint(10, 99) if kind == "2x2" else rng.randint(10, 30)
    c = a * b
    s = mul_sign(rng)
    return Problem("multiplication", f"Each {box} holds {count(b, thing)}, and there are {a} of them.",
                   f"All {a} together hold", f"How many {thing[1]} are there in total?",
                   c, lambda v: count(v, thing), [f"{a} {s} {b} = {c}"], [(f"{a} * {b}", c)])


def mul_rows(rng):
    a, b = rng.randint(2, 12), rng.randint(2, 15)
    c = a * b
    s = mul_sign(rng)
    seat = ("chair", "chairs")
    return Problem("multiplication", f"A hall is set up with {a} rows and {count(b, seat)} in each row.",
                   "The hall has", "How many chairs are set up?",
                   c, lambda v: count(v, seat), [f"{a} {s} {b} = {c}"], [(f"{a} * {b}", c)])


def div_share(rng):
    name, thing = rng.choice(NAMES), rng.choice(THINGS)
    b = rng.randint(2, 9)
    q = number(rng, {1: 0.5, 2: 0.4, 3: 0.1})
    a = b * q
    s = div_sign(rng)
    return Problem("division", f"{name} split {count(a, thing)} evenly among {b} friends.",
                   "Every friend received", "How many did every friend receive?",
                   q, lambda v: count(v, thing), [f"{a} {s} {b} = {q}"], [(f"{a} // {b}", q), (f"{a} % {b}", 0)])


def div_remainder(rng):
    thing = rng.choice(THINGS)
    b = rng.randint(2, 9)
    q = number(rng, {1: 0.6, 2: 0.4})
    r = rng.randint(1, b - 1)
    a = b * q + r
    return Problem("division", f"A pile of {count(a, thing)} is packed into bags of {b}.",
                   "That makes", "How many full bags are there, and how many are left over?",
                   q, lambda v: f"{v} full bags with {r} left over",
                   [f"{a} ÷ {b} = {q} remainder {r}"], [(f"{a} // {b}", q), (f"{a} % {b}", r)])


def money_change(rng):
    name = rng.choice(NAMES)
    bill = rng.choice([1, 5, 10, 20])
    price = rng.randint(5, bill * 100 - 1)
    change = bill * 100 - price
    return Problem("money", f"{name} paid for a {dollars(price)} lunch with a {bill}-dollar bill.",
                   f"The cashier gave {name}", "How much change did the cashier give back?",
                   change, lambda v: f"{dollars(v)} back", [f"{dollars(bill * 100)} - {dollars(price)} = {dollars(change)}"],
                   [(f"{bill * 100} - {price}", change)])


def money_cents_total(rng):
    a, b = rng.randint(5, 95), rng.randint(5, 95)
    item1, item2 = rng.sample(["a pencil", "an eraser", "a ruler", "a bookmark", "a balloon", "a sticker sheet"], 2)
    c = a + b
    return Problem("money", f"At the school store, {item1} costs {a} cents and {item2} costs {b} cents.",
                   "Buying both costs", "How much do the two items cost together?",
                   c, lambda v: f"{v} cents", [f"{a} + {b} = {c}"], [(f"{a} + {b}", c)])


def money_several(rng):
    name, k = rng.choice(NAMES), rng.randint(2, 9)
    price = rng.randint(25, 450)
    c = k * price
    return Problem("money", f"{name} bought {k} notebooks priced at {dollars(price)} apiece.",
                   "The bill came to", "What was the total cost?",
                   c, dollars, [f"{k} × {dollars(price)} = {dollars(c)}"], [(f"{k} * {price}", c)])


def time_end(rng):
    start = rng.randint(6 * 12, 10 * 12) * 5  # between 6:00 and 10:00, on a 5 minute mark
    length = rng.choice([15, 20, 25, 30, 35, 40, 45, 50, 55, 70, 85, 90, 105])
    end = start + length
    event = rng.choice(["A piano lesson", "A soccer practice", "A cartoon", "A science class", "A bus trip"])
    return Problem("time", f"{event} begins at {clock(start)} and goes on for {length} minutes.",
                   "It finishes at", "At what time does it finish?",
                   end, clock, [f"{clock(start)} + {length} minutes = {clock(end)}"], [(f"{start} + {length}", end)])


def time_length(rng):
    start = rng.randint(7 * 12, 11 * 12) * 5
    length = rng.randint(2, 11) * 5
    end = start + length
    return Problem("time", f"The art club met from {clock(start)} to {clock(end)}.",
                   "The meeting lasted", "How long did the meeting last?",
                   length, lambda v: f"{v} minutes", [f"{clock(end)} - {clock(start)} = {length} minutes"],
                   [(f"{end} - {start}", length)])


# (big unit, small unit, factor, things measured in that unit)
CONVERSIONS = [
    ("meters", "centimeters", 100, ["The rope", "The garden path", "The hallway"]),
    ("kilograms", "grams", 1000, ["The bag of flour", "The pumpkin", "The box of nails"]),
    ("hours", "minutes", 60, ["The road trip", "The hike", "The baking contest"]),
    ("feet", "inches", 12, ["The bookshelf", "The kite string", "The wooden plank"]),
    ("liters", "milliliters", 1000, ["The fish tank", "The water jug", "The soup pot"]),
    ("yards", "feet", 3, ["The ribbon", "The swimming lane", "The roll of fabric"]),
    ("weeks", "days", 7, ["The summer camp", "The science project", "The school break"]),
]


def measure_convert(rng):
    big, small, factor, things = rng.choice(CONVERSIONS)
    n = rng.randint(2, 12)
    c = n * factor
    thing = rng.choice(things)
    return Problem("measurement", f"{thing} measured {n} {big}.",
                   f"Written in {small}, that is", f"How many {small} is that?",
                   c, lambda v: f"{v} {small}", [f"{n} × {factor} = {c}"], [(f"{n} * {factor}", c)])


def measure_join(rng):
    a, b = rng.randint(5, 150), rng.randint(5, 150)
    c = a + b
    item = rng.choice(["boards", "strips of tape", "pieces of yarn", "train tracks"])
    return Problem("measurement", f"Two {item} measure {a} centimeters and {b} centimeters.",
                   "Placed end to end, they reach", "What is their combined length?",
                   c, lambda v: f"{v} centimeters", [f"{a} + {b} = {c}"], [(f"{a} + {b}", c)])


def perimeter(rng):
    if rng.random() < 0.3:
        s = rng.randint(2, 25)
        p = 4 * s
        return Problem("perimeter", f"A square tile has sides of {s} inches.",
                       "Around its edge, the tile measures", "What is the perimeter of the tile?",
                       p, lambda v: f"{v} inches", [f"4 × {s} = {p}"], [(f"4 * {s}", p)])
    length, width = rng.randint(3, 40), rng.randint(2, 30)
    p = 2 * (length + width)
    return Problem("perimeter", f"A fence goes around a rectangular yard {length} meters long and {width} meters wide.",
                   "The fence is", "How long is the fence?",
                   p, lambda v: f"{v} meters long", [f"{length} + {width} = {length + width}", f"2 × {length + width} = {p}"],
                   [(f"2 * ({length} + {width})", p)])


def area(rng):
    length, width = rng.randint(2, 20), rng.randint(2, 15)
    a = length * width
    thing = rng.choice(["rug", "poster", "garden bed", "table top", "classroom floor"])
    return Problem("area", f"A {thing} is {length} feet by {width} feet.",
                   f"The {thing} covers", f"What is the area of the {thing}?",
                   a, lambda v: f"{v} square feet", [f"{length} × {width} = {a}"], [(f"{length} * {width}", a)])


FRACTIONS = [(1, 2, "half"), (1, 3, "one third"), (1, 4, "one fourth"), (1, 5, "one fifth"),
             (2, 3, "two thirds"), (3, 4, "three fourths"), (1, 10, "one tenth")]


def fraction_of(rng):
    k, den, words = rng.choice(FRACTIONS)
    total = den * rng.randint(2, 20)
    part = total * k // den
    group = rng.choice(["students in a class", "pages in a book", "seeds in a packet", "marbles in a bag"])
    steps = [f"{total} ÷ {den} = {total // den}"] + ([f"{total // den} × {k} = {part}"] if k > 1 else [])
    return Problem("fractions", f"There are {total} {group}, and {words} of them are blue.",
                   "The blue ones number", "How many of them are blue?",
                   part, str, steps,
                   [(f"{total} * {k} // {den}", part), (f"{total} * {k} % {den}", 0)])


def compare_gap(rng):
    n1, n2 = rng.sample(NAMES, 2)
    a, b = sorted([number(rng, ADD_DIGITS), number(rng, ADD_DIGITS)], reverse=True)
    if a == b:
        a += 1
    d = a - b
    return Problem("comparison", f"{n1} scored {a} points in a game and {n2} scored {b} points.",
                   "The gap between their scores was", f"How many more points did {n1} score than {n2}?",
                   d, lambda v: f"{v} points", [f"{a} - {b} = {d}"], [(f"{a} - {b}", d)])


def compare_larger(rng):
    a, b = rng.sample(range(10, 10_000), 2)
    big = max(a, b)
    return Problem("comparison", f"Two numbers are {a} and {b}.",
                   "Of the two, the greater one is", "Which of the two numbers is greater?",
                   big, str, [f"{big} > {min(a, b)}"], [(f"max({a}, {b})", big)])


def pattern(rng):
    kind = rng.choices(["add", "sub", "double"], weights=[0.6, 0.25, 0.15])[0]
    if kind == "double":
        start, terms = rng.randint(1, 12), []
        for i in range(4):
            terms.append(start * 2 ** i)
        nxt, rule, expr = terms[-1] * 2, "doubles each time", f"{terms[-1]} * 2"
    else:
        stepsize = rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 25, 50, 100])
        if kind == "add":
            start = rng.randint(0, 200)
        else:
            start = rng.randint(4 * stepsize, 4 * stepsize + 300)
            stepsize = -stepsize
        terms = [start + i * stepsize for i in range(4)]
        nxt = terms[-1] + stepsize
        rule = f"goes up by {stepsize}" if stepsize > 0 else f"goes down by {-stepsize}"
        expr = f"{terms[-1]} + {stepsize}" if stepsize > 0 else f"{terms[-1]} - {-stepsize}"
    seq = ", ".join(map(str, terms))
    return Problem("patterns", f"Look at the sequence {seq}. It {rule}.",
                   "The term after that is", "What comes next?",
                   nxt, str, [f"{expr.replace('*', '×')} = {nxt}"], [(expr, nxt)])


def data_table(rng):
    labels = rng.sample(["apples", "bananas", "grapes", "mangoes", "pears", "plums", "kiwis"], 3)
    counts = [rng.randint(2, 60) for _ in labels]
    listing = ", ".join(f"{label} {n}" for label, n in zip(labels, counts))
    if rng.random() < 0.5:
        total = sum(counts)
        return Problem("data", f"A tally of favorite fruits shows {listing}.",
                       "Counting every vote gives", "How many votes were counted in all?",
                       total, lambda v: f"{v} votes", [f"{' + '.join(map(str, counts))} = {total}"],
                       [(" + ".join(map(str, counts)), total)])
    i, j = sorted(rng.sample(range(3), 2), key=lambda k: -counts[k])
    if counts[i] == counts[j]:
        counts[i] += 1
        listing = ", ".join(f"{label} {n}" for label, n in zip(labels, counts))
    d = counts[i] - counts[j]
    return Problem("data", f"A tally of favorite fruits shows {listing}.",
                   f"Votes for {labels[i]} beat votes for {labels[j]} by a margin of",
                   f"How many more votes did {labels[i]} get than {labels[j]}?",
                   d, str, [f"{counts[i]} - {counts[j]} = {d}"], [(f"{counts[i]} - {counts[j]}", d)])


def two_step_find_lose(rng):
    name, thing = rng.choice(NAMES), rng.choice(THINGS)
    a, b = number(rng, {1: 0.4, 2: 0.6}), number(rng, {1: 0.5, 2: 0.5})
    c = rng.randint(1, a + b)
    mid, end = a + b, a + b - c
    return Problem("two_step", f"{name} owned {count(a, thing)}, got {b} more as a gift, and later dropped {c}.",
                   f"In the end {name} owned", f"How many {thing[1]} does {name} own at the end?",
                   end, lambda v: count(v, thing), [f"{a} + {b} = {mid}", f"{mid} - {c} = {end}"],
                   [(f"{a} + {b} - {c}", end)])


def two_step_bus(rng):
    a = rng.randint(20, 60)
    b = rng.randint(1, a // 2)
    c = rng.randint(1, a - b)
    mid, end = a - b, a - b - c
    rider = ("rider", "riders")
    return Problem("two_step", f"A bus left the depot with {a} riders. At the first stop {b} got off, and at the "
                   f"second stop {c} got off.", "Still on board were", "How many riders were still on the bus?",
                   end, lambda v: count(v, rider), [f"{a} - {b} = {mid}", f"{mid} - {c} = {end}"],
                   [(f"{a} - {b} - {c}", end)])


def two_step_shopping(rng):
    name, k = rng.choice(NAMES), rng.randint(2, 6)
    price = rng.randint(10, 150)
    bill = next(b for b in (1, 2, 5, 10, 20) if b * 100 > k * price)
    cost, change = k * price, bill * 100 - k * price
    return Problem("two_step", f"{name} bought {k} stickers at {price} cents each and paid with {bill} dollars.",
                   f"{name} got back", f"How much change did {name} get?",
                   change, lambda v: f"{v} cents", [f"{k} × {price} = {cost}", f"{bill * 100} - {cost} = {change}"],
                   [(f"{bill} * 100 - {k} * {price}", change)])


TEMPLATES: list[tuple[Callable[[random.Random], Problem], float]] = [
    (add_into_container, 1.0), (add_two_people, 1.0), (add_two_days, 0.8),
    (sub_hand_out, 1.0), (sub_still_needs, 0.8), (sub_taken_away, 1.0),
    (mul_groups, 1.0), (mul_rows, 0.7), (div_share, 1.0), (div_remainder, 0.5),
    (money_change, 0.8), (money_cents_total, 0.6), (money_several, 0.5),
    (time_end, 0.8), (time_length, 0.6), (measure_convert, 0.8), (measure_join, 0.6),
    (perimeter, 0.8), (area, 0.8), (fraction_of, 0.8),
    (compare_gap, 0.8), (compare_larger, 0.4), (pattern, 0.9), (data_table, 0.7),
    (two_step_find_lose, 0.7), (two_step_bus, 0.5), (two_step_shopping, 0.5),
]
FORMATS = {"sentence": 0.40, "qa": 0.25, "equation": 0.20, "steps": 0.15}


def random_problem(rng: random.Random) -> Problem:
    template = rng.choices([t for t, _ in TEMPLATES], weights=[w for _, w in TEMPLATES])[0]
    return template(rng)


def generate_document(rng: random.Random) -> str:
    """One training document: an optional heading and 5-15 problems in mixed formats."""
    heading = rng.choice(HEADINGS)
    lines = [heading] if heading else []
    for _ in range(rng.randint(5, 15)):
        fmt = rng.choices(list(FORMATS), weights=list(FORMATS.values()))[0]
        lines.append(random_problem(rng).write(fmt))
    return rng.choice(["\n", "\n\n"]).join(lines)


def generate_documents(seed: int, n: int) -> list[str]:
    rng = random.Random(seed)
    return [generate_document(rng) for _ in range(n)]
