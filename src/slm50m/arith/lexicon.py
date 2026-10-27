"""Word lists for the v2 arithmetic generator (D-045), split into train and held-out parts.

About 1 in 5 entries of every list is held out (a stable crc32 of the entry decides). The training
shards never use held-out entries. The transfer proxy (arith/proxy.py) builds its items only from them,
so it measures whether the skill fires on names, things, verbs and phrases the model never saw in
arithmetic text. Entries in CORE are everyday words that must be trained, so they are never held out.

Written from general English, never from ArithMark items (D-011). tests/arith checks the 13-gram rule.
"""

import zlib
from dataclasses import dataclass

from slm50m.arith import generator as v1

HELD_OUT_EVERY = 5

# ---------------------------------------------------------------- people

FEMALE = """
Aaliyah Abby Abigail Ada Adaeze Adele Adriana Agnes Aisha Aiko Alice Alicia Alina Allison Alma Amaka Amara Amber
Amelia Amina Amira Amy Ana Anaya Andrea Angela Anika Anita Anna Annie Aria Ariana Ariel Asha Astrid Aubrey Audrey
Aurora Ava Ayesha Bea Bella Bethany Bianca Bonnie Brianna Bridget Brooke Camila Carla Carmen Caroline Cassie Catalina
Cecilia Celeste Charlotte Chiara Chioma Chloe Claire Clara Cora Daisy Dalia Daniela Daphne Deepa Delia Diana Dina
Divya Dora Eden Edith Elena Eliza Ella Ellie Eloise Elsa Emily Emma Erin Esme Esther Eva Evelyn Fatima Fatou Faye
Felicity Fiona Flora Frida Gabriela Gemma Georgia Gia Gita Gloria Grace Greta Gwen Hailey Hana Hannah Harper Hazel
Heidi Helen Hina Holly Hope Ida Imani Ines Ingrid Irene Iris Isabel Isla Ivy Jada Jade Jana Jane Jasmine Jenna
Jessica Jia Joanna Josephine Joy Julia Juliet June Kaia Kalani Kamala Kara Kate Katie Kavya Keiko Kendra Kiara Kim
Kira Laila Lana Laura Layla Leah Leila Lena Lila Lily Lina Linnea Liv Lola Lorena Lucia Lucy Luisa Luna Lydia Mabel
Madison Maeve Maggie Maisie Malia Mara Maria Mariam Marina Marisol Martha Maya Megan Mei Melody Mia Mila Mina Miriam
Molly Mona Nadia Nala Naomi Natalia Nell Nia Nina Noor Nora Olga Olive Olivia Paige Paloma Pari Pearl Penny Phoebe
Pooja Priya Rachel Rania Rebecca Rhea Riya Rosa Rose Rosie Ruby Ruth Sabrina Sadie Sakura Salma Samira Sana Sara
Sarah Savannah Selena Serena Shanti Simone Sofia Sophie Stella Sunita Susan Talia Tamar Tara Tess Thea Tina Trinity
Uma Valentina Vera Victoria Violet Vivian Wanda Wendy Willa Xia Yara Yasmin Yuki Yuna Zainab Zara Zoe Zuri
""".split()

MALE = """
Aarav Aaron Abdul Adam Adrian Ahmed Aidan Akira Alan Albert Alejandro Alex Ali Amir Amos Andre Andrew Angelo Anton
Arjun Arthur Asher Ben Benjamin Bilal Blake Bo Brandon Bruno Caleb Calvin Carlos Carter Cesar Chen Chidi Chris
Christian Colin Connor Cyrus Daniel Dante Darius David Dev Diego Dmitri Dominic Duncan Dylan Eddie Edgar Eli Elijah
Emeka Emil Emmanuel Enzo Eric Ethan Evan Ezra Felix Femi Finn Frank Gabriel George Gideon Gus Hamza Hank Harry Hassan
Hector Henry Hiro Hugo Ian Ibrahim Igor Isaac Ivan Jack Jacob Jake Jamal James Jared Jason Javier Jay Jonah Jordan
Jorge Jose Joseph Joshua Juan Julian Kai Kamal Karim Karl Kenji Kevin Khalid Kofi Kwame Lars Leo Leon Levi Liam
Lorenzo Louis Luca Lucas Luis Luke Malik Marco Marcus Mario Mark Martin Mason Mateo Matthew Max Miguel Milo Mohammed
Musa Nathan Ned Neil Nico Nikhil Noah Nolan Obi Oliver Omar Oscar Owen Pablo Paolo Patrick Paul Pedro Peter Rafael
Rahim Raj Ravi Ray Reza Ricardo Riley Rohan Roman Ross Ruben Ryan Sam Samuel Santiago Saul Sean Sebastian Seth Simon
Soren Stefan Sunil Tariq Theo Thomas Tim Tobias Tom Tomas Tyler Umar Victor Vikram Vincent Viktor Wes Will William
Xavier Yosef Yusuf Zach Zeke Zion
""".split()

NEUTRAL = """
Alexis Ash Avery Bailey Cameron Casey Charlie Dakota Drew Eden Emery Finley Frankie Hayden Jesse Jamie Jules Kendall
Kris Lee Logan Morgan Parker Peyton Quinn Reese Remy Robin Rowan Sage Sasha Skyler Taylor Wren
""".split()

SURNAMES = """
Abara Adams Ahmed Alvarez Anand Bailey Baker Banerjee Bell Bennett Brooks Brown Campbell Carter Castillo Chen Chow
Clarke Cohen Costa Cruz Das Diaz Dubois Edwards Evans Fischer Flores Ford Garcia Gomez Gray Green Gupta Hall Hansen
Harris Hayes Hill Ito Jackson James Jensen Johnson Jones Kang Kaur Kelly Khan Kim King Kowalski Kumar Larsen Lee Lewis
Lim Lopez Martin Mendez Meyer Miller Mitchell Moore Morales Murphy Nakamura Nguyen Novak Okafor Okoye Olsen Ortiz Owens
Park Patel Perez Petrov Phillips Price Quinn Ramirez Reed Reyes Rivera Roberts Rossi Russo Sanchez Santos Sato Schmidt
Scott Shah Silva Singh Smith Suzuki Tan Taylor Thomas Torres Turner Wagner Walker Wang Ward Watson White Williams Wilson
Wong Wright Yamamoto Young Zhang
""".split()

# (phrase, pronoun) for people met by role. "they" roles take plural verbs
ROLES = [
    ("the teacher", "she"), ("the coach", "he"), ("the baker", "he"), ("a farmer", "she"), ("the librarian", "she"),
    ("Grandma", "she"), ("Grandpa", "he"), ("my sister", "she"), ("my brother", "he"), ("our neighbor", "he"),
    ("the shopkeeper", "she"), ("a gardener", "he"), ("my cousin", "she"), ("Mom", "she"), ("Dad", "he"),
    ("the art teacher", "he"), ("the class", "they"), ("the team", "they"), ("the twins", "they"),
    ("the scouts", "they"), ("the club", "they"), ("my aunt", "she"), ("my uncle", "he"), ("the chef", "she"),
]
TITLES = {"she": ["Ms.", "Mrs.", "Dr.", "Aunt", "Coach"], "he": ["Mr.", "Dr.", "Uncle", "Coach"]}

# ---------------------------------------------------------------- things
# Each entry: singular, plural (blank: add "s"). Grouped by what fits them, so sentences stay sensible

THINGS: dict[str, list[tuple[str, str]]] = {
    "school": [("pencil", ""), ("crayon", ""), ("marker", ""), ("eraser", ""), ("sticker", ""), ("notebook", ""),
               ("ruler", ""), ("paper clip", ""), ("glue stick", ""), ("folder", ""), ("pen", ""),
               ("highlighter", ""), ("index card", ""), ("sheet of paper", "sheets of paper"), ("book", ""),
               ("bookmark", ""), ("piece of chalk", "pieces of chalk"), ("sharpener", ""), ("worksheet", ""),
               ("flash card", ""), ("map", ""), ("calculator", "")],
    "toys": [("marble", ""), ("block", ""), ("toy car", ""), ("doll", ""), ("puzzle piece", ""), ("ball", ""),
             ("yo-yo", ""), ("kite", ""), ("domino", "dominoes"), ("game token", ""), ("trading card", ""),
             ("die", "dice"), ("toy robot", ""), ("teddy bear", ""), ("action figure", ""), ("balloon", ""),
             ("toy train", ""), ("puppet", ""), ("spinning top", ""), ("rubber duck", ""), ("board game", ""),
             ("jigsaw puzzle", ""), ("toy truck", ""), ("bouncy ball", "")],
    "craft": [("bead", ""), ("button", ""), ("ribbon", ""), ("pom-pom", ""), ("sequin", ""), ("feather", ""),
              ("paper star", ""), ("rubber band", ""), ("stamp", ""), ("pipe cleaner", ""), ("magnet", ""),
              ("badge", ""), ("pin", ""), ("paper crane", ""), ("bracelet", ""), ("necklace", ""), ("card", ""),
              ("paper flower", ""), ("clay pot", ""), ("painted rock", ""), ("friendship band", ""),
              ("paper plane", ""), ("spool of thread", "spools of thread"), ("patch", "patches")],
    "food": [("apple", ""), ("orange", ""), ("banana", ""), ("strawberry", "strawberries"),
             ("cherry", "cherries"), ("cookie", ""), ("muffin", ""), ("cupcake", ""), ("pancake", ""),
             ("sandwich", "sandwiches"), ("carrot", ""), ("tomato", "tomatoes"), ("potato", "potatoes"),
             ("egg", ""), ("lemon", ""), ("peach", "peaches"), ("pear", ""), ("plum", ""), ("bagel", ""),
             ("cracker", ""), ("pretzel", ""), ("donut", ""), ("roll", ""), ("walnut", ""), ("almond", ""),
             ("pepper", ""), ("onion", ""), ("cucumber", ""), ("mango", "mangoes"), ("kiwi", ""), ("taco", ""),
             ("dumpling", ""), ("waffle", ""), ("biscuit", ""), ("grape", ""), ("blueberry", "blueberries"),
             ("pie", ""), ("brownie", ""), ("granola bar", ""), ("lime", "")],
    "nature": [("seashell", ""), ("shell", ""), ("rock", ""), ("stone", ""), ("pebble", ""), ("leaf", "leaves"),
               ("acorn", ""), ("pinecone", ""), ("flower", ""), ("stick", ""), ("seed", ""), ("twig", ""),
               ("daisy", "daisies"), ("tulip", ""), ("rose", ""), ("sunflower", ""), ("chestnut", ""),
               ("crystal", ""), ("fossil", ""), ("snowball", ""), ("sea glass piece", "sea glass pieces")],
    "animals": [("chicken", ""), ("duck", ""), ("cow", ""), ("sheep", "sheep"), ("goat", ""), ("horse", ""),
                ("pig", ""), ("rabbit", ""), ("goldfish", "goldfish"), ("frog", ""), ("bird", ""),
                ("ladybug", ""), ("butterfly", "butterflies"), ("puppy", "puppies"), ("kitten", ""),
                ("turtle", ""), ("hamster", ""), ("lamb", ""), ("pony", "ponies"), ("goose", "geese")],
    "kitchen": [("cup", ""), ("plate", ""), ("bowl", ""), ("spoon", ""), ("fork", ""), ("jar", ""), ("napkin", ""),
                ("glass", "glasses"), ("mug", ""), ("pot", ""), ("pan", ""), ("candle", ""), ("bottle", ""),
                ("can", ""), ("lid", ""), ("straw", ""), ("towel", "")],
    "things": [("coin", ""), ("postcard", ""), ("letter", ""), ("photo", ""), ("key", ""), ("ticket", ""),
               ("token", ""), ("invitation", ""), ("flag", ""), ("poster", ""), ("tile", ""), ("brick", ""),
               ("board", ""), ("nail", ""), ("screw", ""), ("battery", "batteries"), ("sock", ""), ("hat", ""),
               ("glove", ""), ("shirt", ""), ("blanket", ""), ("pillow", ""), ("comic", ""), ("magazine", ""),
               ("envelope", ""), ("light bulb", ""), ("flower pot", ""), ("bucket", ""), ("chair", ""),
               ("lantern", ""), ("umbrella", ""), ("wreath", "wreaths"), ("medal", ""), ("trophy", "trophies"),
               ("raffle ticket", ""), ("cone", ""), ("jump rope", ""), ("hula hoop", ""), ("sticker book", "")],
}
COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "white", "black", "brown", "gold", "silver",
          "striped", "spotted", "small", "large", "round", "shiny", "wooden", "plastic", "glass", "paper", "new", "old"]

# where things are kept: (name, preposition)
CONTAINERS = [("box", "in"), ("bag", "in"), ("basket", "in"), ("jar", "in"), ("bin", "in"), ("crate", "in"),
              ("bucket", "in"), ("tray", "on"), ("bowl", "in"), ("drawer", "in"), ("shelf", "on"), ("sack", "in"),
              ("tin", "in"), ("case", "in"), ("chest", "in"), ("carton", "in"), ("pouch", "in"), ("cup", "in"),
              ("backpack", "in"), ("pail", "in"), ("barrel", "in"), ("cart", "in"), ("wagon", "in"),
              ("cupboard", "in"), ("table", "on"), ("desk", "on"), ("windowsill", "on"), ("bench", "on"),
              ("cooler", "in"), ("toy chest", "in"), ("pencil case", "in"), ("lunch box", "in")]
PLACES = ["the park", "the library", "the market", "the school fair", "the beach", "the zoo", "the bakery",
          "the farm", "the garden", "the museum", "the toy store", "the craft fair", "the bake sale", "the playground",
          "the swimming pool", "the pet shop", "the post office", "the book fair", "the art room", "the gym",
          "the bus stop", "the camp", "the pond", "the classroom", "the lunchroom", "the hallway", "the yard sale"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TIMES_OF_DAY = ["In the morning", "Before lunch", "After school", "In the afternoon", "That evening", "At recess",
                "Before dinner", "Later that day", "Early on", "At first", "Last week", "Yesterday", "This morning"]

# groups that arrive or leave: (plural noun, arrive phrase, leave phrase, where)
CROWDS = [
    ("birds", "flew down", "flew away", "on the fence"), ("children", "came over", "went home", "at the park"),
    ("kids", "joined in", "left", "in the game"), ("ducks", "swam over", "swam off", "on the pond"),
    ("people", "got on", "got off", "on the bus"), ("cars", "pulled in", "drove away", "in the parking lot"),
    ("players", "ran onto the field", "sat down on the bench", "on the field"),
    ("students", "walked in", "walked out", "in the classroom"), ("frogs", "hopped over", "hopped away", "by the pond"),
    ("ants", "marched in", "marched off", "on the picnic blanket"), ("bees", "buzzed in", "flew off", "in the garden"),
    ("fish", "swam in", "swam away", "near the rock"), ("guests", "arrived", "went home", "at the party"),
    ("runners", "joined the race", "dropped out", "in the race"), ("boats", "sailed in", "sailed out", "in the harbor"),
    ("squirrels", "ran up", "ran off", "in the tree"), ("shoppers", "came in", "went out", "in the store"),
]

# ---------------------------------------------------------------- verbs
# {n} number, {x} things (plural), {o} another person, {c} the container

GAIN = [
    "found {n} more", "got {n} more", "picked up {n} more", "bought {n} more", "received {n} more",
    "collected {n} more", "added {n} more", "brought home {n} more", "won {n} more", "got {n} more from {o}",
    "was given {n} more", "traded for {n} more", "saved up for {n} more", "gathered {n} more", "earned {n} more",
    "borrowed {n} more from {o}", "got another {n}", "picked out {n} more", "was handed {n} more", "found another {n}",
    "ordered {n} more", "came home with {n} more", "dug up {n} more", "was sent {n} more by {o}",
    "unpacked {n} more", "added another {n}", "took home {n} more", "swapped for {n} more",
]
GAIN_BY_KIND = {
    "food": ["baked {n} more", "picked {n} more", "made {n} more", "cooked {n} more", "packed {n} more"],
    "craft": ["made {n} more", "painted {n} more", "folded {n} more", "glued on {n} more", "strung {n} more"],
    "nature": ["picked {n} more", "dug up {n} more", "spotted {n} more", "gathered {n} more"],
    "school": ["got {n} more from the supply room", "was given {n} more by {o}"],
}
LOSE = [
    "lost {n}", "gave away {n}", "gave {n} to {o}", "sold {n}", "used {n}", "handed out {n}", "traded away {n}",
    "lent {n} to {o}", "donated {n}", "dropped {n}", "misplaced {n}", "threw away {n}", "shared {n} with {o}",
    "put {n} away", "left {n} at school", "gave {n} back to {o}", "returned {n}", "used up {n}", "sent {n} to {o}",
    "packed {n} for {o}", "mailed {n} to {o}", "swapped away {n}", "let {o} keep {n}", "passed out {n}",
    "gave {o} {n} of them", "sold {n} at the yard sale", "lost {n} on the way home", "forgot {n} at the park",
    "set aside {n}", "gave up {n}",
]
LOSE_BY_KIND = {
    "food": ["ate {n}", "served {n}", "shared {n} with {o}", "sold {n} at the bake sale", "dropped {n} on the floor"],
    "toys": ["broke {n}", "lost {n} under the couch", "popped {n}"],
    "kitchen": ["broke {n}", "chipped {n}", "washed and put away {n}"],
    "craft": ["used {n} on a project", "glued {n} onto a card", "used up {n}"],
    "school": ["used up {n}", "broke {n}", "handed in {n}"],
    "nature": ["planted {n}", "threw {n} back", "left {n} on the path"],
}
# someone removes things from a container: {s} the subject
TAKE_OUT = [
    "{s} took {n} out of the {c}", "{s} removed {n} from the {c}", "{s} took out {n}", "{s} emptied {n} out",
    "{s} pulled {n} out", "{s} grabbed {n} from the {c}", "{s} moved {n} to another {c2}",
    "{s} lifted {n} off the {c}", "{s} picked {n} from the {c}", "{s} handed {n} to {o}",
]
TAKE_OUT_PASSIVE = ["{n} fell out", "{n} were taken out", "{n} rolled away", "{n} were sold", "{n} were used",
                    "{n} were given away", "{n} were moved", "{n} went missing", "{n} were handed out"]
PUT_IN = [
    "{s} put {n} more {p} it", "{s} added {n} more", "{s} dropped in {n} more", "{s} placed {n} more {p} it",
    "{s} packed in {n} more", "{s} tossed in {n} more", "{s} filled it with {n} more", "{s} stacked {n} more {p} it",
    "{s} slipped {n} more {p} it", "{s} poured in {n} more",
]
PUT_IN_PASSIVE = ["{n} more were added", "{n} more were put {p} it", "{n} more arrived", "{n} more were dropped in"]

# ---------------------------------------------------------------- leads: the words the answer completes
# {s} subject (name or pronoun), {has}/{had} agree with it, {c} container, {x} things

LEAD_OWN = [
    "Now {s} {has}", "{s} now {has}", "After that, {s} had", "In the end, {s} had", "Then {s} had", "That left {s} with",
    "{s} ended up with", "So {s} {has}", "At the end of the day, {s} had", "Now the number {s} {has} is",
    "By the end, {s} had", "Afterward, {s} had", "Altogether {s} now {has}", "This means {s} {has}",
    "Counting them again, {s} found", "{s} {was} left with", "When {s} counted, there were", "Now {s} owns",
    "That evening, {s} had", "After all that, {s} had", "So now {s} {has}", "Right now {s} {has}",
    "The number of {x} {s} {has} now is", "When {s} checked, {s} had", "In total, {s} now {has}",
]
LEAD_LEFT = [
    "Now {s} {has}", "{s} {has} only", "That left {s} with", "{s} still {has}", "{s} {was} left with", "Now there are",
    "After that, {s} had", "The number left is", "{s} kept", "What remained was", "In the end, {s} had", "So {s} {has}",
    "Left over were", "{s} now {has}", "There were still", "The rest came to", "The amount {s} kept was",
    "Once that was done, {s} had", "At the end, there were", "{s} ended up keeping",
]
LEAD_BOX = [
    "Now the {c} holds", "Now there are", "The {c} now has", "After that, the {c} had", "Now {p} the {c} there are",
    "In the end, the {c} held", "The {c} ended up with", "So the {c} has", "That makes", "The {c} was left with",
    "Counting again, there were", "Now the total {p} the {c} is", "After that there were", "Then there were",
    "This leaves", "The {c} now contains", "So there are now",
]
LEAD_BOTH = [
    "Together they have", "In all, they have", "Altogether there are", "Both of them together have",
    "The two of them have", "Combined, they have", "The total is", "That makes", "Between them they have",
    "Adding them up gives", "Put together, that is", "All together that makes", "In total there are",
    "If they put them together, they would have", "Their total is", "So in all there are", "Altogether they own",
    "Counting both, there are", "The sum is", "Together that makes",
]
LEAD_TOTAL = ["The total is", "That makes", "In all there are", "Altogether there are", "All together that is",
              "The sum comes to", "Adding them gives", "In total there are", "So there are", "That comes to"]
LEAD_DIFF = [
    "The difference is", "{a} has this many more:", "That is a difference of", "The gap is", "The difference comes to",
    "So {a} has", "{b} would need", "The extra amount is", "That means {a} has", "Comparing them, {a} has",
]
LEAD_EACH = ["Each one gets", "Every friend receives", "Each person gets", "Each group has", "That gives each one",
             "So each gets", "Every child gets", "In each group there are", "Each share is", "One share is"]

# document frames
HEADINGS = [
    "Word problems", "Math practice", "Story problems", "Homework", "Practice sheet", "Math journal", "Warm-up",
    "Math class notes", "Problems for today", "Number stories", "Review", "Quiz practice", "Morning work",
    "Math centers", "Exit ticket", "Brain teasers", "Mental math", "Problem of the day", "Weekend practice",
    "Lesson notes", "Practice set", "Challenge problems", "Solve these", "Math corner", "Think and solve",
    "Story time math", "Classroom notes", "Practice page", "Counting stories", "Show your work",
]
QA_LABELS = [("Question:", "Answer:"), ("Q:", "A:"), ("Problem:", "Answer:"), ("Word problem:", "Solution:"),
             ("Try this:", "Answer:"), ("Puzzle:", "Answer:"), ("Ask:", "Reply:"), ("Exercise:", "Answer:"),
             ("Task:", "Result:"), ("Challenge:", "Answer:")]
ANSWER_INTROS = ["The answer is", "Answer:", "Solution:", "It is", "The result is", "Result:", "That is",
                 "The correct answer is", "Final answer:", "So the answer is"]
SPEAKERS = [("Teacher", "Student"), ("Mom", "Kid"), ("Dad", "Child"), ("Coach", "Player"), ("Tutor", "Learner"),
            ("Grandpa", "Grandchild"), ("Ms. Rivera", "Class"), ("Librarian", "Reader")]

# kept in training no matter what the hash says: common words the skill must work with
CORE = {
    "lost {n}", "gave away {n}", "sold {n}", "used {n}", "ate {n}", "gave {n} to {o}", "found {n} more", "got {n} more",
    "bought {n} more", "{s} took {n} out of the {c}", "{s} took out {n}", "{s} put {n} more {p} it", "Now {s} {has}",
    "Now there are", "Now the {c} holds", "Together they have", "In all, they have", "That makes", "The total is",
    "The difference is", "Each one gets", "{s} now {has}", "In the end, {s} had", "That left {s} with",
    "Left over were", "Altogether there are",
}


# v1 problems stay in the v2 data (generator_v2.V1_SHARE), so v1's names and things are never held out
V1_WORDS = set(v1.NAMES) | {s for s, _ in v1.THINGS}


def held_out(entry) -> bool:
    key = entry if isinstance(entry, str) else repr(entry)
    word = entry if isinstance(entry, str) else entry[0]
    return key not in CORE and word not in V1_WORDS and zlib.crc32(key.encode()) % HELD_OUT_EVERY == 0


def part(items, split: str):
    """The train or held-out entries of a list. A held-out part never comes back empty."""
    if split == "all":
        return list(items)
    keep = [x for x in items if held_out(x) == (split == "heldout")]
    return keep or list(items)


@dataclass
class Lexicon:
    split: str  # "train", "heldout" or "all"

    def __post_init__(self):
        s = self.split
        self.female, self.male, self.neutral = part(FEMALE, s), part(MALE, s), part(NEUTRAL, s)
        self.surnames, self.roles = part(SURNAMES, s), part(ROLES, s)
        self.things = {k: part(v, s) for k, v in THINGS.items()}
        self.colors, self.containers, self.places = part(COLORS, s), part(CONTAINERS, s), part(PLACES, s)
        self.crowds, self.times_of_day = part(CROWDS, s), part(TIMES_OF_DAY, s)
        self.gain, self.lose = part(GAIN, s), part(LOSE, s)
        self.gain_by_kind = {k: part(v, s) for k, v in GAIN_BY_KIND.items()}
        self.lose_by_kind = {k: part(v, s) for k, v in LOSE_BY_KIND.items()}
        self.take_out, self.take_out_passive = part(TAKE_OUT, s), part(TAKE_OUT_PASSIVE, s)
        self.put_in, self.put_in_passive = part(PUT_IN, s), part(PUT_IN_PASSIVE, s)
        self.lead_own, self.lead_left, self.lead_box = part(LEAD_OWN, s), part(LEAD_LEFT, s), part(LEAD_BOX, s)
        self.lead_both, self.lead_total = part(LEAD_BOTH, s), part(LEAD_TOTAL, s)
        self.lead_diff, self.lead_each = part(LEAD_DIFF, s), part(LEAD_EACH, s)
        self.headings, self.qa_labels = part(HEADINGS, s), part(QA_LABELS, s)
        self.answer_intros, self.speakers = part(ANSWER_INTROS, s), part(SPEAKERS, s)


def held_out_words() -> set[str]:
    """Held-out names and thing words, for the test that training text never uses them."""
    words = {n for n in FEMALE + MALE + NEUTRAL + SURNAMES if held_out(n)}
    for items in THINGS.values():
        words |= {(p or s + "s") for s, p in items if held_out((s, p))}
    return words
