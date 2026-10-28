"""Held-out arithmetic check: ASDiv and SVAMP word problems as 4-choice continuations, scored like ArithMark-3
(context and ending tokenized separately, mean log-prob per ending token = acc_norm). Model card, "Held-out arithmetic".

Orca-Math, in our `word_problems` source, paraphrases most of both sets. Only problems that share no number-masked
8-gram with any Orca-Math question are kept: 232 problems (ASDiv 224, SVAMP 8), listed in heldout_arith_items.json.

Fetch the four pinned files into one folder (curl, since Python HF calls can hang on IPv6), then run on HF exports:

    H=https://huggingface.co/datasets
    curl -L -o raw/asdiv.parquet $H/EleutherAI/asdiv/resolve/8f95807222d87b4c688c3c22a6ba2801e1fa03e2/asdiv/validation-00000-of-00001.parquet
    curl -L -o raw/svamp_train.json $H/ChilleD/SVAMP/resolve/5e0bf1e5e7c0e9c4bc39180d224f41f3f801b7ef/train.json
    curl -L -o raw/svamp_test.json $H/ChilleD/SVAMP/resolve/5e0bf1e5e7c0e9c4bc39180d224f41f3f801b7ef/test.json
    curl -L -o raw/orca.parquet $H/microsoft/orca-math-word-problems-200k/resolve/29255d1770cc4eac66e5e7fa378cba542c026350/data/train-00000-of-00001.parquet
    uv run slm-heldout-arith raw kash8080/Mantis-50M --out heldout_results.json
"""

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ITEMS_FILE = Path(__file__).with_name("heldout_arith_items.json")
RAW_SHA256 = {
    "asdiv.parquet": "79dbad6536fe3e8cd449e67da859d7a6f5ac0d8f852efe563742ca5faee7fb75",
    "svamp_train.json": "bb3b4cd2957f07643bbf9f8f8f5faba3bfb47fcb1d9034655b03060a76675e2b",
    "svamp_test.json": "9b5590fce66ea78ecfc7d0fc4a91fb801ffe289340269f8d9d28b21acad04ae6",
    "orca.parquet": "38513af71b04e2f710c6cd98f750bf72a87b7a4da1804caaa76f898585c1cfaf",  # ORCA_REV in dataprep
}
NGRAM = 8
KEEP_ASDIV = {"Addition", "Subtraction", "Multiplication", "Common-Division", "Sum", "Difference", "Surplus",
              "TVQ-Initial", "TVQ-Change", "TVQ-Final"}
OP_NAME = {"+": "add", "-": "sub", "*": "mul", "/": "div"}


def check_raw(raw: Path):
    for name, want in RAW_SHA256.items():
        got = hashlib.sha256((raw / name).read_bytes()).hexdigest()
        if got != want:
            raise ValueError(f"{raw / name}: sha256 {got}, expected {want}")


def masked_ngrams(text: str, n: int = NGRAM) -> set[tuple[str, ...]]:
    """Lowercase word n-grams with every number replaced by #, so paraphrases with new numbers still match."""
    words = ["#" if w[0].isdigit() else w for w in re.findall(r"[a-z]+|\d+(?:\.\d+)?", text.lower())]
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def clean_texts(texts: list[str], orca_questions: list[str]) -> set[str]:
    """Problem texts with no exact match and no masked 8-gram shared with Orca-Math."""
    orca_grams = set().union(*(masked_ngrams(q) for q in orca_questions))
    exact = set(orca_questions)
    return {t for t in texts if t not in exact and not masked_ngrams(t) & orca_grams}


def as_int(x: float) -> int | None:
    return round(x) if abs(x - round(x)) < 1e-9 else None


def op_distractors(nums: list[float], n: int) -> list[int]:
    """Results of the wrong operation on the problem's own numbers: the classic word-problem mistake."""
    out = set()
    for i, a in enumerate(nums):
        for b in nums[i + 1:]:
            for v in (a + b, abs(a - b), a * b, a / b if b else 0, b / a if a else 0):
                v = as_int(v)
                if v is not None and v > 0 and v != n:
                    out.add(v)
        v = as_int(a)
        if v is not None and v > 0 and v != n:
            out.add(v)
    return sorted(out)


def near_distractors(n: int, rng: random.Random) -> list[int]:
    pool = [v for v in (n - 2, n - 1, n + 1, n + 2, n + 3) if v >= 0 and v != n][:4]
    return rng.sample(pool, 3)


def items(raw: Path) -> list[dict]:
    """Every integer-answer +, -, x, / problem of both sets, with distractors. Order and seed fix the distractors."""
    rng = random.Random(0)
    rows = []
    for f in ("svamp_train.json", "svamp_test.json"):
        for r in json.loads((raw / f).read_text()):
            n = as_int(float(r["Answer"]))
            ops = re.findall(r"[-+*/]", r["Equation"].replace("( -", "("))
            if n is None or n < 0 or n > 100000:
                continue
            nums = [float(x) for x in re.findall(r"\d+\.?\d*", r["Equation"])]
            rows.append({"src": "svamp", "id": r["ID"], "body": r["Body"].strip(), "q": r["Question"].strip(),
                         "n": n, "unit": None, "nums": nums,
                         "kind": "+".join(sorted({OP_NAME[o] for o in ops})) or "none", "steps": len(ops)})
    for i, r in enumerate(pq.read_table(raw / "asdiv.parquet").to_pylist()):
        m = re.fullmatch(r"\s*([\d.]+)\s*\(([^)]*)\)\s*", r["answer"])
        if r["solution_type"] not in KEEP_ASDIV or not m:
            continue
        n = as_int(float(m.group(1)))
        lhs = r["formula"].split("=")[0]
        if n is None or n > 100000 or not re.fullmatch(r"[\d.+\-*/() ]+", lhs):
            continue
        ops = re.findall(r"[-+*/]", lhs)
        rows.append({"src": "asdiv", "id": i, "body": r["body"].strip(), "q": r["question"].strip(), "n": n,
                     "unit": m.group(2).strip() or None, "nums": [float(x) for x in re.findall(r"\d+\.?\d*", lhs)],
                     "kind": "+".join(sorted({OP_NAME[o] for o in ops})) or "none", "steps": len(ops)})
    for r in rows:
        r["near"] = near_distractors(r["n"], rng)
        ops = op_distractors(r["nums"], r["n"])
        rng.shuffle(ops)
        r["op"] = (ops + [v for v in r["near"] if v not in ops])[:3]
    return rows


def heldout_items(raw: Path) -> list[dict]:
    """The 232 problems, checked against the committed list."""
    check_raw(raw)
    rows = items(raw)
    orca = pq.read_table(raw / "orca.parquet", columns=["question"]).column(0).to_pylist()
    clean = clean_texts([f"{r['body']} {r['q']}" for r in rows], orca)
    rows = [r for r in rows if f"{r['body']} {r['q']}" in clean]
    ids = {s: [r["id"] for r in rows if r["src"] == s] for s in ("asdiv", "svamp")}
    if ids != json.loads(ITEMS_FILE.read_text()):
        raise ValueError(f"held-out problems differ from {ITEMS_FILE.name}")
    return rows


def formats(r: dict) -> dict[str, tuple[str, list[str]]]:
    """Each format: (context, endings with the answer first)."""
    story = f"{r['body']} {r['q']}"
    out = {}
    for dname in ("near", "op"):
        vals = [r["n"]] + r[dname]
        out[f"qa/{dname}"] = (f"Question: {story}\nAnswer:", [f" {v}" for v in vals])
        out[f"answer-is/{dname}"] = (f"{story} The answer is", [f" {v}." for v in vals])
        if r["unit"]:
            out[f"unit/{dname}"] = (f"{story} The answer is", [f" {v} {r['unit']}." for v in vals])
    return out


@torch.no_grad()
def hit(model, tok, ctx: str, endings: list[str]) -> tuple[int, int]:
    """(acc_norm hit, acc hit) for one item, answer at index 0."""
    c = tok(ctx, add_special_tokens=False)["input_ids"]
    seqs = [(c, tok(e, add_special_tokens=False)["input_ids"]) for e in endings]
    width = max(len(a) + len(b) for a, b in seqs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.tensor([a + b + [pad] * (width - len(a) - len(b)) for a, b in seqs])
    logp = F.log_softmax(model(ids).logits.float(), -1)
    mean, total = [], []
    for i, (a, b) in enumerate(seqs):
        pos = torch.arange(len(a) - 1, len(a) - 1 + len(b))
        lp = logp[i, pos, torch.tensor(b)]
        mean.append(lp.mean().item())
        total.append(lp.sum().item())
    return int(max(range(4), key=mean.__getitem__) == 0), int(max(range(4), key=total.__getitem__) == 0)


def score(model_dir: str, rows: list[dict]) -> dict:
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
    agg = defaultdict(lambda: [0, 0, 0])  # norm hits, raw hits, total
    for r in rows:
        for fmt, (ctx, endings) in formats(r).items():
            hn, hr = hit(model, tok, ctx, endings)
            small = max(r["nums"] + [r["n"]]) <= 100 and r["steps"] <= 2
            for key in (fmt, f"{fmt}|{r['src']}", f"{fmt}|{r['kind']}") + ((f"{fmt}|small<=100,<=2 steps",) if small else ()):
                agg[key][0] += hn
                agg[key][1] += hr
                agg[key][2] += 1
    return {k: {"acc_norm": 100 * v[0] / v[2], "acc": 100 * v[1] / v[2], "n": v[2]} for k, v in agg.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_dir", help="folder with the four pinned files (see the module docstring)")
    parser.add_argument("model_dirs", nargs="+", help="HF model folders or Hub ids")
    parser.add_argument("--out", default="heldout_results.json")
    args = parser.parse_args()
    rows = heldout_items(Path(args.raw_dir))
    print(f"{len(rows)} problems: " + ", ".join(f"{s} {sum(r['src'] == s for r in rows)}" for s in ("asdiv", "svamp")))
    results = {}
    for d in args.model_dirs:
        results[d] = score(d, rows)
        print(f"\n== {d}")
        for k in sorted(results[d]):
            v = results[d][k]
            if "|" not in k or v["n"] >= 40:
                print(f"  {k:<32} acc_norm {v['acc_norm']:5.1f}  acc {v['acc']:5.1f}  n={v['n']}")
    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"\nwrote {args.out}")
