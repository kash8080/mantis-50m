"""Math evals outside ArithMark, to show the arithmetic skill is real and not tuned to one benchmark (Q-27).

  - GPT-3 arithmetic (Brown et al. 2020), 10 tasks x 2,000: exact match of the greedy continuation
  - ASDiv/SVAMP held-out, 232 problems never paraphrased by Orca-Math (heldout_arith.py)
  - MathQA test, 2,985 problems, 5 choices
  - MMLU elementary mathematics, 378 questions, 4 choices: letter format and continuation (answer text) format
Zero-shot lm-eval 0.4.12 with the stock prompts and metrics. lm-eval's own `arithmetic` and `mathqa` tasks need
dataset loading scripts, which datasets 5 dropped, so the same configs run here on the raw files at pinned commits.

    uv run modal run modal_app.py::run_math_evals          # pre-decay and final, on the runs Volume
"""

import hashlib
import json
import urllib.request
from pathlib import Path

from slm50m.evaluation.evaluate import versions
from slm50m.evaluation.heldout_arith import heldout_items, score

H = "https://huggingface.co/datasets"
ARITH = f"{H}/EleutherAI/arithmetic/resolve/cf5ec4512aaa47cdebf02ca032b7af870528c272/data"
ARITH_TASKS = {"2da": "two_digit_addition", "2ds": "two_digit_subtraction", "3da": "three_digit_addition",
               "3ds": "three_digit_subtraction", "4da": "four_digit_addition", "4ds": "four_digit_subtraction",
               "5da": "five_digit_addition", "5ds": "five_digit_subtraction", "2dm": "two_digit_multiplication",
               "1dc": "single_digit_three_ops"}
MMLU = f"{H}/cais/mmlu/resolve/c30699e8356da336a370243923dbaf21066bb9fe/elementary_mathematics"
URLS = {
    **{f"arithmetic_{k}.jsonl": f"{ARITH}/{v}.jsonl" for k, v in ARITH_TASKS.items()},
    "mathqa_test.parquet": f"{H}/allenai/math_qa/resolve/fafb9f7ee5b9ec4da9499f9c4177a4c91389f2d6/default/test/0000.parquet",
    "mmlu_elementary_mathematics_test.parquet": f"{MMLU}/test-00000-of-00001.parquet",
    # heldout_arith.py's four pinned files
    "asdiv.parquet": f"{H}/EleutherAI/asdiv/resolve/8f95807222d87b4c688c3c22a6ba2801e1fa03e2/asdiv/validation-00000-of-00001.parquet",
    "svamp_train.json": f"{H}/ChilleD/SVAMP/resolve/5e0bf1e5e7c0e9c4bc39180d224f41f3f801b7ef/train.json",
    "svamp_test.json": f"{H}/ChilleD/SVAMP/resolve/5e0bf1e5e7c0e9c4bc39180d224f41f3f801b7ef/test.json",
    "orca.parquet": f"{H}/microsoft/orca-math-word-problems-200k/resolve/29255d1770cc4eac66e5e7fa378cba542c026350/data/train-00000-of-00001.parquet",
}


def fetch(raw: Path) -> dict[str, str]:
    """Download missing files into raw. Returns sha256 per file."""
    raw.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        if not (raw / name).exists():
            urllib.request.urlretrieve(url, raw / name)
    return {name: hashlib.sha256((raw / name).read_bytes()).hexdigest() for name in URLS}


def arithmetic_docs(dataset):
    """What EleutherAI/arithmetic's loading script did: "Question: What is 98 plus 45?\nAnswer:" + " 143"."""
    return dataset.map(lambda d: {"context": d["context"].strip().replace("\n\n", "\n").replace("Q:", "Question:")
                                  .replace("A:", "Answer:")})


def mathqa_choices(doc):
    """lm-eval's tasks/mathqa/utils.py."""
    import re

    return [c[4:].rstrip(" ,") for c in re.findall(r"[abcd] \) .*?, |e \) .*?$", doc["options"])]


def task_configs(raw: Path) -> list[dict]:
    """The stock lm-eval 0.4.12 configs (tasks/arithmetic, mathqa, mmlu/default, mmlu/continuation) on local files."""
    acc = [{"metric": "acc", "aggregation": "mean", "higher_is_better": True}]
    norm = acc + [{"metric": "acc_norm", "aggregation": "mean", "higher_is_better": True}]
    mmlu = {"dataset_path": "parquet", "output_type": "multiple_choice", "test_split": "test", "doc_to_target": "answer",
            "dataset_kwargs": {"data_files": {"test": str(raw / "mmlu_elementary_mathematics_test.parquet")}}}
    configs = [{"task": f"arithmetic_{k}", "dataset_path": "json", "output_type": "loglikelihood",
                "dataset_kwargs": {"data_files": {"validation": str(raw / f"arithmetic_{k}.jsonl")}},
                "validation_split": "validation", "process_docs": arithmetic_docs, "doc_to_text": "{{context}}",
                "doc_to_target": "{{completion}}", "target_delimiter": "", "metric_list": acc} for k in ARITH_TASKS]
    configs.append({"task": "mathqa", "dataset_path": "parquet", "output_type": "multiple_choice",
                    "dataset_kwargs": {"data_files": {"test": str(raw / "mathqa_test.parquet")}}, "test_split": "test",
                    "doc_to_text": "Question: {{Problem}}\nAnswer:",
                    "doc_to_target": "{{['a', 'b', 'c', 'd', 'e'].index(correct)}}", "doc_to_choice": mathqa_choices,
                    "metric_list": norm})
    configs.append({**mmlu, "task": "mmlu_elementary_mathematics", "metric_list": acc,
                    "description": "The following are multiple choice questions (with answers) about elementary"
                                   " mathematics.\n\n",
                    "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\n"
                                   "D. {{choices[3]}}\nAnswer:", "doc_to_choice": ["A", "B", "C", "D"]})
    configs.append({**mmlu, "task": "mmlu_elementary_mathematics_continuation", "metric_list": norm,
                    "description": "The following are questions (with answers) about elementary mathematics.\n\n",
                    "doc_to_text": "Question: {{question.strip()}}\nAnswer:", "doc_to_choice": "{{choices}}"})
    return configs


def run_math_evals(hf_dir: Path, raw: Path, out_dir: Path, device: str = "cpu", limit: int | None = None) -> dict:
    """Every math eval on one HF export. Writes out_dir/lm_eval_math.json, heldout_arith.json and math_summary.json."""
    import lm_eval

    file_sha256 = fetch(raw)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = lm_eval.simple_evaluate(model="hf", model_args=f"pretrained={hf_dir},dtype=float32",
                                      tasks=task_configs(raw), num_fewshot=0, batch_size=32, device=device,
                                      limit=limit)
    (out_dir / "lm_eval_math.json").write_text(json.dumps(results, indent=2, default=str))
    scores = {task: {m.split(",")[0]: 100 * v for m, v in r.items() if m in ("acc,none", "acc_norm,none")}
              for task, r in results["results"].items()}
    arith = [scores[f"arithmetic_{k}"]["acc"] for k in ARITH_TASKS]
    scores["arithmetic_mean"] = {"acc": sum(arith) / len(arith)}

    rows = heldout_items(raw)[:limit] if limit else heldout_items(raw)
    heldout = score(str(hf_dir), rows)
    (out_dir / "heldout_arith.json").write_text(json.dumps(heldout, indent=1))
    for fmt in (k for k in heldout if "|" not in k):  # the six format/distractor rows of the model card table
        scores[f"heldout_arith/{fmt}"] = {"acc_norm": heldout[fmt]["acc_norm"]}

    summary = {"model": str(hf_dir), "scores": scores, "limit": limit, "versions": versions(),
               "file_sha256": file_sha256}
    (out_dir / "math_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
