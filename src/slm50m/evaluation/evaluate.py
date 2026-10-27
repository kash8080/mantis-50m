"""Benchmarks and the Intelligence Index, with every artifact kept (D-009, D-010).

  - lm-eval 0.4.12, zero-shot, acc_norm: hellaswag, arc_easy, arc_challenge, piqa
  - ArithMark-3 with the official script, unmodified, default settings, on the HF export
  - II with the leaderboard formula (docs/01-goal-and-leaderboard.md)
Every eval writes its raw JSON plus a summary.json with versions, step and git commit.
"""

import hashlib
import importlib.metadata
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

from slm50m.config import BUILD_DIR
from slm50m.dataprep.benchmarks import arithmark_path

LM_EVAL_TASKS = ("hellaswag", "arc_easy", "arc_challenge", "piqa")
ARITHMARK_SCRIPT_URL = "https://huggingface.co/datasets/AxiomicLabs/ArithMark-3.0/resolve/main/bencharithmark-3.py"
ARITHMARK_SCRIPT = BUILD_DIR / "cache" / "arithmark_script" / "bencharithmark-3.py"


def normalize(score: float, chance: float) -> float:
    """Chance-normalized score, all in percent."""
    return 100 * (score - chance) / (100 - chance)


def intelligence_index(hellaswag: float, arc_easy: float, arc_challenge: float, piqa: float, arithmark: float) -> float:
    """Scores in percent (acc_norm)."""
    return (normalize(hellaswag, 25) + normalize((arc_easy + arc_challenge) / 2, 25) + normalize(piqa, 50)
            + 0.65 * normalize(arithmark, 25)) / 3.65


def versions() -> dict:
    out = {p: importlib.metadata.version(p) for p in ("torch", "transformers", "tokenizers", "lm_eval")}
    from slm50m.training.trainer import git_commit

    out["python"] = sys.version.split()[0]
    out["git_commit"] = git_commit()
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_lm_eval(hf_dir: Path, out_path: Path, tasks=LM_EVAL_TASKS, limit: int | None = None,
                device: str = "cpu", batch_size: int = 32) -> dict[str, float]:
    """Zero-shot lm-eval on the exported folder in fp32. Returns acc_norm per task, in percent."""
    import lm_eval

    results = lm_eval.simple_evaluate(model="hf", model_args=f"pretrained={hf_dir},dtype=float32",
                                      tasks=list(tasks), num_fewshot=0, batch_size=batch_size,
                                      device=device, limit=limit)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str))
    return {task: 100 * results["results"][task]["acc_norm,none"] for task in tasks}


def arithmark_script() -> Path:
    if not ARITHMARK_SCRIPT.exists():
        ARITHMARK_SCRIPT.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(ARITHMARK_SCRIPT_URL, ARITHMARK_SCRIPT)
    return ARITHMARK_SCRIPT


def run_arithmark(hf_dir: Path, out_dir: Path, device: str = "auto") -> float:
    """The official script on the export, default settings (bf16, batch 32, context 1,024). Returns acc_norm %."""
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(arithmark_script()), "--model", str(hf_dir), "--data-path", str(arithmark_path()),
                    "--results-dir", str(out_dir), "--device", device], check=True)
    result_file = next(out_dir.glob("*_arithmark-3_results.json"))
    return json.loads(result_file.read_text())["results"]["arithmark-3"]["acc_norm"]


def evaluate_checkpoint(ckpt_path: Path, tokenizer_dir: Path, out_dir: Path, device: str = "cpu",
                        limit: int | None = None) -> dict:
    """Export a checkpoint, run every benchmark, write out_dir/summary.json. Returns the summary."""
    import torch

    from slm50m.model.export import export, load_checkpoint_model

    hf_dir = export(load_checkpoint_model(ckpt_path), out_dir / "hf", tokenizer_dir)
    scores = run_lm_eval(hf_dir, out_dir / "lm_eval.json", limit=limit, device=device)
    scores["arithmark"] = run_arithmark(hf_dir, out_dir / "arithmark", "cuda" if device.startswith("cuda") else "cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    summary = {
        "checkpoint": str(ckpt_path), "step": ckpt.get("step"), "tokens": ckpt.get("tokens"),
        "scores": scores, "ii": intelligence_index(scores["hellaswag"], scores["arc_easy"], scores["arc_challenge"],
                                                   scores["piqa"], scores["arithmark"]),
        "limit": limit, "versions": versions(),
        "arithmark_data_sha256": sha256(arithmark_path()), "arithmark_script_sha256": sha256(arithmark_script()),
        "command": " ".join(sys.argv),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
