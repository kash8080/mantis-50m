"""Checkpoints (D-007): atomic writes, keep the last 2 regular ones, never delete pre-decay or final.

run_dir/ckpt/step_0001234.pt      regular, every ~20 min (pruned to the last 2)
run_dir/ckpt/predecay_0030400.pt  at the start of the decay, kept forever (D-036 extends from it)
run_dir/ckpt/final_0034000.pt     end of the decay: the release (D-009)
run_dir/milestones/milestone_0003815.pt   weights only, for the benchmark evals
"""

import dataclasses
import os
import re
from pathlib import Path

import torch

_NAME = re.compile(r"^(step|predecay|final)_(\d+)\.pt$")


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)  # a crash mid-save leaves only the .tmp file, never a broken .pt


def save_checkpoint(path: Path, model, optimizers, run_config, state: dict) -> None:
    """Everything resume needs. The data order is a function of the step, so no loader state."""
    _atomic_save({
        "model": model.state_dict(),
        "optimizers": [opt.state_dict() for opt in optimizers],
        "model_config": dataclasses.asdict(model.cfg),
        "run_config": dataclasses.asdict(run_config),
        **state,  # step, tokens, raw_tokens, seconds, usd, git_commit
    }, path)


def save_weights(path: Path, model, state: dict) -> None:
    _atomic_save({"model": model.state_dict(), "model_config": dataclasses.asdict(model.cfg), **state}, path)


def checkpoint_name(kind: str, step: int) -> str:
    return f"{kind}_{step:07d}.pt"


def latest_checkpoint(ckpt_dir: Path) -> Path | None:
    """The full checkpoint with the highest step, of any kind."""
    found = [(int(m.group(2)), p) for p in ckpt_dir.glob("*.pt") if (m := _NAME.match(p.name))]
    return max(found)[1] if found else None


def prune(ckpt_dir: Path, keep_last: int = 2) -> None:
    """Delete regular checkpoints beyond the newest keep_last. Pre-decay and final are never touched."""
    regular = sorted(p for p in ckpt_dir.glob("step_*.pt"))
    for p in regular[:-keep_last]:
        p.unlink()
