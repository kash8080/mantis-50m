"""Gate 0: the II formula reproduces the leaderboard, and lm_eval 0.4.12 runs on an exported model (D-035)."""

import dataclasses

import pytest
import torch

from slm50m.config import PRESETS
from slm50m.evaluation.evaluate import intelligence_index, run_lm_eval
from slm50m.model.export import export
from slm50m.model.qwen3 import Qwen3LM


def test_ii_matches_leaderboard():
    # fromziro/50M-pro on 24 Oct 2026 (docs/01-goal-and-leaderboard.md): II 17.93
    assert round(intelligence_index(33.25, 45.62, 25.51, 64.80, 37.4), 2) == 17.93


@pytest.mark.network
def test_lm_eval_runs_on_export(tmp_path, tokenizer_dir):
    """hellaswag --limit 20 on a tiny random export. Downloads the HellaSwag dataset the first time."""
    cfg = dataclasses.replace(PRESETS["test"], vocab_size=1_024, max_seq_len=1_024)
    torch.manual_seed(0)
    hf_dir = export(Qwen3LM(cfg), tmp_path / "hf", tokenizer_dir)
    scores = run_lm_eval(hf_dir, tmp_path / "lm_eval.json", tasks=("hellaswag",), limit=20, batch_size=4)
    assert 0 <= scores["hellaswag"] <= 100
    assert (tmp_path / "lm_eval.json").exists()
