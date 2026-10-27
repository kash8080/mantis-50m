"""Gate 0: exact param count (D-019) and fp32 logits equal to Qwen3ForCausalLM (D-012, D-017)."""

import torch
from transformers import AutoModelForCausalLM

from slm50m.config import PRESETS
from slm50m.model.export import export
from slm50m.model.qwen3 import Qwen3LM

MAIN_PARAMS = 49_322_240


def randomized(cfg) -> Qwen3LM:
    """Random weights everywhere. The real init zeroes o_proj and down_proj, which would make blocks the identity
    and hide mistakes in attention or the MLP."""
    torch.manual_seed(0)
    model = Qwen3LM(cfg)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() > 1:
                p.normal_(0, 0.05)
            else:
                p.uniform_(0.5, 1.5)  # norm scales away from 1
    return model.eval()


def test_main_preset_param_count():
    model = Qwen3LM(PRESETS["main"])
    assert PRESETS["main"].n_params() == MAIN_PARAMS
    assert model.n_params() == MAIN_PARAMS


def test_export_matches_hf(tmp_path, tokenizer_dir):
    """Export the real shape, load it back with plain transformers, compare count and fp32 logits."""
    model = randomized(PRESETS["main"])
    hf = AutoModelForCausalLM.from_pretrained(export(model, tmp_path / "hf", tokenizer_dir), dtype=torch.float32).eval()

    assert type(hf).__name__ == "Qwen3ForCausalLM"
    assert sum(p.numel() for p in hf.parameters()) == MAIN_PARAMS  # how the leaderboard counts
    assert hf.lm_head.weight.data_ptr() == hf.model.embed_tokens.weight.data_ptr()

    ids = torch.randint(0, PRESETS["main"].vocab_size, (2, 256), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ours, theirs = model(ids), hf(ids).logits
    # 25 layers of fp32 with a different SDPA kernel path leave ~4e-6 relative rounding noise. A real
    # mismatch (RoPE layout, norm order, GQA grouping) shows up at order 1
    assert (ours - theirs).abs().max() / ours.abs().max() < 1e-5
