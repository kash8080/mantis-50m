"""Export our model to a standard Qwen3ForCausalLM folder (D-012, D-017).

The folder loads with plain `AutoModelForCausalLM.from_pretrained`, no custom code, so lm-eval,
the ArithMark script and llama.cpp's converter all read it.
"""

from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from slm50m.config import EOS_ID, ModelConfig
from slm50m.model.qwen3 import Qwen3LM


def hf_config(cfg: ModelConfig) -> Qwen3Config:
    """The config.json fields listed in docs/03-architecture.md."""
    return Qwen3Config(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.d_model,
        intermediate_size=cfg.d_ff,
        num_hidden_layers=cfg.n_layers,
        num_attention_heads=cfg.n_heads,
        num_key_value_heads=cfg.n_kv_heads,
        head_dim=cfg.head_dim,
        max_position_embeddings=cfg.max_seq_len,
        rms_norm_eps=cfg.rms_eps,
        rope_parameters={"rope_type": "default", "rope_theta": cfg.rope_theta},
        tie_word_embeddings=True,
        attention_bias=False,
        use_sliding_window=False,
        bos_token_id=None,  # no BOS (D-018)
        eos_token_id=EOS_ID,
        pad_token_id=EOS_ID,
        torch_dtype="float32",
    )


def to_hf(model: Qwen3LM) -> Qwen3ForCausalLM:
    """Build a Qwen3ForCausalLM holding our weights (fp32, on CPU)."""
    hf = Qwen3ForCausalLM(hf_config(model.cfg))
    # our names are HF's names without the "model." prefix. lm_head is the same tensor as the embedding
    state = {f"model.{name}": t.detach().float().cpu() for name, t in model.state_dict().items()}
    state["lm_head.weight"] = state["model.embed_tokens.weight"]
    missing, unexpected = hf.load_state_dict(state, strict=False)
    # rotary inv_freq is a non-persistent buffer in HF, so nothing should be missing or extra
    assert not unexpected, f"unexpected keys: {unexpected}"
    assert not [k for k in missing if "rotary" not in k], f"missing keys: {missing}"
    hf.tie_weights()
    assert hf.lm_head.weight.data_ptr() == hf.model.embed_tokens.weight.data_ptr(), "head is not tied"
    return hf.eval()


def export(model: Qwen3LM, out_dir: str | Path, tokenizer_dir: str | Path | None = None) -> Path:
    """Write config.json + model.safetensors (+ tokenizer files) to out_dir."""
    out_dir = Path(out_dir)
    hf = to_hf(model)
    hf.save_pretrained(out_dir)
    if tokenizer_dir is not None:
        from slm50m.tokenizer.train_tokenizer import load_hf_tokenizer

        load_hf_tokenizer(tokenizer_dir).save_pretrained(out_dir)
    return out_dir


def load_checkpoint_model(ckpt_path: str | Path) -> Qwen3LM:
    """Rebuild our model from a training checkpoint (full or milestone)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = Qwen3LM(ModelConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    return model.eval()
