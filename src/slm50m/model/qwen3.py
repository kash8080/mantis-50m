"""The decoder (D-016, D-019 to D-021).

Same maths as transformers' Qwen3ForCausalLM, so the export is only a rename of the weights.
Parameter names match HF too (minus the "model." prefix), which keeps export.py trivial.

Shapes in comments: B batch, S sequence, D d_model, H query heads, K kv heads, hd head_dim, V vocab.
"""

import torch
import torch.nn.functional as F
from torch import nn

from slm50m.config import ModelConfig


class RMSNorm(nn.Module):
    """RMSNorm in fp32, written exactly like Qwen3RMSNorm so bf16 rounding matches too."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # HF casts back first, then scales. Same order here
        return self.weight * x.to(input_dtype)


def rope_tables(cfg: ModelConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin for positions 0..max_seq_len-1, each (S_max, hd). Computed in fp32 like HF."""
    hd = cfg.head_dim
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, hd, 2, dtype=torch.float) / hd))  # (hd/2,)
    positions = torch.arange(cfg.max_seq_len, dtype=torch.float)
    freqs = positions[:, None] * inv_freq[None, :]  # (S_max, hd/2)
    # HF's layout: the two halves of the head dim share frequencies (not interleaved pairs)
    emb = torch.cat((freqs, freqs), dim=-1)  # (S_max, hd)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """(x1, x2) -> (-x2, x1) over the two halves of the last dim."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Attention(nn.Module):
    """GQA with per-head QK-norm before RoPE (Qwen3's form, D-020)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.head_dim = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        # project, split into heads, norm each head, then put heads before positions
        q = self.q_norm(self.q_proj(x).view(B, S, self.n_heads, self.head_dim)).transpose(1, 2)  # (B, H, S, hd)
        k = self.k_norm(self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim)).transpose(1, 2)  # (B, K, S, hd)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B, K, S, hd)

        # RoPE. cos and sin are (S, hd) and broadcast over batch and heads
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin

        # causal attention. enable_gqa lets kv head j serve query heads j*H/K .. (j+1)*H/K - 1, like HF's repeat_kv.
        # Default scale is 1/sqrt(hd), same as HF. No document masking (D-030)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)  # (B, H, S, hd)
        out = out.transpose(1, 2).reshape(B, S, self.n_heads * self.head_dim)
        return self.o_proj(out)


class MLP(nn.Module):
    """SwiGLU: down(silu(gate(x)) * up(x))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    """Pre-norm block: x + attn(norm(x)), then x + mlp(norm(x))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3LM(nn.Module):
    """The language model. Tied embedding: the LM head is the embedding matrix."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        cos, sin = rope_tables(cfg)
        # not saved in checkpoints, rebuilt from the config
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """D-021: normal(0, 0.02) everywhere, zero o_proj and down_proj, norm scales 1.

        With zero output projections every block starts as the identity.
        """
        std = self.cfg.init_std
        nn.init.normal_(self.embed_tokens.weight, std=std)
        for block in self.layers:
            for linear in (block.self_attn.q_proj, block.self_attn.k_proj, block.self_attn.v_proj,
                           block.mlp.gate_proj, block.mlp.up_proj):
                nn.init.normal_(linear.weight, std=std)
            nn.init.zeros_(block.self_attn.o_proj.weight)
            nn.init.zeros_(block.mlp.down_proj.weight)
        for module in self.modules():
            if isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

    def forward(self, input_ids: torch.Tensor | None = None, inputs_embeds: torch.Tensor | None = None) -> torch.Tensor:
        """Returns logits (B, S, V). Pass token ids (B, S), or ready embeddings (B, S, D) for TST bags."""
        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds  # (B, S, D)
        S = x.shape[1]
        # HF casts cos/sin to the dtype of the embeddings (fp32 here, also under autocast)
        cos, sin = self.rope_cos[:S].to(x.dtype), self.rope_sin[:S].to(x.dtype)
        for block in self.layers:
            x = block(x, cos, sin)
        x = self.norm(x)
        return F.linear(x, self.embed_tokens.weight)  # tied head: (B, S, D) @ (D, V)

    def n_params(self) -> int:
        """Counts the tied matrix once, like sum(p.numel() for p in model.parameters()) on the HF model."""
        return sum(p.numel() for p in self.parameters())
