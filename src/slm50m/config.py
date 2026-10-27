"""Model presets and the run config.

Every number here comes from docs/decisions.md. The D-xxx next to a field says which decision.
Values that wait on gate 3 (total steps, micro-batch) are marked Q-16 / Q-17.
"""

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# The frozen trained tokenizer, an output. In git after gate 1 (D-018)
TOKENIZER_DIR = REPO_ROOT / "outputs" / "tokenizer"
# Its training corpus, an input: data/tokenizer/corpus/ is gitignored, README.md and manifest.json are not
TOKENIZER_CORPUS_DIR = REPO_ROOT / "data" / "tokenizer"
# One JSON per source, recording exactly which parquet files we used (D-036)
MANIFEST_DIR = REPO_ROOT / "manifests"
# Local scratch: downloads, test outputs. Gitignored
BUILD_DIR = REPO_ROOT / "_build"

EOS_ID = 0  # <|endoftext|> is id 0: separator, eos and pad (D-018)


@dataclass(frozen=True)
class ModelConfig:
    """Shape of the decoder. Field names follow Qwen3Config where they map one to one."""

    vocab_size: int = 16_384  # D-018
    d_model: int = 384  # D-019
    n_layers: int = 25
    n_heads: int = 6
    n_kv_heads: int = 2
    head_dim: int = 64
    d_ff: int = 1_152
    rope_theta: float = 100_000.0  # D-020
    rms_eps: float = 1e-6
    max_seq_len: int = 1_024  # D-029
    init_std: float = 0.02  # D-021

    def n_params(self) -> int:
        """Total params with the tied embedding counted once (how the leaderboard counts)."""
        d, hd = self.d_model, self.head_dim
        attn = d * self.n_heads * hd * 2 + d * self.n_kv_heads * hd * 2  # q, o + k, v
        mlp = 3 * d * self.d_ff  # gate, up, down
        norms = 2 * d + 2 * hd  # input + post-attention norm, q_norm + k_norm
        return self.vocab_size * d + self.n_layers * (attn + mlp + norms) + d  # + final norm

    def flops_per_token(self, seq_len: int) -> float:
        """Training FLOPs per token: 6N for the matmuls plus causal attention (fwd + bwd)."""
        attn = 6 * self.n_layers * seq_len * self.n_heads * self.head_dim
        return 6 * self.n_params() + attn


PRESETS: dict[str, ModelConfig] = {
    # the released model: exactly 49,322,240 params (D-019)
    "main": ModelConfig(),
    # gate 2 smoke test and gate 4 ablations: ~13.6M params, same tokenizer
    "proxy": ModelConfig(d_model=256, n_layers=12, n_heads=4, n_kv_heads=2, d_ff=768),
    # unit tests on the Mac: a few hundred thousand params with a small test vocab
    "test": ModelConfig(vocab_size=512, d_model=64, n_layers=2, n_heads=2, n_kv_heads=1, head_dim=32,
                        d_ff=192, max_seq_len=64),
}

# Mix weights (D-023). TxT360 QA is two subsets at half each, so it is two sources here
STABLE_MIX = {
    "climbmix": 0.45,
    "txt360_web": 0.12,
    "fineweb_edu": 0.10,
    "cosmopedia_v2": 0.05,
    "wikihow": 0.02,
    "finemath": 0.07,
    "txt360_qa": 0.04,
    "txt360_nltk_qa": 0.04,
    "word_problems": 0.04,
    "arith": 0.07,  # D-024
}
DECAY_MIX = {
    "climbmix": 0.24,
    "txt360_web": 0.06,
    "fineweb_edu": 0.14,
    "cosmopedia_v2": 0.08,
    "wikihow": 0.06,
    "finemath": 0.08,
    "txt360_qa": 0.07,
    "txt360_nltk_qa": 0.07,
    "word_problems": 0.08,
    "arith": 0.12,
}


@dataclass
class RunConfig:
    """Everything one training run needs. Saved as config.json in the run folder."""

    run_name: str
    run_dir: str  # where ckpt/, metrics.jsonl and evals/ go
    data_dir: str | None = None  # holds tokens/<source>/*.bin. None = random tokens (gate 3 benchmark)
    tokenizer_dir: str | None = None  # for the arithmetic proxy eval. None skips it
    model: str = "main"  # key into PRESETS
    seed: int = 0
    device: str = "auto"

    # batch (D-029). Global batch = batch_seqs x seq_len tokens
    seq_len: int = 1_024
    batch_seqs: int = 256
    micro_batch_seqs: int = 32  # set by memory at gate 3 (Q-17)

    # schedule (D-028). total_steps fixed at gate 3 (D-042)
    total_steps: int = 34_000
    warmup_frac: float = 0.01
    min_warmup_steps: int = 200
    decay_start_frac: float = 0.80

    # TST (D-026). tst_end_frac = 0 turns it off
    tst_bag: int = 6
    tst_end_frac: float = 0.30

    # optimizer (D-027). "muon" = Muon + AdamW, "adamw" = AdamW everywhere (the gate 4 ablation)
    optimizer: str = "muon"
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.01
    muon_ns_steps: int = 5
    adam_lr: float = 1.5e-3
    adam_betas: tuple[float, float] = (0.9, 0.95)
    adam_eps: float = 1e-8
    adam_weight_decay: float = 0.0  # embedding and norms
    adamw_only_matrix_weight_decay: float = 0.1  # block matrices in the AdamW-only ablation
    grad_clip: float = 1.0
    z_loss: float = 1e-4  # D-021

    # mix (D-023). Weights per source name, see data/sources.py
    stable_mix: dict[str, float] = field(default_factory=lambda: dict(STABLE_MIX))
    decay_mix: dict[str, float] = field(default_factory=lambda: dict(DECAY_MIX))

    # precision and speed (D-031)
    precision: str = "bf16"  # "bf16" on GPU, "fp32" for exact tests on the Mac
    compile: bool = True

    # monitoring (D-014)
    log_every: int = 10
    eval_every: int = 250
    eval_at_start: bool = False  # one eval before the first step, e.g. a baseline for a resumed test run (D-045)
    val_windows_per_source: int = 32
    proxy_items: int = 400
    milestone_every_tokens: float = 1e9  # milestone checkpoint for the real benchmarks

    # checkpoints (D-007) and budget (D-015)
    ckpt_every_minutes: float = 20.0
    gpu_usd_per_hour: float = 0.0
    n_gpus: int = 1
    max_usd: float = float("inf")

    # --- derived step counts ---
    @property
    def accum_steps(self) -> int:
        assert self.batch_seqs % self.micro_batch_seqs == 0, "batch_seqs must be a multiple of micro_batch_seqs"
        return self.batch_seqs // self.micro_batch_seqs

    @property
    def tokens_per_step(self) -> int:
        """Compute tokens per optimizer step (positions the model sees). TST reads 6x this in raw tokens."""
        return self.batch_seqs * self.seq_len

    @property
    def decay_start_step(self) -> int:
        return round(self.decay_start_frac * self.total_steps)

    @property
    def warmup_steps(self) -> int:
        return min(max(self.min_warmup_steps, round(self.warmup_frac * self.total_steps)), self.decay_start_step)

    @property
    def tst_end_step(self) -> int:
        return round(self.tst_end_frac * self.total_steps)

    def model_config(self) -> ModelConfig:
        return PRESETS[self.model]

    # --- save / load ---
    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "RunConfig":
        """Fields missing from the JSON keep their defaults."""
        raw = json.loads(text)
        if "adam_betas" in raw:
            raw["adam_betas"] = tuple(raw["adam_betas"])
        return cls(**raw)
