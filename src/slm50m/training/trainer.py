"""The training loop.

Same call starts a run or resumes it: on start it loads the newest checkpoint in run_dir/ckpt (D-007).
One optimizer step:
  1. pick the phase from the step: TST bags (first 30%) or plain, stable or decay mix, lr multiplier
  2. for each micro-batch: draw a stratified batch, forward under bf16 autocast, backward
  3. clip the global grad norm, step both optimizers
Then: log, eval, checkpoint, budget check.
"""

import math
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import torch

from slm50m.config import RunConfig
from slm50m.dataprep.mixture import MixtureLoader, RandomTokens
from slm50m.model.qwen3 import Qwen3LM
from slm50m.training.checkpoint import checkpoint_name, latest_checkpoint, prune, save_checkpoint, save_weights
from slm50m.training.loss import lm_loss
from slm50m.training.metrics import Alerts, MetricsLog
from slm50m.training.optim import build_optimizers, set_lr
from slm50m.training.schedule import lr_multiplier
from slm50m.training.tst import forward_loss, window_length

CPU_MEMORY_OVERHEAD = 1.05  # ~5% on top of the GPU price for Modal CPU and memory (D-015)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False,
                             cwd=Path(__file__).parent)
    except OSError:  # no git in the container
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 else "unknown"


class Trainer:
    def __init__(self, cfg: RunConfig, on_save: Callable[[Path, str], None] | None = None):
        """on_save(path, kind) runs after every save. On Modal it commits the Volume and spawns milestone evals."""
        self.cfg = cfg
        self.on_save = on_save or (lambda path, kind: None)
        self.device = resolve_device(cfg.device)
        self.run_dir = Path(cfg.run_dir)
        self.ckpt_dir = self.run_dir / "ckpt"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.json").write_text(cfg.to_json())

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True  # for the few fp32 matmuls outside autocast
            torch.backends.cudnn.allow_tf32 = True

        torch.manual_seed(cfg.seed)
        self.model = Qwen3LM(cfg.model_config()).to(self.device)
        self.optimizers = build_optimizers(self.model, cfg)
        self.params = list(self.model.parameters())

        # sources with any weight in either phase
        self.sources = sorted({s for mix in (cfg.stable_mix, cfg.decay_mix) for s, w in mix.items() if w > 0})
        if cfg.data_dir is None:
            self.loader = RandomTokens(cfg.model_config().vocab_size, cfg.seed)
            self.val = {}
        else:
            self.loader = MixtureLoader(cfg.data_dir, self.sources, cfg.seed)
            self.val = {s: self.loader.val_windows(s, cfg.val_windows_per_source, cfg.seq_len + 1)
                        for s in self.sources}

        self.proxy_items, self.tokenizer = [], None
        if cfg.tokenizer_dir is not None:
            from slm50m.arith.proxy import make_items
            from slm50m.tokenizer.train_tokenizer import load_tokenizer

            self.tokenizer = load_tokenizer(cfg.tokenizer_dir)
            self.proxy_items = make_items(cfg.proxy_items)

        # progress counters, restored on resume
        self.step = 0  # optimizer steps done
        self.tokens = 0  # compute tokens (positions the model saw)
        self.raw_tokens = 0  # tokens read from the data (6x during TST)
        self.seconds = 0.0  # training wall time, summed over all resumes
        self.resumed_from = self._resume()

        # static shapes: the TST switch compiles a second graph for bag 1. Without dynamic=False that recompile
        # makes the window length and bag symbolic, and every plain step runs ~45% slower (26 Oct 2026 log)
        self.loss_fn = torch.compile(forward_loss, dynamic=False) if cfg.compile else forward_loss
        self.autocast_dtype = torch.bfloat16 if cfg.precision == "bf16" else None
        self.log = MetricsLog(self.run_dir / "metrics.jsonl")
        self.alerts = Alerts(self.log)

    # ------------------------------------------------------------------ state

    def usd(self) -> float:
        return self.seconds / 3600 * self.cfg.gpu_usd_per_hour * self.cfg.n_gpus * CPU_MEMORY_OVERHEAD

    def _state(self) -> dict:
        return {"step": self.step, "tokens": self.tokens, "raw_tokens": self.raw_tokens,
                "seconds": self.seconds, "usd": self.usd(), "git_commit": git_commit()}

    def _resume(self) -> str | None:
        path = latest_checkpoint(self.ckpt_dir)
        if path is None:
            return None
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        for opt, state in zip(self.optimizers, ckpt["optimizers"], strict=True):
            opt.load_state_dict(state)
        self.step, self.tokens, self.raw_tokens = ckpt["step"], ckpt["tokens"], ckpt["raw_tokens"]
        self.seconds = ckpt["seconds"]
        return path.name

    def save(self, kind: str) -> Path:
        state = self._state()
        if kind == "milestone":
            path = self.run_dir / "milestones" / f"milestone_{self.step:07d}.pt"
            save_weights(path, self.model, state)
        else:
            path = self.ckpt_dir / checkpoint_name(kind, self.step)
            save_checkpoint(path, self.model, self.optimizers, self.cfg, state)
            prune(self.ckpt_dir)
        self.log.write("checkpoint", step=self.step, kind=kind, path=str(path))
        self.on_save(path, kind)
        return path

    # ------------------------------------------------------------------ one step

    def phase(self, step: int) -> tuple[int, dict[str, float]]:
        """(bag size, mix) for a step. Bag 1 is plain next-token training."""
        cfg = self.cfg
        bag = cfg.tst_bag if step < cfg.tst_end_step else 1
        mix = cfg.stable_mix if step < cfg.decay_start_step else cfg.decay_mix
        return bag, mix

    def train_step(self) -> dict:
        cfg = self.cfg
        bag, mix = self.phase(self.step)
        mult = lr_multiplier(self.step, cfg.total_steps, cfg.warmup_steps, cfg.decay_start_step)
        set_lr(self.optimizers, mult)

        ce_sum = torch.zeros((), device=self.device)
        z_sum = torch.zeros((), device=self.device)
        for micro in range(cfg.accum_steps):
            window, _ = self.loader.batch(mix, self.step, micro, cfg.micro_batch_seqs, window_length(cfg.seq_len, bag))
            window = window.to(self.device, non_blocking=True)
            with torch.autocast(self.device.type, dtype=self.autocast_dtype, enabled=self.autocast_dtype is not None):
                loss, ce, z = self.loss_fn(self.model, window, bag, cfg.z_loss)
            (loss / cfg.accum_steps).backward()
            ce_sum += ce
            z_sum += z

        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, cfg.grad_clip)
        for opt in self.optimizers:
            opt.step()
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)

        self.step += 1
        self.tokens += cfg.tokens_per_step
        self.raw_tokens += cfg.tokens_per_step * bag
        return {"loss": (ce_sum / cfg.accum_steps).item(), "z": (z_sum / cfg.accum_steps).item(),
                "grad_norm": grad_norm.item(), "lr_mult": mult, "bag": bag}

    # ------------------------------------------------------------------ eval

    @torch.no_grad()
    def evaluate(self) -> dict:
        """Plain next-token val loss per source (comparable across the TST switch) + arithmetic proxy."""
        self.model.eval()
        out = {}
        for source, windows in self.val.items():
            if len(windows) == 0:
                continue
            total = 0.0
            for chunk in windows.split(self.cfg.micro_batch_seqs):
                chunk = chunk.to(self.device)
                with torch.autocast(self.device.type, dtype=self.autocast_dtype, enabled=self.autocast_dtype is not None):
                    logits = self.model(input_ids=chunk[:, :-1])
                total += lm_loss(logits, chunk[:, 1:])[1].item() * len(chunk)
            out[f"val/{source}"] = total / len(windows)
        if out:
            out["val/mean"] = sum(out.values()) / len(out)
        if self.proxy_items:
            from slm50m.arith.proxy import score_items

            out["proxy_acc"] = score_items(self.model, self.tokenizer, self.proxy_items, self.device)
        self.model.train()
        return out

    # ------------------------------------------------------------------ loop

    def run(self, max_steps: int | None = None, deadline: float | None = None) -> str:
        """Train to total_steps, or until the budget stop, the wall-clock deadline (a time.time() value) or
        max_steps more steps. Returns why it stopped."""
        cfg = self.cfg
        stop_at = cfg.total_steps if max_steps is None else min(cfg.total_steps, self.step + max_steps)
        gpu = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None
        self.log.write("start", step=self.step, resumed_from=self.resumed_from, device=str(self.device), gpu=gpu,
                       n_params=self.model.n_params(), git_commit=git_commit(),
                       shards={s: [p.name for p in sh.paths] for s, sh in getattr(self.loader, "shards", {}).items()})
        self.model.train()
        last_save = time.time()
        milestone = int(self.tokens // cfg.milestone_every_tokens)

        while self.step < stop_at:
            t0 = time.time()
            m = self.train_step()
            dt = time.time() - t0
            self.seconds += dt

            if not math.isfinite(m["loss"]):
                # don't save: resume from the last good checkpoint instead
                self.alerts.train(self.step, m["loss"], m["bag"])
                self.log.write("stop", step=self.step, reason="nan")
                raise FloatingPointError(f"loss is {m['loss']} at step {self.step}")
            self.alerts.train(self.step, m["loss"], m["bag"])

            if self.step % cfg.log_every == 0 or self.step == 1:
                self.log.write("train", step=self.step, tokens=self.tokens, raw_tokens=self.raw_tokens,
                               tokens_per_s=cfg.tokens_per_step / dt, usd=self.usd(), **m)
            if cfg.eval_every and self.step % cfg.eval_every == 0 and (self.val or self.proxy_items):
                evals = self.evaluate()
                self.log.write("eval", step=self.step, tokens=self.tokens, **evals)
                self.alerts.val(self.step, {k: v for k, v in evals.items() if k.startswith("val/")})

            if self.step == cfg.decay_start_step:
                self.save("predecay")
            if int(self.tokens // cfg.milestone_every_tokens) > milestone:
                milestone = int(self.tokens // cfg.milestone_every_tokens)
                self.save("milestone")
            if time.time() - last_save > cfg.ckpt_every_minutes * 60:
                self.save("step")
                last_save = time.time()
            if self.usd() >= cfg.max_usd:
                self.save("step")
                self.log.write("stop", step=self.step, reason="budget", usd=self.usd())
                return "budget"
            # usd counts step time only. The deadline also bounds compile, evals and resumes (06-infra-modal.md)
            if deadline is not None and time.time() >= deadline and self.step < stop_at:
                self.save("step")
                self.log.write("stop", step=self.step, reason="deadline", usd=self.usd())
                return "deadline"

        if self.step >= cfg.total_steps:
            self.save("final")
            self.log.write("stop", step=self.step, reason="done", usd=self.usd())
            return "done"
        self.save("step")
        self.log.write("stop", step=self.step, reason="max_steps", usd=self.usd())
        return "max_steps"
