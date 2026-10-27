"""Optimizers (D-027): Muon for the block matrices, AdamW for the embedding and norms.

`optimizer="adamw"` is the gate 4 ablation: AdamW everywhere, weight decay 0.1 on the matrices.
"""

from collections import defaultdict

import torch
from torch.optim._muon import _adjust_lr

from slm50m.config import RunConfig


def newton_schulz(g: torch.Tensor, coefficients: tuple[float, float, float], steps: int, eps: float) -> torch.Tensor:
    """torch.optim.Muon's orthogonalization on a stack (n, A, B) of same-shape matrices at once."""
    a, b, c = coefficients
    x = g.bfloat16()
    tall = g.size(-2) > g.size(-1)
    if tall:
        x = x.mT
    x = x / x.norm(dim=(-2, -1), keepdim=True).clamp(min=eps)  # spectral norm at most 1
    for _ in range(steps):
        gram = x @ x.mT
        x = torch.baddbmm(x, torch.baddbmm(gram, gram, gram, beta=b, alpha=c), x, beta=a)
    return x.mT if tall else x


class BatchedMuon(torch.optim.Muon):
    """torch.optim.Muon with one Newton-Schulz per group of same-shape matrices (e.g. all 25 layers' q and o),
    not one per matrix. Same math, same state, so checkpoints load either way. torch's loop launched ~4,000 small
    kernels per step and left the H100 idle ~12% of the step (27 Oct 2026 profile)."""

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is None
        for group in self.param_groups:
            lr, momentum, wd = group["lr"], group["momentum"], group["weight_decay"]
            by_shape = defaultdict(list)
            for p in group["params"]:
                if p.grad is not None:
                    by_shape[p.shape].append(p)
            for shape, params in by_shape.items():
                grads = [p.grad for p in params]
                for p in params:
                    if "momentum_buffer" not in self.state[p]:
                        self.state[p]["momentum_buffer"] = torch.zeros_like(p.grad)
                bufs = [self.state[p]["momentum_buffer"] for p in params]
                torch._foreach_lerp_(bufs, grads, 1 - momentum)
                updates = torch._foreach_lerp(grads, bufs, momentum) if group["nesterov"] else bufs
                ortho = newton_schulz(torch.stack(updates), group["ns_coefficients"], group["ns_steps"], group["eps"])
                torch._foreach_mul_(params, 1 - lr * wd)
                torch._foreach_add_(params, list(ortho.to(params[0].dtype).unbind(0)),
                                    alpha=-_adjust_lr(lr, group["adjust_lr_fn"], shape))


def split_params(model) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """(2D matrices inside the blocks: q, k, v, o, gate, up, down) and (tied embedding, all norm scales)."""
    matrices, others = [], []
    for name, p in model.named_parameters():
        (matrices if name.startswith("layers.") and p.dim() == 2 else others).append(p)
    return matrices, others


def build_optimizers(model, cfg: RunConfig) -> list[torch.optim.Optimizer]:
    matrices, others = split_params(model)
    adam_kwargs = {"lr": cfg.adam_lr, "betas": cfg.adam_betas, "eps": cfg.adam_eps}
    if cfg.optimizer == "muon":
        optimizers = [
            BatchedMuon(matrices, lr=cfg.muon_lr, momentum=cfg.muon_momentum, nesterov=True,
                             weight_decay=cfg.muon_weight_decay, ns_steps=cfg.muon_ns_steps,
                             adjust_lr_fn="original"),  # Keller's sqrt(max(1, rows / cols)) scaling
            torch.optim.AdamW(others, weight_decay=cfg.adam_weight_decay, **adam_kwargs),
        ]
    elif cfg.optimizer == "adamw":
        optimizers = [torch.optim.AdamW([
            {"params": matrices, "weight_decay": cfg.adamw_only_matrix_weight_decay},
            {"params": others, "weight_decay": cfg.adam_weight_decay},
        ], **adam_kwargs)]
    else:
        raise ValueError(f"unknown optimizer {cfg.optimizer!r}")
    for opt in optimizers:
        for group in opt.param_groups:
            group["base_lr"] = group["lr"]  # the schedule multiplies this peak value
    return optimizers


def set_lr(optimizers: list[torch.optim.Optimizer], multiplier: float) -> None:
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["base_lr"] * multiplier
