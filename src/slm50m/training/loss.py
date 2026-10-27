"""Cross-entropy with z-loss, in fp32 (D-021, D-031)."""

import torch


def lm_loss(logits: torch.Tensor, targets: torch.Tensor, z_loss: float = 0.0):
    """Returns (loss to backprop, cross-entropy, z) where the last two are detached for logging.

    logits (B, S, V). targets (B, S) for next-token, or (B, S, k) for TST: then the cross-entropy is
    the plain mean of the k ordinary cross-entropies (TST paper eq. 3 without its constant log k).
    z = mean(logsumexp^2), weighted by z_loss.
    """
    logits = logits.float()
    lse = torch.logsumexp(logits, dim=-1)  # (B, S)
    if targets.dim() == lse.dim():
        targets = targets.unsqueeze(-1)  # (B, S, 1)
    target_logits = logits.gather(-1, targets)  # (B, S, k)
    ce = (lse.unsqueeze(-1) - target_logits).mean()  # -log softmax at each target, averaged
    z = lse.pow(2).mean()
    return ce + z_loss * z, ce.detach(), z.detach()
