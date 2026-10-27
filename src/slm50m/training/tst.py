"""Token-Superposition Training (D-026). Paper: arXiv 2605.06546 v2, appendix A.

A window of (S + 1) x s raw tokens is cut into S + 1 bags of s tokens. The input at position t is
the mean of the s embeddings in bag t. Bag t predicts the s tokens of bag t + 1. RoPE positions
are the bag indices, so the model sees S positions and a step costs the same FLOPs as plain training.

Plain next-token training is the same code with s = 1, which keeps one path through the trainer.
"""

import torch
import torch.nn.functional as F

from slm50m.training.loss import lm_loss


def window_length(seq_len: int, bag: int) -> int:
    """Raw tokens one sequence reads: seq_len + 1 for plain training, 6 x that for TST."""
    return (seq_len + 1) * bag


def split_window(window: torch.Tensor, bag: int) -> tuple[torch.Tensor, torch.Tensor]:
    """window (B, (S+1) x bag) -> inputs (B, S x bag) and targets (B, S, bag), target bag t = input bag t + 1."""
    B, L = window.shape
    S = L // bag - 1
    inputs = window[:, :S * bag]
    targets = window[:, bag:].reshape(B, S, bag)
    return inputs, targets


def bag_embeddings(embed_weight: torch.Tensor, inputs: torch.Tensor, bag: int) -> torch.Tensor:
    """inputs (B, S x bag) -> (B, S, D): each bag's embeddings summed in fp32, then divided by bag."""
    B, L = inputs.shape
    e = F.embedding(inputs, embed_weight).float()  # (B, S x bag, D)
    return (e.view(B, L // bag, bag, -1).sum(dim=2) / bag).to(embed_weight.dtype)


def forward_loss(model, window: torch.Tensor, bag: int, z_loss: float):
    """One micro-batch forward. Same model, same embedding and head in both phases."""
    inputs, targets = split_window(window, bag)
    if bag == 1:
        logits = model(input_ids=inputs)
    else:
        logits = model(inputs_embeds=bag_embeddings(model.embed_tokens.weight, inputs, bag))
    return lm_loss(logits, targets, z_loss)
