"""WSD learning-rate schedule (D-028): linear warmup, flat, then 1 - sqrt decay to 0.

Returns a multiplier. Both optimizers scale their own peak lr by it.
"""

import math


def lr_multiplier(step: int, total_steps: int, warmup_steps: int, decay_start: int) -> float:
    """step is 0-based: the index of the optimizer step about to run."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if step < decay_start:
        return 1.0
    t = (step - decay_start) / (total_steps - decay_start)  # 0 at decay start, -> 1 at the end
    return 1.0 - math.sqrt(t)
