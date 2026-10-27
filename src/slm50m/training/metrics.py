"""metrics.jsonl writer and the training alerts (D-014, D-034).

Each line is one JSON object with a "type": start, train, eval, alert, checkpoint or stop.
`uv run slm-status` prints it, `uv run slm-plot` draws it.
"""

import json
import math
import time
from collections import defaultdict, deque
from pathlib import Path


class MetricsLog:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record_type: str, /, **fields) -> None:
        """record_type is positional-only, so records can still carry a field called "kind"."""
        record = {"type": record_type, "time": time.time(), **fields}
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")


class Alerts:
    """Loss spikes and per-source val loss rising while train loss falls (the HobbyLM failure)."""

    def __init__(self, log: MetricsLog, spike_factor: float = 1.5, window: int = 50):
        self.log = log
        self.spike_factor = spike_factor
        self.recent: deque[float] = deque(maxlen=window)
        self.bag = None
        self.val_history: dict[str, list[float]] = defaultdict(list)
        self.train_at_eval: list[float] = []

    def train(self, step: int, loss: float, bag: int) -> None:
        if not math.isfinite(loss):
            self.log.write("alert", step=step, kind="nan", message=f"train loss is {loss}")
            return
        if bag != self.bag:  # the TST switch changes the objective, so start the running average again
            self.recent.clear()
            self.bag = bag
        if len(self.recent) >= 20:
            mean = sum(self.recent) / len(self.recent)
            if loss > self.spike_factor * mean:
                self.log.write("alert", step=step, kind="loss_spike", message=f"loss {loss:.3f} vs recent {mean:.3f}")
        self.recent.append(loss)

    def val(self, step: int, losses: dict[str, float]) -> None:
        train_now = sum(self.recent) / len(self.recent) if self.recent else float("nan")
        self.train_at_eval.append(train_now)
        for source, loss in losses.items():
            hist = self.val_history[source]
            hist.append(loss)
            # rose 3 evals in a row while the train loss fell over the same evals
            if len(hist) >= 4 and hist[-4] < hist[-3] < hist[-2] < hist[-1] \
                    and self.train_at_eval[-1] < self.train_at_eval[-4]:
                self.log.write("alert", step=step, kind="val_rising", source=source,
                               message=f"{source} val loss rose 3 evals in a row while train loss fell")
