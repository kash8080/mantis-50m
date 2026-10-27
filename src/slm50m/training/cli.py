"""Start or resume a training run, print a run's status, draw its charts (D-014).

    uv run slm-train configs/mac.json
    uv run slm-status _build/runs/mac/metrics.jsonl
    uv run slm-plot _build/runs/mac --watch 30
    uv run modal volume get slm50m-runs main/metrics.jsonl _build/ && uv run slm-status _build/metrics.jsonl

Locally slm-train is for debugging on the Mac. Paid runs go through modal_app.py.
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

from slm50m.config import RunConfig

STALL_SECONDS = 5 * 60


def load_config(path: str, overrides: list[str]) -> RunConfig:
    raw = json.loads(Path(path).read_text())
    fields = {f.name for f in dataclasses.fields(RunConfig)}
    for item in overrides:
        key, value = item.split("=", 1)
        assert key in fields, f"unknown config field {key}"
        try:
            raw[key] = json.loads(value)  # numbers, true/false, null, lists
        except json.JSONDecodeError:
            raw[key] = value  # plain strings like paths or "cpu"
    return RunConfig.from_json(json.dumps(raw))


def train():
    from slm50m.training.trainer import Trainer  # torch import, not needed for slm-status

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--set", nargs="*", default=[], help="field=value overrides")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    reason = Trainer(cfg).run(args.max_steps)
    print(f"stopped: {reason}")


def status():
    """Latest numbers, trends, alerts and stalls from a run's metrics.jsonl."""
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics")
    parser.add_argument("--evals", type=int, default=5, help="how many recent evals to show")
    args = parser.parse_args()
    records = [json.loads(line) for line in Path(args.metrics).read_text().splitlines() if line.strip()]
    by_type: dict[str, list[dict]] = {}
    for r in records:
        by_type.setdefault(r["type"], []).append(r)

    steps = by_type.get("train", [])
    if steps:
        last = steps[-1]
        print(f"step {last['step']}  loss {last['loss']:.4f}  bag {last['bag']}  lr x{last['lr_mult']:.3f}  "
              f"grad {last['grad_norm']:.3f}  {last['tokens_per_s']:,.0f} tok/s  "
              f"{last['tokens'] / 1e9:.3f}B tokens  ${last['usd']:.2f}")
        age = time.time() - last["time"]
        stopped = by_type.get("stop", [{}])[-1].get("time", 0) > last["time"]
        if age > STALL_SECONDS and not stopped:
            print(f"ALERT stall: no new step for {age / 60:.0f} min")

    for e in by_type.get("eval", [])[-args.evals:]:
        vals = "  ".join(f"{k[4:]} {v:.3f}" for k, v in e.items() if k.startswith("val/"))
        proxy = f"  proxy {100 * e['proxy_acc']:.1f}%" if "proxy_acc" in e else ""
        print(f"eval step {e['step']}: {vals}{proxy}")

    for a in by_type.get("alert", []):
        print(f"ALERT step {a['step']} {a['kind']}: {a['message']}")
    for s in by_type.get("stop", [])[-1:]:
        print(f"stopped at step {s['step']}: {s['reason']}")


def plot():
    """plots.png + a self-reloading plots.html in the run folder. --watch redraws until the run stops."""
    from slm50m.training.plot import draw, is_stopped, load  # matplotlib is a dev dependency

    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--watch", type=float, default=0, help="redraw every this many seconds")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    while True:
        out = draw(run_dir)
        print(f"{time.strftime('%H:%M:%S')} wrote {out} (open {run_dir / 'plots.html'} in a browser)")
        if not args.watch or is_stopped(load(run_dir)):
            return
        time.sleep(args.watch)
