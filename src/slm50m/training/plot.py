"""Charts of a run's metrics.jsonl, for watching local runs (D-034).

    uv run slm-plot _build/runs/mac               # writes plots.png + plots.html in the run folder
    uv run slm-plot _build/runs/mac --watch 30    # redraws every 30 s until the run stops

Open plots.html in a browser: it reloads itself every 30 s, so it follows a live run.
Dashed lines mark the TST end and the decay start, dotted lines a resume. Train loss jumps at the
TST end because the objective changes there (6 bag targets -> 1 next token), not because training broke.
"""

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt

from slm50m.config import RunConfig

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
LINE = "#2a78d6"  # one series per panel, so one hue everywhere
MARK = "#8a8984"  # phase and resume lines
REFRESH_SECONDS = 30


def load(run_dir: Path) -> dict[str, list[dict]]:
    """Records by type, in file order. For train and eval, a resume's record replaces the earlier one at that step."""
    out: dict[str, list[dict]] = {"start": [], "stop": [], "alert": []}
    by_step: dict[str, dict[int, dict]] = {"train": {}, "eval": {}}
    for line in (run_dir / "metrics.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["type"] in by_step:
            by_step[r["type"]][r["step"]] = r
        else:
            out.setdefault(r["type"], []).append(r)
    for kind, rows in by_step.items():
        out[kind] = [rows[s] for s in sorted(rows)]
    return out


def is_stopped(data: dict[str, list[dict]]) -> bool:
    """A stop record newer than the last start."""
    return bool(data["stop"]) and (not data["start"] or data["stop"][-1]["time"] > data["start"][-1]["time"])


def _style(ax, title: str, ylabel: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=6)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=INK_2)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=7, length=0)


def _markers(ax, cfg: RunConfig | None, resumes: list[int]) -> None:
    if cfg is not None:
        if cfg.tst_end_step > 0:
            ax.axvline(cfg.tst_end_step, color=MARK, linewidth=0.8, linestyle="--")
        ax.axvline(cfg.decay_start_step, color=MARK, linewidth=0.8, linestyle="--")
    for step in resumes:
        ax.axvline(step, color=MARK, linewidth=0.8, linestyle=":")


def _line(ax, rows: list[dict], key: str, points: bool = False) -> None:
    rows = [r for r in rows if key in r]
    if rows:
        ax.plot([r["step"] for r in rows], [r[key] for r in rows], color=LINE, linewidth=1.5,
                marker="o" if points else None, markersize=4)


def draw(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    data = load(run_dir)
    train, evals = data.get("train", []), data.get("eval", [])
    cfg_path = run_dir / "config.json"
    cfg = RunConfig.from_json(cfg_path.read_text()) if cfg_path.exists() else None
    resumes = [s["step"] for s in data["start"] if s["step"] > 0]
    sources = sorted({k[4:] for e in evals for k in e if k.startswith("val/") and k != "val/mean"})

    n_src_rows = (len(sources) + 4) // 5
    fig = plt.figure(figsize=(15, 6.4 + 2.4 * n_src_rows), facecolor=SURFACE)
    grid = fig.add_gridspec(2 + n_src_rows, 5, height_ratios=[3, 2.2] + [2] * n_src_rows, hspace=0.55, wspace=0.3,
                            top=0.95)

    top = [("Train loss", "loss", False, fig.add_subplot(grid[0, 0:2])),
           ("Val loss, mean of sources", "val/mean", True, fig.add_subplot(grid[0, 2:4])),
           ("Arithmetic proxy accuracy", "proxy_acc", True, fig.add_subplot(grid[0, 4]))]
    for title, key, points, ax in top:
        _style(ax, title)
        _line(ax, evals if points else train, key, points)
        _markers(ax, cfg, resumes)
    top[2][3].axhline(0.25, color=MARK, linewidth=0.8)  # chance on 4 choices
    top[2][3].annotate("chance", (0, 0.25), xycoords=("axes fraction", "data"), xytext=(2, 3),
                       textcoords="offset points", fontsize=7, color=INK_2)

    for i, (title, key) in enumerate([("LR multiplier", "lr_mult"), ("Grad norm (before clip)", "grad_norm"),
                                      ("Tokens / s", "tokens_per_s"), ("z-loss", "z"), ("Cost, $", "usd")]):
        ax = fig.add_subplot(grid[1, i])
        _style(ax, title)
        _line(ax, train, key)
        _markers(ax, cfg, resumes)

    # one small panel per source, same y range, so the sources compare by position
    vals = [e[f"val/{s}"] for e in evals for s in sources if f"val/{s}" in e]
    first = None
    for i, source in enumerate(sources):
        ax = fig.add_subplot(grid[2 + i // 5, i % 5], sharey=first)
        first = first or ax
        _style(ax, f"Val loss: {source}")
        _line(ax, evals, f"val/{source}", points=True)
        _markers(ax, cfg, resumes)
    if vals:
        pad = 0.05 * (max(vals) - min(vals) or 1)
        first.set_ylim(min(vals) - pad, max(vals) + pad)

    last = train[-1] if train else {}
    alerts = data["alert"]
    status = f"stopped: {data['stop'][-1]['reason']}" if is_stopped(data) else "running"
    head = f"{run_dir.name}  ·  step {last.get('step', 0):,}  ·  {last.get('tokens', 0) / 1e6:,.1f}M tokens  ·  {status}"
    if alerts:
        head += f"  ·  {len(alerts)} alert(s), last: {alerts[-1]['kind']} at step {alerts[-1]['step']}"
    fig.suptitle(head, x=0.125, y=0.995, ha="left", fontsize=12, color=INK)
    fig.text(0.125, 0.975, "x axis: optimizer step.  Dashed: TST end, decay start.  Dotted: resume.",
             fontsize=8, color=INK_2, va="top")

    out = run_dir / "plots.png"
    fig.savefig(out, dpi=110, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    (run_dir / "plots.html").write_text(
        f'<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="{REFRESH_SECONDS}">'
        f'<title>{run_dir.name} metrics</title><body style="margin:0;background:{SURFACE}">'
        f'<img src="plots.png?{int(time.time())}" style="max-width:100%">')
    return out
