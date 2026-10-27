"""Gate 0: schedule shape, TST on hand-made examples, overfitting, exact resume, the TST switch."""

import copy
import dataclasses
import json
import math
import time

import pytest
import torch
import torch.nn.functional as F

from slm50m.config import PRESETS, RunConfig
from slm50m.model.qwen3 import Qwen3LM
from slm50m.training.metrics import Alerts, MetricsLog
from slm50m.training.optim import BatchedMuon, build_optimizers, split_params
from slm50m.training.schedule import lr_multiplier
from slm50m.training.trainer import Trainer
from slm50m.training.tst import bag_embeddings, forward_loss, split_window
from tests.conftest import SOURCES


def test_schedule_shape():
    total, warmup, decay = 10_000, 100, 8_000
    lr = [lr_multiplier(s, total, warmup, decay) for s in range(total)]
    assert lr[0] == 1 / warmup and lr[warmup - 1] == 1.0  # linear warmup over the first 1%
    assert all(x == 1.0 for x in lr[warmup:decay])  # flat to 80%
    assert lr[decay + 1_000] == 1 - math.sqrt(0.5)  # 1 - sqrt in the decay
    assert all(a > b for a, b in zip(lr[decay:], lr[decay + 1:]))
    assert lr[-1] < 0.03  # ends near 0


def test_tst_bags_targets_and_loss():
    bag, S, B = 6, 4, 2
    window = torch.arange(B * (S + 1) * bag).reshape(B, (S + 1) * bag) % 50
    inputs, targets = split_window(window, bag)
    # bag t covers tokens 6t..6t+5 and predicts tokens 6t+6..6t+11
    assert inputs.shape == (B, S * bag) and targets.shape == (B, S, bag)
    for t in range(S):
        assert targets[0, t].tolist() == window[0, 6 * t + 6:6 * t + 12].tolist()

    weight = torch.randn(50, 8)
    emb = bag_embeddings(weight, inputs, bag)
    assert torch.allclose(emb[1, 2], weight[inputs[1, 12:18]].mean(0), atol=1e-6)  # mean of the 6 embeddings

    torch.manual_seed(0)
    model = Qwen3LM(dataclasses.replace(PRESETS["test"], vocab_size=50))
    _, ce, _ = forward_loss(model, window, bag, z_loss=0.0)
    logits = model(inputs_embeds=bag_embeddings(model.embed_tokens.weight, inputs, bag))
    expected = sum(F.cross_entropy(logits.reshape(-1, 50), targets[..., k].reshape(-1)) for k in range(bag)) / bag
    assert torch.allclose(ce, expected, atol=1e-6)


def test_tiny_model_overfits_one_batch():
    torch.manual_seed(0)
    model = Qwen3LM(PRESETS["test"])
    cfg = RunConfig(run_name="t", run_dir="unused", model="test", muon_lr=0.02, adam_lr=1e-2)
    optimizers = build_optimizers(model, cfg)
    window = torch.randint(0, PRESETS["test"].vocab_size, (4, 33))
    for _ in range(300):
        loss, ce, _ = forward_loss(model, window, 1, z_loss=1e-4)
        loss.backward()
        for opt in optimizers:
            opt.step()
            opt.zero_grad()
    assert ce.item() < 0.05


def test_batched_muon_matches_torch_muon():
    """Same params and momentum as torch.optim.Muon after a few steps, bitwise on CPU, and the state loads across."""
    torch.manual_seed(0)
    model = Qwen3LM(PRESETS["test"])
    ours, theirs = split_params(model)[0], split_params(copy.deepcopy(model))[0]
    kwargs = {"lr": 0.02, "momentum": 0.95, "nesterov": True, "weight_decay": 0.01, "adjust_lr_fn": "original"}
    batched, reference = BatchedMuon(ours, **kwargs), torch.optim.Muon(theirs, **kwargs)
    for _ in range(4):
        for p, q in zip(ours, theirs, strict=True):
            p.grad = torch.randn_like(p)
            q.grad = p.grad.clone()
        batched.step()
        reference.step()
    for p, q in zip(ours, theirs, strict=True):
        assert torch.equal(p, q)
        assert torch.equal(batched.state[p]["momentum_buffer"], reference.state[q]["momentum_buffer"])
    batched.load_state_dict(reference.state_dict())


def tiny_run(tmp_path, data_dir, name, **overrides) -> RunConfig:
    mix = {s: 1 / len(SOURCES) for s in SOURCES}
    return RunConfig(
        run_name=name, run_dir=str(tmp_path / name), data_dir=str(data_dir), model="test", device="cpu",
        precision="fp32", compile=False, seq_len=16, batch_seqs=8, micro_batch_seqs=4, total_steps=10,
        min_warmup_steps=2, tst_end_frac=0.3, stable_mix=mix, decay_mix={"alpha": 0.5, "beta": 0.5},
        log_every=1, eval_every=5, val_windows_per_source=4, **overrides,
    )


def losses(run_dir) -> dict[int, float]:
    lines = (run_dir / "metrics.jsonl").read_text().splitlines()
    return {r["step"]: r["loss"] for r in map(json.loads, lines) if r["type"] == "train"}


def test_resume_gives_identical_losses(tmp_path, data_dir):
    """Stop after 4 steps, resume in a new Trainer: steps 5-10 match an uninterrupted run exactly.
    The run crosses the TST switch (step 3) and the decay start (step 8)."""
    full = Trainer(tiny_run(tmp_path, data_dir, "full"))
    assert full.run() == "done"

    cfg = tiny_run(tmp_path, data_dir, "split")
    assert Trainer(cfg).run(max_steps=4) == "max_steps"
    resumed = Trainer(cfg)
    assert resumed.resumed_from == "step_0000004.pt" and resumed.step == 4
    assert resumed.run() == "done"

    a, b = losses(tmp_path / "full"), losses(tmp_path / "split")
    assert len(a) == 10 and a == b
    assert (tmp_path / "split" / "ckpt" / "predecay_0000008.pt").exists()
    assert (tmp_path / "split" / "ckpt" / "final_0000010.pt").exists()


def test_deadline_stops_and_resumes(tmp_path, data_dir):
    """A passed deadline stops after one step with a checkpoint. The next run resumes from it."""
    cfg = tiny_run(tmp_path, data_dir, "deadline")
    assert Trainer(cfg).run(deadline=time.time() - 1) == "deadline"
    resumed = Trainer(cfg)
    assert resumed.resumed_from == "step_0000001.pt"
    assert resumed.run(deadline=time.time() + 3600) == "done"


def records(run_dir) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]


def test_nan_loss_alerts_and_stops_without_saving(tmp_path, data_dir):
    """A NaN loss writes an alert and a stop, raises (so Modal retries from the last good checkpoint) and saves nothing."""
    trainer = Trainer(tiny_run(tmp_path, data_dir, "nan"))
    step = trainer.train_step
    trainer.train_step = lambda: {**step(), "loss": float("nan")}
    with pytest.raises(FloatingPointError):
        trainer.run()
    recs = records(tmp_path / "nan")
    assert [r["kind"] for r in recs if r["type"] == "alert"] == ["nan"]
    assert recs[-1]["type"] == "stop" and recs[-1]["reason"] == "nan"
    assert not list((tmp_path / "nan").rglob("*.pt"))


def test_loss_spike_alert_and_tst_switch_reset(tmp_path):
    """A loss 1.5x the recent mean alerts. The jump at the TST switch (new objective) doesn't."""
    log = MetricsLog(tmp_path / "metrics.jsonl")
    alerts = Alerts(log)
    for step in range(30):
        alerts.train(step, 6.0, bag=6)
    alerts.train(30, 3.0, bag=1)  # the switch: plain loss is lower and the window starts again
    for step in range(31, 60):
        alerts.train(step, 3.0, bag=1)
    assert not (tmp_path / "metrics.jsonl").exists()  # nothing written, so no alert
    alerts.train(60, 5.0, bag=1)
    assert [r["kind"] for r in records(tmp_path) if r["type"] == "alert"] == ["loss_spike"]


def test_tst_switch_keeps_model_and_optimizer_state(tmp_path, data_dir):
    trainer = Trainer(tiny_run(tmp_path, data_dir, "switch"))
    optimizers, params = trainer.optimizers, [id(p) for p in trainer.model.parameters()]
    assert trainer.phase(2)[0] == 6 and trainer.phase(3)[0] == 1
    trainer.run(max_steps=4)  # 3 TST steps, then 1 plain step
    assert trainer.optimizers is optimizers and [id(p) for p in trainer.model.parameters()] == params
    adam = trainer.optimizers[1]
    # AdamW's step counter kept counting through the switch: its moments were not reset
    assert all(adam.state[p]["step"].item() == 4 for p in adam.param_groups[0]["params"])
    muon = trainer.optimizers[0]
    assert all("momentum_buffer" in muon.state[p] for p in muon.param_groups[0]["params"])
