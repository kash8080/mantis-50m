"""Modal app: image, Volumes and functions (docs/06-infra-modal.md).

Every paid entry point prints its worst-case cost and waits for a yes (D-015, CLAUDE.md). Never pass
--yes without Rahul's approval of that run.

    uv run modal run modal_app.py::upload_tokenizer                       # copy outputs/tokenizer to the data Volume
    uv run modal run modal_app.py::prep_benchmarks                         # CPU: benchmark items for the 13-gram filter
    uv run modal run modal_app.py::prep --source wikihow --n-files 1       # CPU: parquet -> shards + manifests/
    uv run modal run modal_app.py::prep_arith --n-shards 4 --tokens-per-shard 50000000
    uv run modal run modal_app.py::prep_arith --source arith_v2 --n-shards 4 --tokens-per-shard 75000000  # D-045
    uv run modal run modal_app.py::sync_manifests                          # free: manifests on the Mac <-> Volume
    uv run modal run modal_app.py::overlap                                 # CPU: cross-source duplicates -> reports/
    uv run modal run modal_app.py::run_benchmark --gpu H100 --data --tst   # gate 3 throughput (no --data: random)
    uv run modal run modal_app.py::run_train --config configs/smoke.json --gpu L4 --hours 0.4
    uv run modal run --detach modal_app.py::run_train --config configs/main.json --gpu H100 --hours 5.5
    uv run modal run modal_app.py::run_eval --run main --checkpoint ckpt/final_0034000.pt
"""

import json
import time
from pathlib import Path

import modal

# $/hour, modal.com/pricing on 24 Oct 2026 (H200: 26 Oct) (docs/06-infra-modal.md)
GPU_PRICES = {"H200": 4.54, "H100": 3.95, "A100-80GB": 2.50, "L40S": 1.95, "A10": 1.10, "L4": 0.80}
CPU_CORE_HOUR = 0.0000131 * 3600
MEM_GIB_HOUR = 0.00000222 * 3600
PREP_CONTAINER_HOUR = 2 * CPU_CORE_HOUR + 8 * MEM_GIB_HOUR  # 2 cores, 8 GiB
OVERHEAD = 1.05
IDLE = 2  # seconds a container stays up after its last call. Modal's default is 60, and idle time is billed

app = modal.App("slm50m")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(frozen=True)  # same pins as uv.lock (D-035)
    .add_local_python_source("slm50m")
)
data_vol = modal.Volume.from_name("slm50m-data", create_if_missing=True)
runs_vol = modal.Volume.from_name("slm50m-runs", create_if_missing=True)
DATA, RUNS = "/data", "/runs"
BUILD_PROFILE = Path(__file__).parent / "_build" / "profile"  # on the Mac: benchmark traces
# HF read token as HF_TOKEN, for every function that downloads from the Hub. Anonymous use is
# rate-limited (3,000 requests per 5 min). Create it once: Modal dashboard > Secrets, name "huggingface"
hf_secret = modal.Secret.from_name("huggingface")


def confirm(what: str, usd: float, yes: bool) -> None:
    print(f"{what}: worst case ${usd:.2f}")
    if yes:
        return
    # the question on its own line: Modal's live "Running (n containers)" status overwrites an input() prompt
    print("Spend this? Type y and press Enter (anything else cancels)")
    if input().strip().lower() != "y":
        raise SystemExit("cancelled, nothing spent")


# ---------------------------------------------------------------- data (gate 1, CPU)

@app.function(image=image, scaledown_window=IDLE, volumes={DATA: data_vol})
def _put_tokenizer(files: dict[str, bytes]) -> None:
    out = Path(DATA) / "tokenizer"
    out.mkdir(parents=True, exist_ok=True)
    for name, blob in files.items():
        (out / name).write_bytes(blob)
    data_vol.commit()


@app.local_entrypoint()
def upload_tokenizer(tokenizer_dir: str = "outputs/tokenizer"):
    files = {p.name: p.read_bytes() for p in Path(tokenizer_dir).iterdir() if p.is_file()}
    _put_tokenizer.remote(files)
    print(f"uploaded {sorted(files)} to {DATA}/tokenizer")


def _hf_user() -> str:
    """The HF account the container's token belongs to. Fails instead of downloading anonymously."""
    import os

    from huggingface_hub import whoami

    if not os.environ.get("HF_TOKEN", "").strip():
        raise RuntimeError("HF_TOKEN is empty in the container: check the 'huggingface' secret")
    return whoami()["name"]


@app.function(image=image, scaledown_window=IDLE, secrets=[hf_secret], timeout=600)
def hf_files(source: str) -> tuple[str, list[list[str]]]:
    """HF user, and the parquet files of each part of a source. Listed in a container, not on the Mac, where
    Python's HF calls hang on broken IPv6 (26 Oct 2026)."""
    from slm50m.dataprep.sources import SOURCES, list_parquet_files

    return _hf_user(), [list_parquet_files(part) for part in SOURCES[source].parts]


@app.function(image=image, scaledown_window=IDLE, volumes={DATA: data_vol}, secrets=[hf_secret], cpu=2, memory=8192,
              timeout=3600)
def _prep_benchmarks() -> tuple[str, dict]:
    from slm50m.dataprep.benchmarks import iter_benchmark_texts

    user = _hf_user()
    out = Path(DATA) / "benchmarks" / "items.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with open(out, "w") as f:
        for bench, text in iter_benchmark_texts():
            f.write(json.dumps({"benchmark": bench, "text": text}) + "\n")
            counts[bench] = counts.get(bench, 0) + 1
    data_vol.commit()
    return user, counts


@app.local_entrypoint()
def prep_benchmarks():
    """Write the benchmark texts the 13-gram filter checks against, once. Prints the items per benchmark."""
    user, counts = _prep_benchmarks.remote()
    print(f"HF token in the container: {user}")
    print(json.dumps(counts, indent=2))


_FILTER = None


def _bench_filter():
    """Built once per container from the items prep_benchmarks wrote."""
    global _FILTER
    if _FILTER is None:
        from slm50m.dataprep.decontam import BenchmarkFilter

        with open(Path(DATA) / "benchmarks" / "items.jsonl") as f:
            _FILTER = BenchmarkFilter((r["benchmark"], r["text"]) for r in map(json.loads, f))
    return _FILTER


@app.function(image=image, scaledown_window=IDLE, volumes={DATA: data_vol}, secrets=[hf_secret], cpu=2, memory=8192,
              timeout=2 * 3600, retries=2, max_containers=20)
def prep_file(source: str, part_index: int, filename: str, shard_index: int, val_docs: int) -> dict:
    from slm50m.dataprep.prep import prep_hf_file
    from slm50m.tokenizer.train_tokenizer import load_tokenizer

    entry = prep_hf_file(source, part_index, filename, shard_index, DATA, load_tokenizer(Path(DATA) / "tokenizer"),
                         _bench_filter(), "/tmp/hf", val_docs)
    data_vol.commit()
    return entry


def _read_volume_json(path: str) -> dict | None:
    try:
        return json.loads(b"".join(data_vol.read_file(path)))
    except FileNotFoundError:
        return None


def _save_manifest(source: str, entries: list[dict]) -> list[dict]:
    """Add entries to manifests/<source>.json on the Mac, copy the file to the Volume. Returns all its entries."""
    from slm50m.dataprep.prep import read_manifest, write_manifest

    path = write_manifest(source, entries)
    with data_vol.batch_upload(force=True) as batch:
        batch.put_file(path, f"/manifests/{source}.json")
    return read_manifest(source)


def _synced_manifest(source: str) -> list[dict]:
    """Every finished shard of a source: the Mac's manifest, the Volume's copy, and any shard on the Volume that
    neither lists (a prep that died before writing them, D-036). Saves the result to both copies.

    Reads the Volume from the Mac, no container. Stops on a shard with no full entry next to it: it can't be
    traced to a source file, and the trainer reads every train_*.bin, so it must be deleted first.
    """
    from modal.exception import NotFoundError

    from slm50m.dataprep.prep import merge_entries, read_manifest, unlisted_shards

    copy = _read_volume_json(f"manifests/{source}.json")
    entries = merge_entries(read_manifest(source), copy["files"] if copy else [])
    try:
        names = [Path(e.path).name for e in data_vol.listdir(f"tokens/{source}")]
    except NotFoundError:
        names = []
    found, untraced = [], []
    for name in unlisted_shards(entries, (n for n in names if n.startswith("train_") and n.endswith(".bin"))):
        entry = _read_volume_json(f"tokens/{source}/{Path(name).with_suffix('.json')}")
        if entry and "file" in entry:
            found.append(entry)
        else:
            untraced.append(name)
    if untraced:
        rm = "\n".join(f"  uv run modal volume rm slm50m-data tokens/{source}/{n}" for n in untraced)
        raise SystemExit(f"{source}: shards on the Volume that no manifest lists and no entry explains. "
                         f"Delete them, then run again:\n{rm}")
    if found:
        print(f"{source}: recovered {len(found)} finished shards no manifest listed: {[e['shard'] for e in found]}")
    entries = merge_entries(entries, found)
    return _save_manifest(source, entries) if entries else []


def _record(source: str, jobs: list[tuple], results: list) -> None:
    """Save the jobs that finished to both manifests, then report the ones that failed after their retries."""
    done = [r for r in results if isinstance(r, dict)]
    _save_manifest(source, done)
    print(f"{len(done)} of {len(jobs)} done. manifests/{source}.json updated on the Mac and the Volume")
    failed = [(job, r) for job, r in zip(jobs, results) if not isinstance(r, dict)]
    for job, err in failed:
        print(f"failed {job}: {err!r}")
    if failed:
        raise SystemExit(f"{len(failed)} failed. Run the same command again to retry them")


@app.local_entrypoint()
def prep(source: str, n_files: int, val_docs: int = 2000, yes: bool = False):
    """Prep the next n_files unused parquet files of every part of a source. Updates manifests/<source>.json
    on the Mac and on the Volume, also for the files that finished when others fail."""
    from slm50m.dataprep.prep import shard_index

    done = _synced_manifest(source)
    used = {e["file"] for e in done}
    next_shard = 1 + max((shard_index(e["shard"]) for e in done), default=-1)
    has_val = any(e["role"] == "val+train" for e in done)
    user, part_files = hf_files.remote(source)
    print(f"HF token in the container: {user}")
    jobs = []
    for part_index, files in enumerate(part_files):
        for filename in [f for f in files if f not in used][:n_files]:
            first = not has_val and not jobs
            jobs.append((source, part_index, filename, next_shard, val_docs if first else 0))
            next_shard += 1
    if not jobs:
        raise SystemExit(f"every file of {source} is prepped already")
    # 20 min per file is the worst case. WikiHow took ~1 min (26 Oct 2026)
    confirm(f"prep {len(jobs)} files of {source}", len(jobs) * PREP_CONTAINER_HOUR / 3 * OVERHEAD + 0.01, yes)
    _record(source, jobs, list(prep_file.starmap(jobs, return_exceptions=True)))


@app.function(image=image, scaledown_window=IDLE, volumes={DATA: data_vol}, cpu=2, memory=4096, timeout=2 * 3600,
              max_containers=20)
def prep_arith_shard(shard_index: int, n_tokens: int, val_docs: int, source: str = "arith") -> dict:
    from slm50m.dataprep.prep import prep_arith
    from slm50m.tokenizer.train_tokenizer import load_tokenizer

    bench = _bench_filter() if source == "arith_v2" else None  # D-045: v2 runs the 13-gram filter too
    entry = prep_arith(DATA, load_tokenizer(Path(DATA) / "tokenizer"), shard_index, n_tokens, val_docs, source, bench)
    data_vol.commit()
    return entry


@app.local_entrypoint()
def prep_arith(n_shards: int, tokens_per_shard: int, val_docs: int = 2000, source: str = "arith", yes: bool = False):
    from slm50m.dataprep.prep import ARITH_GENERATORS, shard_index

    if source not in ARITH_GENERATORS:
        raise SystemExit(f"--source must be one of {list(ARITH_GENERATORS)}")
    done = _synced_manifest(source)
    start = 1 + max((shard_index(e["shard"]) for e in done), default=-1)  # also the seed of each shard
    has_val = any(e["role"] == "val+train" for e in done)
    jobs = [(start + i, tokens_per_shard, val_docs if i == 0 and not has_val else 0, source) for i in range(n_shards)]
    confirm(f"generate {n_shards} {source} shards", n_shards * PREP_CONTAINER_HOUR * OVERHEAD + 0.01, yes)
    _record(source, jobs, list(prep_arith_shard.starmap(jobs, return_exceptions=True)))


@app.local_entrypoint()
def sync_manifests():
    """Free, no container. Syncs every source's manifest between the Mac and the Volume, recovers finished shards
    neither lists, and rewrites each shard's entry file on the Volume from the manifest. Run it before training."""
    import io

    from slm50m.dataprep.sources import SOURCES

    for source in SOURCES:
        entries = _synced_manifest(source)
        if not entries:
            continue
        with data_vol.batch_upload(force=True) as batch:
            for e in entries:
                batch.put_file(io.BytesIO(json.dumps(e, indent=2).encode()),
                               f"/tokens/{source}/{Path(e['shard']).with_suffix('.json')}")
        print(f"{source}: {len(entries)} shards, {sum(e['tokens_train'] for e in entries):,} train tokens")


@app.function(image=image, scaledown_window=IDLE, volumes={DATA: data_vol}, cpu=2, memory=8192, timeout=2 * 3600,
              retries=2, max_containers=20)
def overlap_hashes(source: str, n_docs: int):
    from slm50m.dataprep.overlap import source_hashes
    from slm50m.tokenizer.train_tokenizer import load_tokenizer

    return source_hashes(DATA, source, load_tokenizer(Path(DATA) / "tokenizer"), n_docs)


@app.local_entrypoint()
def overlap(n_docs: int = 1_000_000, out: str = "reports/overlap.json", yes: bool = False):
    """Cross-source exact duplicates on the first n_docs train documents of each source (D-025). One container
    per source decodes ~2M tokens/s (Mac, 26 Oct 2026), so the biggest source takes ~15-30 min."""
    from slm50m.dataprep.overlap import overlap_report
    from slm50m.dataprep.sources import SOURCES

    sources = [s for s in SOURCES if not s.startswith("arith") and _read_volume_json(f"manifests/{s}.json")]  # arith is generated
    confirm(f"hash {n_docs:,} documents of {len(sources)} sources", len(sources) * PREP_CONTAINER_HOUR * OVERHEAD + 0.01,
            yes)
    # list() drains starmap's generator. zip() alone stops reading it early and Modal prints asyncgen errors
    hashes = dict(zip(sources, list(overlap_hashes.starmap([(s, n_docs) for s in sources]))))
    report = {"n_docs": n_docs, "date": time.strftime("%d %b %Y"), **overlap_report(hashes)}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2))
    for s, r in report["sources"].items():
        print(f"{s:15} {r['docs']:>10,} docs  {r['dup_within']:.2%} duplicated within")
    for p in sorted(report["pairs"], key=lambda p: -max(p["of_a"], p["of_b"]))[:10]:
        print(f"{p['a']:15} {p['b']:15} {p['shared']:>8,} shared  {p['of_a']:.3%} of {p['a']}  {p['of_b']:.3%} of {p['b']}")
    print(f"largest pair share {report['max_pair_share']:.3%} (act if above 5%). Written to {out}")


# ---------------------------------------------------------------- training (gates 2, 4, 5)

# The dashboard labels the function with this gpu, but run_train's with_options picks the real one per run
@app.function(image=image, scaledown_window=IDLE, volumes={DATA: data_vol, RUNS: runs_vol}, gpu="H100", timeout=3600,
              retries=modal.Retries(max_retries=2, initial_delay=10.0))
def train(config_json: str, deadline: float | None = None) -> dict:
    """Starts or resumes (D-007). Modal retries re-run it after a preemption, and it picks up the last checkpoint.
    Returns why it stopped and the ids of the milestone evals it spawned, which may still be running.

    deadline (a time.time() value) holds across attempts. Modal gives every retry, including one after a timeout,
    a fresh full timeout (modal.com/docs/guide/timeouts), so a run that hangs would otherwise cost 3x its hours.
    Past the deadline an attempt returns at once, and the trainer stops cleanly when it passes."""
    from slm50m.config import RunConfig
    from slm50m.training.trainer import Trainer

    if deadline is not None and time.time() >= deadline:
        return {"stop": "deadline", "evals": []}
    cfg = RunConfig.from_json(config_json)
    runs_vol.reload()
    evals: list[str] = []

    def on_save(path: Path, kind: str) -> None:
        runs_vol.commit()  # other containers (evals) can see it now
        if kind == "milestone":
            evals.append(evaluate.spawn(cfg.run_name, str(path.relative_to(cfg.run_dir))).object_id)

    return {"stop": Trainer(cfg, on_save).run(deadline=deadline), "evals": evals}


DEADLINE_GRACE_S = 600  # Modal's timeout sits this far past the deadline: it only fires if the trainer hangs


@app.local_entrypoint()
def run_train(config: str, gpu: str = "H100", hours: float = 0.5, yes: bool = False):
    """hours is a wall-clock limit from launch, over all attempts (preemptions, retries). The trainer stops
    cleanly at it. max_usd (step time only) usually stops first."""
    from slm50m.config import RunConfig

    cfg = RunConfig.from_json(Path(config).read_text())
    cfg.gpu_usd_per_hour = GPU_PRICES[gpu]
    # GPU time is bounded by the deadline plus one grace period, whatever happens inside
    cap = GPU_PRICES[gpu] * (hours + DEADLINE_GRACE_S / 3600) * OVERHEAD
    n_evals = int(cfg.total_steps * cfg.tokens_per_step // cfg.milestone_every_tokens)
    confirm(f"train {cfg.run_name} on {gpu}, stop at {hours} h wall clock (budget stop ${cfg.max_usd} of step time), "
            f"plus {n_evals} milestone evals on L4 (a few minutes, cents each, not in the total)", cap, yes)
    deadline = time.time() + hours * 3600
    result = train.with_options(gpu=gpu, timeout=int(hours * 3600) + DEADLINE_GRACE_S).remote(cfg.to_json(), deadline)
    print(f"training stopped: {result['stop']}")
    # the app, and every eval still running in it, stops when this entrypoint returns, so wait for them
    for call_id in result["evals"]:
        summary = modal.FunctionCall.from_id(call_id).get()
        print(f"{summary['checkpoint']}: II {summary['ii']:.2f}, {json.dumps(summary['scores'])}")


@app.function(image=image, scaledown_window=IDLE, volumes={DATA: data_vol, RUNS: runs_vol}, secrets=[hf_secret], gpu="L4",
              timeout=3 * 3600)
def evaluate(run_name: str, checkpoint: str, limit: int | None = None) -> dict:
    from slm50m.evaluation.evaluate import evaluate_checkpoint

    runs_vol.reload()
    run_dir = Path(RUNS) / run_name
    out = run_dir / "evals" / Path(checkpoint).stem
    summary = evaluate_checkpoint(run_dir / checkpoint, Path(DATA) / "tokenizer", out, "cuda", limit)
    runs_vol.commit()
    return summary


@app.local_entrypoint()
def run_eval(run: str, checkpoint: str, limit: int = 0, yes: bool = False):
    confirm(f"evaluate {run}/{checkpoint} on L4", GPU_PRICES["L4"] * 1.0 * OVERHEAD, yes)
    summary = evaluate.remote(run, checkpoint, limit or None)
    print(json.dumps({"scores": summary["scores"], "ii": summary["ii"]}, indent=2))


KERNEL_KINDS = [  # first match wins, on the lowercased kernel name
    ("attention", ("flash", "fmha", "attention", "sdpa")),
    ("matmul", ("gemm", "cutlass", "nvjet", "xmma", "triton_tem")),
    ("fused", ("triton_",)),  # torch.compile's elementwise and reduction kernels
    ("copy", ("memcpy", "memset")),
]


def profile_steps(trainer, n_steps: int, trace_path: str) -> dict:
    """n_steps more steps under torch.profiler. GPU kernel time per step by kind and by phase (the optimizers'
    step, grad clipping), the step's wall time, a top-30 kernel table and a gzipped Chrome trace."""
    import gzip

    import torch
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile, record_function

    clip = torch.nn.utils.clip_grad_norm_

    def labelled_clip(*args, **kwargs):
        with record_function("clip_grad_norm"):
            return clip(*args, **kwargs)

    torch.nn.utils.clip_grad_norm_ = labelled_clip  # the trainer looks it up on each call
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
    torch.cuda.synchronize()
    t0 = time.time()
    with profile(activities=activities) as prof:
        for _ in range(n_steps):
            trainer.train_step()
        torch.cuda.synchronize()
    wall_ms = (time.time() - t0) * 1000 / n_steps
    torch.nn.utils.clip_grad_norm_ = clip

    out = {"steps": n_steps, "step_ms": wall_ms}
    try:  # a parsing surprise must not lose the run: the table and trace still come back
        kinds, phases = {}, {}
        for e in prof.events():
            if e.device_type == DeviceType.CUDA and not e.is_user_annotation:
                name = e.name.lower()
                kind = next((k for k, keys in KERNEL_KINDS if any(s in name for s in keys)), "other")
                kinds[kind] = kinds.get(kind, 0.0) + e.time_range.elapsed_us() / 1000 / n_steps
            elif e.device_type == DeviceType.CPU and (e.name.startswith("Optimizer.step#") or e.name == "clip_grad_norm"):
                phases[e.name] = phases.get(e.name, 0.0) + e.device_time_total / 1000 / n_steps
        busy = sum(kinds.values())
        out.update(gpu_busy_ms=busy, gpu_idle_share=1 - busy / wall_ms,
                   kernels_ms=dict(sorted(kinds.items(), key=lambda kv: -kv[1])), phases_ms=phases)
    except Exception as err:  # noqa: BLE001
        out["summary_error"] = repr(err)
    sort_by = "self_device_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
    out["kernel_table"] = prof.key_averages().table(sort_by=sort_by, row_limit=30, max_name_column_width=80)
    prof.export_chrome_trace(trace_path)
    out["trace_gz"] = gzip.compress(Path(trace_path).read_bytes())
    return out


@app.function(image=image, scaledown_window=IDLE, volumes={DATA: data_vol}, gpu="H100", timeout=1800)
def benchmark(model: str, seq_len: int, batch_seqs: int, micro_batch_seqs: int, steps: int, tst: bool,
              data: bool, profile: bool = False) -> dict:
    """Gate 3: tokens/s, MFU, peak memory, compile time (Q-16, Q-17). Random tokens, or the shards on the data
    Volume with data=True. data_s is the time in loader.batch() per step: with accumulation the GPU runs the last
    micro-batch while the next one is read, so data only slows the step when data_s nears the step time.
    profile=True adds 3 profiled steps after the timed ones (profile_steps)."""
    import dataclasses

    import torch

    from slm50m.config import PRESETS, RunConfig
    from slm50m.training.trainer import Trainer

    preset = model
    if seq_len > PRESETS[model].max_seq_len:  # the 2,048 test (D-029): longer RoPE tables, same weights
        preset = f"{model}_{seq_len}"
        PRESETS[preset] = dataclasses.replace(PRESETS[model], max_seq_len=seq_len)
    cfg = RunConfig(run_name="bench", run_dir="/tmp/bench", data_dir=DATA if data else None, model=preset,
                    seq_len=seq_len, batch_seqs=batch_seqs, micro_batch_seqs=micro_batch_seqs, total_steps=10 * steps,
                    tst_end_frac=1.0 if tst else 0.0, eval_every=0, log_every=1)
    trainer = Trainer(cfg)
    fetch = [0.0]
    loader_batch = trainer.loader.batch

    def timed_batch(*args, **kwargs):
        t0 = time.time()
        out = loader_batch(*args, **kwargs)
        fetch[-1] += time.time() - t0
        return out

    trainer.loader.batch = timed_batch
    times = []
    for _ in range(steps):
        t0 = time.time()
        trainer.train_step()
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        fetch.append(0.0)

    def median(xs):  # after warmup
        return sorted(xs[steps // 3:])[len(xs[steps // 3:]) // 2]

    steady = median(times)
    tok_s = cfg.tokens_per_step / steady
    result = {"model": model, "seq_len": seq_len, "batch": batch_seqs, "micro_batch": micro_batch_seqs, "tst": tst,
              "data": data, "gpu": torch.cuda.get_device_name(), "first_step_s": times[0], "step_s": steady,
              "data_s": median(fetch[:steps]), "tokens_per_s": tok_s,
              "raw_tokens_per_s": tok_s * (cfg.tst_bag if tst else 1),
              "tflops": tok_s * PRESETS[model].flops_per_token(seq_len) / 1e12,
              "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
    if profile:
        result["profile"] = profile_steps(trainer, 3, str(Path(cfg.run_dir) / "trace.json"))
    return result


@app.local_entrypoint()
def run_benchmark(gpu: str = "H100", model: str = "main", seq_len: int = 1024, batch: int = 256,
                  micro_batch: int = 32, steps: int = 60, tst: bool = False, data: bool = False,
                  profile: bool = False, yes: bool = False):
    confirm(f"benchmark {model} on {gpu}, {steps} steps", GPU_PRICES[gpu] * 0.25 * OVERHEAD, yes)
    result = benchmark.with_options(gpu=gpu).remote(model, seq_len, batch, micro_batch, steps, tst, data, profile)
    if profile:  # the table and trace don't fit in the JSON print
        trace = BUILD_PROFILE / f"{gpu}_{model}_{seq_len}{'_tst' if tst else ''}.json.gz"
        trace.parent.mkdir(parents=True, exist_ok=True)
        trace.write_bytes(result["profile"].pop("trace_gz"))
        print(result["profile"].pop("kernel_table"))
        print(f"Chrome trace written to {trace} (open in ui.perfetto.dev)")
    print(json.dumps(result, indent=2))
