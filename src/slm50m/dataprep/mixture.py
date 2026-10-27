"""Stratified mixture sampler (D-007, D-008).

Every micro-batch draws sequences from every source in the target ratio. The batch is a pure
function of (seed, step, micro-step, rank), so resume needs no loader state and repeats exactly.
"""

from pathlib import Path

import numpy as np
import torch


def stratified_counts(weights: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Split n sequences across sources so each gets floor or ceil of weight x n, in expectation exactly weight x n.

    Each source first gets floor(w x n). The leftover seats go out by systematic sampling over the
    fractional parts: one random offset u, then seats at u, u+1, u+2, ... along the cumulative fractions.
    Source i then wins an extra seat with probability exactly frac_i.
    """
    weights = np.asarray(weights, dtype=np.float64)
    quotas = weights / weights.sum() * n
    counts = np.floor(quotas).astype(np.int64)
    leftover = n - counts.sum()
    if leftover > 0:
        frac = quotas - counts
        cum = np.cumsum(frac) * (leftover / frac.sum())  # rescale so the last edge is exactly `leftover`
        edges = np.concatenate([[0.0], cum])
        u = rng.random()
        # number of seat positions u + k (k = 0, 1, ...) that land in [edges[i], edges[i+1])
        counts += (np.ceil(edges[1:] - u) - np.ceil(edges[:-1] - u)).astype(np.int64)
    assert counts.sum() == n
    return counts


class SourceShards:
    """All train shards of one source, memory-mapped, treated as one pool of windows."""

    def __init__(self, source_dir: Path):
        self.source_dir = source_dir
        self.paths = sorted(source_dir.glob("train_*.bin"))
        assert self.paths, f"no shards in {source_dir}"
        self.shards = [np.memmap(p, dtype=np.uint16, mode="r") for p in self.paths]
        self._starts: dict[int, np.ndarray] = {}

    def sample(self, rng: np.random.Generator, length: int) -> np.ndarray:
        """One random window of `length` tokens. Every start position in every shard is equally likely."""
        if length not in self._starts:
            self._starts[length] = np.array([max(len(s) - length + 1, 0) for s in self.shards], dtype=np.int64)
        starts = self._starts[length]
        assert starts.sum() > 0, f"{self.source_dir} has no window of {length} tokens"
        k = rng.choice(len(starts), p=starts / starts.sum())
        i = rng.integers(starts[k])
        return self.shards[k][i:i + length]


class MixtureLoader:
    def __init__(self, data_dir: str | Path, sources: list[str], seed: int):
        self.data_dir = Path(data_dir)
        self.seed = seed
        self.shards = {s: SourceShards(self.data_dir / "tokens" / s) for s in sources}

    def batch(self, mix: dict[str, float], step: int, micro: int, n_seqs: int, length: int,
              rank: int = 0) -> tuple[torch.Tensor, dict[str, int]]:
        """(n_seqs, length) int64 token windows and the number drawn from each source."""
        rng = np.random.default_rng([self.seed, step, micro, rank])
        names = [s for s, w in mix.items() if w > 0]
        counts = stratified_counts(np.array([mix[s] for s in names]), n_seqs, rng)
        rows = [self.shards[name].sample(rng, length) for name, c in zip(names, counts) for _ in range(c)]
        windows = torch.from_numpy(np.stack(rows).astype(np.int64))
        return windows, dict(zip(names, counts.tolist()))

    def val_windows(self, source: str, n_windows: int, length: int) -> torch.Tensor:
        """The first n_windows back-to-back windows of a source's val.bin. Fixed, so val loss is comparable."""
        val = np.memmap(self.data_dir / "tokens" / source / "val.bin", dtype=np.uint16, mode="r")
        n = min(n_windows, len(val) // length)
        return torch.from_numpy(val[:n * length].astype(np.int64).reshape(n, length))


class RandomTokens:
    """Same interface as MixtureLoader, random tokens. For the gate 3 throughput benchmark (no data needed)."""

    def __init__(self, vocab_size: int, seed: int):
        self.vocab_size, self.seed = vocab_size, seed

    def batch(self, mix, step, micro, n_seqs, length, rank=0):
        g = torch.Generator().manual_seed(hash((self.seed, step, micro, rank)) % 2 ** 63)
        return torch.randint(0, self.vocab_size, (n_seqs, length), generator=g), {"random": n_seqs}

    def val_windows(self, source, n_windows, length):
        return torch.empty(0, length, dtype=torch.long)
