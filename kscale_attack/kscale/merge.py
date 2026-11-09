"""gen(S_1, ..., S_L) and the Eq. (4) generalization cost."""
from dataclasses import dataclass
import numpy as np
from numba import njit

BOUND_COLUMNS = ["x_min", "y_min", "timestamp_min", "x_max", "y_max", "timestamp_max"]


@dataclass
class Generalization:
    G: np.ndarray
    C: np.ndarray
    sample_counts: np.ndarray


@njit(cache=True, nogil=True)
def _minimum_cost_partition(samples, labels, N, L):
    """Quadratic reference DP over contiguous time-sorted partitions."""
    n = len(samples)
    best = np.full(n + 1, np.inf)
    previous = np.full(n + 1, -1, dtype=np.int64)
    seen_at = np.full(L, -1, dtype=np.int64)
    best[0] = 0.0
    for end in range(1, n + 1):
        xmin = ymin = tmin = np.inf
        xmax = ymax = tmax = -np.inf
        distinct = 0
        weight = 0.0
        for start in range(end - 1, -1, -1):
            xmin = min(xmin, samples[start, 0])
            ymin = min(ymin, samples[start, 1])
            tmin = min(tmin, samples[start, 2])
            xmax = max(xmax, samples[start, 0])
            ymax = max(ymax, samples[start, 1])
            tmax = max(tmax, samples[start, 2])
            i = labels[start]
            weight += 1.0 / N[i]
            if seen_at[i] != end:
                seen_at[i] = end
                distinct += 1
            if distinct != L or not np.isfinite(best[start]):
                continue
            # Eq. (4): each original sample contributes, including merged ones.
            candidate = best[start] + (tmax - tmin) * (xmax - xmin + ymax - ymin) * weight
            if candidate < best[end]:
                best[end] = candidate
                previous[end] = start
    return previous


def k_merge(S):
    """Return G and per-input C(G_i) for L raw point trajectories.

    S is an ordered sequence of nonempty (N_i, 3) arrays (x, y, timestamp).
    Minimize sum_i C(G_i) over contiguous complete partitions after stable
    temporal sorting. Each segment contains every input trajectory. Adjacent
    intervals may share an endpoint. This implementation is O((sum N_i)^2),
    not the linear operator claimed by the paper; equivalence is not asserted.
    """
    S = [np.asarray(S_i, dtype=float) for S_i in S]
    L = len(S)
    if L == 0 or any(S_i.ndim != 2 or S_i.shape[1] != 3 or len(S_i) == 0 for S_i in S):
        raise ValueError("S must contain nonempty arrays with columns x,y,timestamp.")
    if any(not np.isfinite(S_i).all() for S_i in S):
        raise ValueError("S contains non-finite samples.")
    N = np.array([len(S_i) for S_i in S])
    samples = np.concatenate(S)
    labels = np.repeat(np.arange(L), N)
    order = np.argsort(samples[:, 2], kind="stable")
    samples, labels = samples[order], labels[order]
    previous = _minimum_cost_partition(samples, labels, N, L)
    if previous[-1] < 0:
        raise RuntimeError("No valid complete partition.")
    cuts, end = [], len(samples)
    while end:
        start = previous[end]
        cuts.append((start, end))
        end = start
    cuts.reverse()
    G = np.empty((len(cuts), 6))
    sample_counts = np.zeros((len(cuts), L), dtype=np.int64)
    for m, (start, end) in enumerate(cuts):
        block = samples[start:end]
        G[m] = np.concatenate([block.min(axis=0), block.max(axis=0)])
        sample_counts[m] = np.bincount(labels[start:end], minlength=L)
    delta = G[:, 3:] - G[:, :3]
    expansion = delta[:, 2] * (delta[:, 0] + delta[:, 1])
    C = (expansion[:, None] * sample_counts).sum(axis=0) / N
    return Generalization(G, C, sample_counts)


def gen(*S):
    """Paper notation: G = gen(S_1, ..., S_L), Eq. (1)."""
    return k_merge(S)
