"""CM_i, RG_i, D(S_i,S_j), and the KNN structure (Section III-B/C)."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import warnings
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from .data import load_S_i
from ._io import new_stage, positive_integer, read_metadata, write_metadata


def CM(S_i):
    """Spatial center of mass, footnote 5: unweighted sample mean."""
    return S_i[["x", "y"]].to_numpy(dtype=float).mean(axis=0)


def RG(S_i):
    """Radius of gyration, not RG squared (footnote 5).

    Match the notebook's convention: each elapsed interval weights its later
    sample. For zero duration, use uniform sample weights instead of NaN.
    """
    S_i = S_i.sort_values("timestamp", kind="stable")
    r = S_i[["x", "y"]].to_numpy(dtype=float)
    squared_distance = ((r - CM(S_i)) ** 2).sum(axis=1)
    delta_t = np.diff(S_i.timestamp.to_numpy(dtype=float))
    if delta_t.sum() == 0:
        return float(np.sqrt(squared_distance.mean()))
    return float(np.sqrt(np.dot(squared_distance[1:], delta_t) / delta_t.sum()))


def D(features_i, features_j):
    """Proxy distance (Eqs. 6-7), using the repository's sum-area denominator.

    Each feature vector is (CM_x, CM_y, RG). It is not assumed to satisfy
    the triangle inequality. This distinction matters for BallTree pruning.
    """
    CM_i, CM_j = np.asarray(features_i[:2]), np.asarray(features_j[:2])
    RG_i, RG_j = features_i[2], features_j[2]
    overlap = np.maximum(0, np.minimum(CM_i + RG_i, CM_j + RG_j)
                         - np.maximum(CM_i - RG_i, CM_j - RG_j))
    intersection = float(np.prod(overlap))
    if intersection == 0:
        return float(np.abs(CM_i - CM_j).sum())
    c_ij = 2 * intersection / (4 * RG_i ** 2 + 4 * RG_j ** 2)
    return (1 - c_ij) / (1 + c_ij)


def _D_row(features_i, features):
    overlap = np.maximum(0, np.minimum(features_i[:2] + features_i[2],
                                      features[:, :2] + features[:, 2, None])
                         - np.maximum(features_i[:2] - features_i[2],
                                      features[:, :2] - features[:, 2, None]))
    intersection = overlap.prod(axis=1)
    distances = np.abs(features[:, :2] - features_i[:2]).sum(axis=1)
    mask = intersection > 0
    c_ij = 2 * intersection[mask] / (4 * features_i[2] ** 2 + 4 * features[mask, 2] ** 2)
    distances[mask] = (1 - c_ij) / (1 + c_ij)
    return distances


def build_KNN(Sigma, output_path, K=30, *, backend="exact", batch_size=1000, workers=1):
    """Save K neighbors OTHER than S_i per row, with distances and row IDs.

    exact: O(U^2) calculations split across workers using shared read-only features.
    Working space is O(workers * U), plus O(U * K) results.
    batch_size sets rows per task and Parquet shard. NumPy operations release
    the GIL; no process spawning or copying of the full features is needed.
    workers applies only to exact search; ball_tree requires workers=1.
    ball_tree: the paper's indexing approach; exactness is not guaranteed for D.
    Neither mode materializes a dense U-by-U distance matrix.
    """
    workers = positive_integer(workers, "workers")
    K = positive_integer(K, "K")
    batch_size = positive_integer(batch_size, "batch_size")
    metadata = read_metadata(Sigma)
    U = metadata["U"]
    if K >= U:
        raise ValueError(f"K must be less than U={U}; self-neighbors are excluded.")
    if backend not in {"exact", "ball_tree"}:
        raise ValueError("backend must be 'exact' or 'ball_tree'.")
    if backend == "ball_tree" and workers != 1:
        raise ValueError("Parallel workers are supported only by backend='exact'.")
    features = np.empty((U, 3))
    for i in range(U):
        S_i = load_S_i(Sigma, i)
        features[i] = [*CM(S_i), RG(S_i)]
    features.setflags(write=False)
    tree = None
    if backend == "ball_tree":
        from sklearn.neighbors import BallTree
        warnings.warn("D need not satisfy the triangle inequality; BallTree may miss neighbors.", stacklevel=2)
        tree = BallTree(features, metric=D)
    with new_stage(output_path) as output:
        (output / "neighbors").mkdir()
        pd.DataFrame(features, columns=["CM_x", "CM_y", "RG"]).to_parquet(output / "features.parquet", index=False)
        neighbor_indices = np.empty((U, K), dtype=np.int64)
        neighbor_distances = np.empty((U, K))
        def compute_block(start):
            stop = min(start + batch_size, U)
            block_indices = np.empty((stop - start, K), dtype=np.int64)
            block_distances = np.empty((stop - start, K))
            if tree is not None:
                tree_distances, tree_indices = tree.query(features[start:stop], k=K + 1)
            for i in range(start, stop):
                if tree is None:
                    distances = _D_row(features[i], features)
                    distances[i] = np.inf
                    # Resolve ties by trajectory index, including at the K boundary.
                    threshold = np.partition(distances, K - 1)[K - 1]
                    less = np.flatnonzero(distances < threshold)
                    equal = np.flatnonzero(distances == threshold)[:K - len(less)]
                    chosen = np.concatenate([less, equal])
                    chosen = chosen[np.lexsort((chosen, distances[chosen]))]
                    selected_distances = distances[chosen]
                else:
                    candidate = tree_indices[i - start]
                    mask = candidate != i
                    candidate, dist = candidate[mask], tree_distances[i - start][mask]
                    order = np.lexsort((candidate, dist))[:K]
                    chosen, selected_distances = candidate[order], dist[order]
                block_indices[i - start], block_distances[i - start] = chosen, selected_distances
            return start, stop, block_indices, block_distances

        def save_block(result):
            start, stop, block_indices, block_distances = result
            neighbor_indices[start:stop] = block_indices
            neighbor_distances[start:stop] = block_distances
            pd.DataFrame({"i": np.repeat(np.arange(start, stop), K),
                          "j": block_indices.ravel(),
                          "D_ij": block_distances.ravel()}).to_parquet(
                output / "neighbors" / f"part_{start:08d}.parquet", index=False)

        effective_workers = min(workers, (U + batch_size - 1) // batch_size)
        if effective_workers == 1:
            for start in range(0, U, batch_size):
                save_block(compute_block(start))
        else:
            # Bound in-flight results; never enqueue the entire dataset.
            with ThreadPoolExecutor(max_workers=effective_workers) as pool:
                for wave in range(0, U, batch_size * effective_workers):
                    starts = range(wave, min(wave + batch_size * effective_workers, U), batch_size)
                    for result in pool.map(compute_block, starts):
                        save_block(result)
        # Connectivity is separate: a zero distance is still a valid neighbor.
        indices = neighbor_indices.ravel()
        indptr = np.arange(0, U * K + 1, K)
        save_npz(output / "KNN.npz", csr_matrix((np.ones(U * K, dtype=bool), indices, indptr), shape=(U, U)))
        save_npz(output / "D_KNN.npz", csr_matrix((neighbor_distances.ravel(), indices, indptr), shape=(U, U)))
        write_metadata(output, stage="KNN", dataset_id=metadata["dataset_id"], U=U,
                       K=K, backend=backend, include_self=False,
                       workers=workers, effective_workers=effective_workers, batch_size=batch_size,
                       proxy_denominator="sum_of_areas", zero_duration_RG="uniform_weights")
    return Path(output_path).resolve()