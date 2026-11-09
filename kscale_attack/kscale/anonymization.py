"""Pairwise C_tilde, nu-k-pick A, and Sigma_tilde (Sections III-C/E)."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz
from ._io import new_stage, positive_integer, read_metadata, write_metadata, require_dataset
from .data import load_S_i
from .merge import gen, BOUND_COLUMNS


def _points(Sigma, i):
    return load_S_i(Sigma, i)[["x", "y", "timestamp"]].to_numpy(dtype=float)


def compute_C_tilde(Sigma, KNN_path, output_path, *, workers=1,
                    reciprocal_candidates=False, batch_size=100):
    """Save directional costs C_tilde_Sigma[i,j] = C_i(gen(S_i,S_j)).

    Missing sparse entries mean unavailable/+infinity, NOT zero cost.
    reciprocal_candidates=True explicitly expands KNN to its undirected union;
    this optional repair is not Algorithm 1's original directed candidate graph.
    Each unordered pair is computed once; both costs retain their own N_i.
    """
    workers = positive_integer(workers, "workers")
    batch_size = positive_integer(batch_size, "batch_size")
    metadata = read_metadata(KNN_path)
    require_dataset(Sigma, metadata)
    KNN = load_npz(Path(KNN_path) / "KNN.npz").tocsr().astype(bool)
    if reciprocal_candidates:
        KNN = KNN.maximum(KNN.T).tocsr()
    KNN.sort_indices()
    U = KNN.shape[0]
    rows = np.repeat(np.arange(U), np.diff(KNN.indptr))
    cols = KNN.indices
    # Encode unordered pairs; inverse maps costs back to the directed graph.
    keys, inverse = np.unique(np.minimum(rows, cols) * U + np.maximum(rows, cols), return_inverse=True)
    pair_costs = np.empty((len(keys), 2))

    def pair_cost(key):
        i, j = divmod(int(key), U)
        return gen(_points(Sigma, i), _points(Sigma, j)).C

    with new_stage(output_path) as output:
        (output / "C_tilde").mkdir()
        # Numba releases the GIL; threads also work from Jupyter without fork.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0, len(keys), batch_size):
                stop = min(start + batch_size, len(keys))
                pair_costs[start:stop] = list(pool.map(pair_cost, keys[start:stop]))
        costs = pair_costs[inverse, (rows > cols).astype(int)]
        if not np.isfinite(costs).all():
            raise ValueError("Non-finite generalization costs; check input scales.")
        C_tilde_Sigma = csr_matrix((costs, cols.copy(), KNN.indptr.copy()), shape=(U, U))
        save_npz(output / "C_tilde_Sigma.npz", C_tilde_Sigma)
        for start in range(0, U, batch_size):
            stop = min(start + batch_size, U)
            a, b = KNN.indptr[start], KNN.indptr[stop]
            pd.DataFrame({"i": rows[a:b], "j": cols[a:b], "C_ij": costs[a:b]}).to_parquet(
                output / "C_tilde" / f"part_{start:08d}.parquet", index=False)
        write_metadata(output, stage="C_tilde", dataset_id=metadata["dataset_id"], U=U,
                       K=metadata["K"], reciprocal_candidates=reciprocal_candidates,
                       knn_backend=metadata["backend"], missing_cost="infinity",
                       cost_unit="m*s", objective="sum_of_per_trajectory_mean_costs")
    return Path(output_path).resolve()


def nu_k_pick(C_tilde_path, output_path, k=10):
    """Algorithm 1: pick k-1 finite costs per column, then repair empty rows."""
    k = positive_integer(k, "k")
    metadata = read_metadata(C_tilde_path)
    U = metadata["U"]
    if not 2 <= k <= U:
        raise ValueError(f"Require 2 <= k <= U={U}.")
    C_tilde_Sigma = load_npz(Path(C_tilde_path) / "C_tilde_Sigma.npz").tocsr()
    columns = C_tilde_Sigma.tocsc()
    indegree = np.diff(columns.indptr)
    if np.any(indegree < k - 1):
        missing = np.flatnonzero(indegree < k - 1)
        raise ValueError(f"{len(missing)} trajectories have fewer than k-1 finite incoming costs "
                         f"(first IDs: {missing[:10].tolist()}). Increase K or explicitly "
                         "recompute costs with reciprocal_candidates=True and K >= k-1.")
    selected_rows, selected_cols = [], []
    for j in range(U):
        a, b = columns.indptr[j:j + 2]
        candidates, costs = columns.indices[a:b], columns.data[a:b]
        # Stored zeros are real costs and must participate in the selection.
        chosen = candidates[np.lexsort((candidates, costs))[:k - 1]]
        selected_rows.extend(chosen.tolist())
        selected_cols.extend([j] * len(chosen))
    A = csr_matrix((np.ones(len(selected_rows), dtype=bool), (selected_rows, selected_cols)), shape=(U, U))
    A = A.tolil()
    fallback_rows = []
    for i in range(U):
        if not A.rows[i]:
            a, b = C_tilde_Sigma.indptr[i:i + 2]
            candidates, costs = C_tilde_Sigma.indices[a:b], C_tilde_Sigma.data[a:b]
            if not len(candidates):
                raise ValueError(f"No finite fallback neighbor for S_{i}.")
            j = candidates[np.lexsort((candidates, costs))[0]]
            A[i, j] = True
            fallback_rows.append(i)
    A = A.tocsr()
    coverage = np.asarray(A.sum(axis=0)).ravel() + 1
    if np.any(coverage < k):
        raise RuntimeError("nu-k-pick coverage condition failed.")
    with new_stage(output_path) as output:
        save_npz(output / "A.npz", A)
        i, j = A.nonzero()
        pd.DataFrame({"i": i, "j": j}).to_parquet(output / "S_sets.parquet", index=False)
        pd.DataFrame({"i": np.arange(U), "generalization_count": coverage}).to_parquet(output / "coverage.parquet", index=False)
        write_metadata(output, stage="A", dataset_id=metadata["dataset_id"], U=U, k=k,
                       K=metadata["K"], reciprocal_candidates=metadata["reciprocal_candidates"],
                       fallback_rows=fallback_rows)
    return Path(output_path).resolve()


def generalize_Sigma(Sigma, A_path, output_path, *, workers=1, batch_size=100):
    """Write G_i = gen(S_i, S(i)) separately; never merge in arbitrary groups of 10."""
    workers = positive_integer(workers, "workers")
    batch_size = positive_integer(batch_size, "batch_size")
    metadata = read_metadata(A_path)
    require_dataset(Sigma, metadata)
    A = load_npz(Path(A_path) / "A.npz").tocsr()
    U = metadata["U"]
    with new_stage(output_path) as output:
        (output / "Sigma_tilde").mkdir()
        save_npz(output / "A.npz", A)

        def generalize_i(i):
            S_indices = sorted([i, *A.indices[A.indptr[i]:A.indptr[i + 1]].tolist()])
            owner = S_indices.index(i)
            result = gen(*[_points(Sigma, j) for j in S_indices])
            G_i = pd.DataFrame(result.G, columns=BOUND_COLUMNS)
            G_i["N_i_segment"] = result.sample_counts[:, owner]
            relative = f"Sigma_tilde/G_{i:08d}.parquet"
            G_i.to_parquet(output / relative, index=False)
            return {"i": i, "path": relative, "M_i": len(G_i),
                    "L": len(S_indices), "C_G_i": result.C[owner]}

        manifest = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0, U, batch_size):
                manifest.extend(pool.map(generalize_i, range(start, min(start + batch_size, U))))
        pd.DataFrame(manifest).to_parquet(output / "manifest.parquet", index=False)
        write_metadata(output, stage="Sigma_tilde", dataset_id=metadata["dataset_id"], U=U,
                       k=metadata["k"], K=metadata["K"],
                       reciprocal_candidates=metadata["reciprocal_candidates"],
                       gen_implementation="quadratic_contiguous_partition_DP",
                       coordinate_unit="m", time_unit="s")
    return Path(output_path).resolve()


def k_scale(Sigma, KNN_path, output_path, k=10, *, workers=1, reciprocal_candidates=False):
    """Run cost calculation, nu-k-pick and gen from an existing KNN stage."""
    with new_stage(output_path) as output:
        C_tilde = compute_C_tilde(Sigma, KNN_path, output / "costs", workers=workers,
                                 reciprocal_candidates=reciprocal_candidates)
        A = nu_k_pick(C_tilde, output / "sets", k=k)
        generalize_Sigma(Sigma, A, output / "generalized", workers=workers)
        write_metadata(output, stage="k_scale", dataset_id=read_metadata(Sigma)["dataset_id"], k=k)
    return Path(output_path).resolve() / "generalized"
