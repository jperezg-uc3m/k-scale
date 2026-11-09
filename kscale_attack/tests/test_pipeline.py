import itertools
import json
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import load_npz
from kscale import (read_and_parse_data, load_S_i, build_KNN, k_scale,
                    calculate_metrics, gen, CM, RG, D, compute_C_tilde, nu_k_pick)
from kscale.neighbors import _D_row


def dataset(tmp_path, U=5, identical=False):
    rows = []
    for i in range(U):
        for t in range(3):
            rows.append({"uid": f"user_{i}", "x": 0 if identical else i + t,
                         "y": 0 if identical else i - t, "timestamp": t * 10 + (0 if identical else i),
                         "site_id": f"site_{t}"})
    path = tmp_path / 'raw.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    return read_and_parse_data(path, tmp_path / 'data', source_format='standard')


def test_pipeline_and_zero_costs(tmp_path):
    Sigma = dataset(tmp_path, identical=True)
    KNN = build_KNN(Sigma, tmp_path / 'knn', K=4, batch_size=2)
    G = k_scale(Sigma, KNN, tmp_path / 'run', k=3, workers=2)
    metrics = calculate_metrics(Sigma, G, tmp_path / 'metrics')
    assert metrics['minimum_generalization_count'] >= 3
    assert metrics['mean_C_G_i_m_s'] == 0
    assert metrics['assigned_sample_containment_verified']
    assert metrics['original_samples'] == 15
    C = load_npz(tmp_path / 'run/costs/C_tilde_Sigma.npz')
    assert C.nnz == 20 and np.all(C.data == 0)
    assert list(load_S_i(Sigma, 0).columns) == ['x', 'y', 'timestamp', 'site_id']
    assert np.all(np.diff(load_npz(KNN / 'KNN.npz').indptr) == 4)
    with pytest.raises(FileExistsError):
        build_KNN(Sigma, KNN, K=4)


def test_nonzero_pipeline(tmp_path):
    Sigma = dataset(tmp_path)
    KNN = build_KNN(Sigma, tmp_path / 'knn', K=2)
    G = k_scale(Sigma, KNN, tmp_path / 'run', k=3, workers=2, reciprocal_candidates=True)
    metrics = calculate_metrics(Sigma, G, tmp_path / 'metrics')
    assert metrics['mean_C_G_i_m_s'] > 0
    assert metrics['nu_k_pick_satisfied']


def test_yjmob(tmp_path):
    path = tmp_path / 'raw.parquet'
    pd.DataFrame({'uid': [4, 4], 'd': [1, 1], 't': [2, 0], 'x': [2, 1], 'y': [3, 1]}).to_parquet(path)
    Sigma = read_and_parse_data(path, tmp_path / 'data')
    S = load_S_i(Sigma, 0)
    assert S.timestamp.tolist() == [86400, 90000]
    assert S.x.tolist() == [250, 750]
    assert S.site_id.tolist() == ['1:1', '2:3']


def test_RG_and_proxy():
    S = pd.DataFrame({'x': [0., 2.], 'y': [0., 2.], 'timestamp': [0., 10.]})
    np.testing.assert_allclose(CM(S), [1, 1])
    assert RG(S) == pytest.approx(np.sqrt(2))
    S.timestamp = 0
    assert RG(S) == pytest.approx(np.sqrt(2))
    features = np.array([[0, 0, 0], [0, 0, 1], [4, 4, 2], [1, 1, 2]], dtype=float)
    for f in features:
        np.testing.assert_allclose(_D_row(f, features), [D(f, g) for g in features])


def test_cost_counts_each_sample():
    result = gen(np.array([[1, 0, 1], [2, 0, 2]]), np.array([[3, 0, 3]]))
    np.testing.assert_allclose(result.C, [4, 4])
    np.testing.assert_array_equal(result.sample_counts, [[2, 1]])


def brute_force_cost(S):
    N = [len(s) for s in S]
    samples = np.concatenate(S)
    labels = np.repeat(np.arange(len(S)), N)
    order = np.argsort(samples[:, 2], kind='stable')
    samples, labels = samples[order], labels[order]
    best = np.inf
    for mask in itertools.product([False, True], repeat=len(samples)-1):
        cuts = [0, *[j+1 for j, chosen in enumerate(mask) if chosen], len(samples)]
        cost = 0
        for start, end in zip(cuts[:-1], cuts[1:]):
            counts = np.bincount(labels[start:end], minlength=len(S))
            if np.any(counts == 0):
                break
            delta = np.ptp(samples[start:end], axis=0)
            cost += delta[2] * (delta[0] + delta[1]) * (counts / N).sum()
        else:
            best = min(best, cost)
    return best


def test_DP_against_exhaustive_partitions():
    rng = np.random.default_rng(12)
    for _ in range(30):
        S = [rng.integers(-2, 5, size=(rng.integers(1, 4), 3)).astype(float) for _ in range(2)]
        result = gen(*S)
        assert result.C.sum() == pytest.approx(brute_force_cost(S))
        assert np.all(result.sample_counts > 0)
        assert np.all(result.G[:-1, 5] <= result.G[1:, 2])


def test_insufficient_incoming_costs_and_repair(tmp_path):
    Sigma = dataset(tmp_path, U=5, identical=True)
    KNN = build_KNN(Sigma, tmp_path / 'knn', K=2)
    costs = compute_C_tilde(Sigma, KNN, tmp_path / 'costs')
    with pytest.raises(ValueError, match='incoming costs'):
        nu_k_pick(costs, tmp_path / 'sets', k=3)
    assert not (tmp_path / 'sets').exists()
    costs = compute_C_tilde(Sigma, KNN, tmp_path / 'repaired', reciprocal_candidates=True)
    A = nu_k_pick(costs, tmp_path / 'sets', k=3)
    assert (A / 'A.npz').exists()


def test_mismatched_dataset(tmp_path):
    Sigma = dataset(tmp_path)
    KNN = build_KNN(Sigma, tmp_path / 'knn', K=2)
    metadata = json.loads((Sigma / 'metadata.json').read_text())
    metadata['dataset_id'] = 'another_dataset'
    (Sigma / 'metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='different'):
        compute_C_tilde(Sigma, KNN, tmp_path / 'costs')


def test_iso_timestamps_and_time_window(tmp_path):
    path = tmp_path / 'raw.csv'
    pd.DataFrame({'uid': [1, 1], 'x': [0, 1], 'y': [0, 2],
                  'timestamp': ['1970-01-01T00:00:00Z', '1970-01-01T00:01:00Z'],
                  'site_id': ['a', 'b']}).to_csv(path, index=False)
    Sigma = read_and_parse_data(path, tmp_path / 'data', source_format='standard')
    assert load_S_i(Sigma, 0).timestamp.tolist() == [0, 60]
    Sigma = read_and_parse_data(path, tmp_path / 'short', source_format='standard', time_window_seconds=30)
    assert len(load_S_i(Sigma, 0)) == 1


def test_ball_tree_full_neighbor_set(tmp_path):
    Sigma = dataset(tmp_path)
    with pytest.warns(UserWarning, match='triangle'):
        KNN = build_KNN(Sigma, tmp_path / 'tree', K=4, backend='ball_tree', batch_size=2)
    matrix = load_npz(KNN / 'KNN.npz')
    assert matrix.nnz == 20
    assert not matrix.diagonal().any()


@pytest.mark.parametrize('identical', [False, True])
@pytest.mark.parametrize('K', [1, 4, 8])
def test_parallel_exact_matches_sequential_and_scalar(tmp_path, identical, K):
    Sigma = dataset(tmp_path, U=9, identical=identical)
    sequential = build_KNN(Sigma, tmp_path / 'serial', K=K, batch_size=2)
    parallel = build_KNN(Sigma, tmp_path / 'parallel', K=K, batch_size=2, workers=4)
    for name in ['KNN.npz', 'D_KNN.npz']:
        expected, actual = load_npz(sequential / name), load_npz(parallel / name)
        np.testing.assert_array_equal(actual.indptr, expected.indptr)
        np.testing.assert_array_equal(actual.indices, expected.indices)
        np.testing.assert_array_equal(actual.data, expected.data)
    features = pd.read_parquet(parallel / 'features.parquet').to_numpy()
    for shard in sorted((parallel / 'neighbors').glob('*.parquet')):
        pd.testing.assert_frame_equal(pd.read_parquet(shard), pd.read_parquet(sequential / 'neighbors' / shard.name))
        for i, row in pd.read_parquet(shard).groupby('i', sort=False):
            distances = np.array([D(features[i], f) for f in features])
            distances[i] = np.inf
            expected = np.lexsort((np.arange(9), distances))[:K]
            np.testing.assert_array_equal(row.j, expected)
            np.testing.assert_allclose(row.D_ij, distances[expected], rtol=1e-14)
    assert json.loads((parallel / 'metadata.json').read_text())['effective_workers'] == 4


@pytest.mark.parametrize('workers', [0, -1, True, 1.5])
def test_invalid_knn_workers(tmp_path, workers):
    with pytest.raises(ValueError, match='workers'):
        build_KNN(tmp_path / 'unused', tmp_path / 'knn', workers=workers)


def test_worker_count_capped_by_blocks(tmp_path):
    Sigma = dataset(tmp_path)
    KNN = build_KNN(Sigma, tmp_path / 'knn', K=2, workers=8, batch_size=100)
    assert json.loads((KNN / 'metadata.json').read_text())['effective_workers'] == 1
