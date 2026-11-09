import json
import numpy as np
import pandas as pd
import pytest
from kscale import read_and_parse_data, build_KNN, k_scale
from kscale.attacks import simulate_attacks, CoverageIndex, single_point_compatible_values, plot_attack_heatmap


def release(tmp_path):
    rows = [dict(uid=i,x=i+t,y=i-t,timestamp=t*10,site_id=str(t)) for i in range(5) for t in range(4)]
    pd.DataFrame(rows).to_csv(tmp_path/'raw.csv',index=False)
    S = read_and_parse_data(tmp_path/'raw.csv',tmp_path/'Sigma',source_format='standard')
    K = build_KNN(S,tmp_path/'knn',K=4)
    G = k_scale(S,K,tmp_path/'release',k=3)
    return S,G


def test_nested_reproducible_end_to_end(tmp_path):
    S,G = release(tmp_path)
    options = dict(c_values=[1,2,3],beta_values=[1,2,4,'all'],trials=5,seed=7,progress=False)
    a = simulate_attacks(S,G,tmp_path/'a',**options)
    b = simulate_attacks(S,G,tmp_path/'b',**options)
    df = pd.read_csv(a/'attacks.csv')
    pd.testing.assert_frame_equal(df,pd.read_csv(b/'attacks.csv'))
    assert len(df) == 2*2*5*3*4
    assert (df.J >= df.gamma).all()
    for _,g in df.groupby(['victim_mode','point_mode','trial','c']):
        assert g.gamma.nunique() == 1
        assert np.all(np.diff(g.J) <= 0)
    for _,g in df.groupby(['victim_mode','point_mode','trial','beta']):
        assert np.all(np.diff(g.J) <= 0)
        assert np.all(np.diff(g.gamma) <= 0)
    # Independent brute-force containment across every public record.
    row = df.iloc[1]
    selected = json.loads(row.point_indices)
    raw = [pd.read_parquet(S/'Sigma'/f'S_{int(i):08d}.parquet')[['x','y','timestamp']].to_numpy()[ids]
           for i,ids in selected.items()]
    Q = np.concatenate(raw)
    count = 0
    from kscale.merge import BOUND_COLUMNS
    for h in range(5):
        boxes = pd.read_parquet(G/'Sigma_tilde'/f'G_{h:08d}.parquet')[BOUND_COLUMNS].to_numpy()
        count += all(any(np.all(p>=b[:3]) and np.all(p<=b[3:]) for b in boxes) for p in Q)
    assert row.J == count
    import matplotlib
    matplotlib.use('Agg')
    ax = plot_attack_heatmap(a/'attack_summary.csv')
    ax.figure.savefig(tmp_path/'heatmap.png')
    with pytest.raises(FileExistsError):
        simulate_attacks(S,G,a,**options)
    with pytest.raises(ValueError,match='enough points'):
        simulate_attacks(S,G,tmp_path/'bad',beta_values=[5],trials=1)
    meta = json.loads((G/'metadata.json').read_text())
    meta['dataset_id'] = 'mismatch'
    (G/'metadata.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='different'):
        simulate_attacks(S,G,tmp_path/'mismatch',**options)


def test_toy_nonreconstruction_and_identity_ambiguity():
    grid = np.arange(11)
    np.testing.assert_array_equal(single_point_compatible_values([0,10],1,[[0],[10]],grid).ravel(),grid)
    np.testing.assert_array_equal(single_point_compatible_values([0,4],1,[[0],[10]],grid).ravel(),[10])
    # Two unknowns: C can be interior while the other unknown supplies max.
    np.testing.assert_array_equal(single_point_compatible_values([0],2,[[0],[10]],grid).ravel(),grid)
    # Both extremes belong to unknowns: C is extreme but not identified.
    np.testing.assert_array_equal(single_point_compatible_values([5],2,[[0],[10]],grid).ravel(),[0,10])
    assert len(single_point_compatible_values([20],1,[[0],[10]],grid)) == 0
    with pytest.raises(ValueError,match='max_worlds'):
        single_point_compatible_values([],8,[[0],[10]],grid)


def test_accidental_coverage_and_boundaries(tmp_path):
    S,G = release(tmp_path)
    index = CoverageIndex(G,5,chunk_size=1)
    p = index.boxes[0,:3]
    expected = {int(h) for h,b in zip(index.owners,index.boxes) if np.all(p>=b[:3]) and np.all(p<=b[3:])}
    assert index.candidates([p]) == expected
    assert index.candidates([[1e9,1e9,1e9]]) == set()
