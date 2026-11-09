import numpy as np
import pandas as pd
import pytest
from kscale.metrics import repository_error_summary, empirical_cdf, repository_histogram_cdf
from kscale import read_and_parse_data, load_S_i


def test_error_definitions_and_independent_trimming():
    r = repository_error_summary([0, 1, 4, 100], [20, 0, 5, 10])
    assert r['trimmed_sample_count'] == 3
    assert r['mean_spatial_error_km'] == pytest.approx(np.sqrt(5/3))
    assert r['mean_time_error_min'] == 5
    assert r['untrimmed_spatial_error_km'] == pytest.approx(np.sqrt(105/4))
    assert repository_error_summary([4], [10])['mean_spatial_error_km'] is None


def test_ecdf_includes_zeros_and_ties():
    c = empirical_cdf([0, 0, .1, 1, 1, 4])
    assert c.spatial_accuracy_km2.tolist() == [0, .1, 1, 4]
    np.testing.assert_allclose(c.cdf, [2/6, 3/6, 5/6, 1])


def test_repository_histogram_matches_original_when_bins_valid():
    data = np.array([0, .05, .1, .2, .5, 1, 2, 10, 500, 700.])
    bins = [.1, .2, .3, .5, 1, 2, 4, 7, 10, 25, 50, 100, 500, max(data)]
    counts, _ = np.histogram(data, bins=bins)
    expected = np.cumsum(counts / counts.sum())[:-1]
    result = repository_histogram_cdf(data)
    np.testing.assert_allclose(result.cdf_repository_histogram, expected)
    assert result.excluded_below_0_1.iloc[0] == 2
    assert repository_histogram_cdf([0, 0]).cdf_repository_histogram.isna().all()


def test_sampling_preserves_users_and_is_reproducible(tmp_path):
    raw = pd.DataFrame([{'uid': i, 'x': i, 'y': t, 'timestamp': t, 'site_id':str(t)}
                        for i in range(20) for t in [0, 60, 120]])
    source = tmp_path / 'raw.csv';raw.to_csv(source,index=False)
    paths = [read_and_parse_data(source,tmp_path/f'data_{r}', source_format='standard',
                                sample_users=6,random_state=42) for r in range(2)]
    manifests = [pd.read_parquet(p/'manifest.parquet') for p in paths]
    pd.testing.assert_frame_equal(*manifests)
    assert manifests[0].N_i.tolist() == [3]*6
    assert manifests[0].i.tolist() == list(range(6))
    for i in range(6):assert load_S_i(paths[0],i).timestamp.tolist()==[0,60,120]
    short = read_and_parse_data(source,tmp_path/'short',source_format='standard',
                               sample_users=6,random_state=42,time_window_seconds=60)
    assert pd.read_parquet(short/'manifest.parquet').N_i.tolist()==[2]*6
    with pytest.raises(ValueError):
        read_and_parse_data(source,tmp_path/'bad',source_format='standard',sample_users=21)


def test_saved_metrics_match_repository_formulas(tmp_path):
    from kscale import build_KNN, k_scale, calculate_metrics
    raw = pd.DataFrame([{'uid':i,'x':i*1000+t*100,'y':i*500,
                         'timestamp':t*300+i*10,'site_id':str(i)}
                        for i in range(4) for t in range(3)])
    source=tmp_path/'input.csv';raw.to_csv(source,index=False)
    Sigma=read_and_parse_data(source,tmp_path/'Sigma',source_format='standard')
    KNN=build_KNN(Sigma,tmp_path/'knn',K=3)
    G=k_scale(Sigma,KNN,tmp_path/'run',k=2)
    result=calculate_metrics(Sigma,G,tmp_path/'metrics',spatial_thresholds_km2=[0,.5,1,10])
    frames=[pd.read_parquet(f) for f in sorted((G/'Sigma_tilde').glob('*.parquet'))]
    boxes=pd.concat(frames,ignore_index=True)
    area=((boxes.x_max-boxes.x_min)*(boxes.y_max-boxes.y_min)/1e6).to_numpy()
    duration=((boxes.timestamp_max-boxes.timestamp_min)/60).to_numpy()
    m=int(.95*len(area))
    assert result['mean_spatial_error_km']==pytest.approx(np.sqrt(np.mean(sorted(area)[:m])))
    assert result['mean_time_error_min']==pytest.approx(np.mean(sorted(duration)[:m]))
    cdf=pd.read_csv(tmp_path/'metrics/spatial_accuracy_thresholds.csv')
    np.testing.assert_allclose(cdf.cdf,[(area<=a).mean() for a in cdf.spatial_accuracy_km2])
