# k-scale metric update

This package uses the six files supplied by the user as the baseline. Changes
are limited to data.py (optional whole-user sampling), metrics.py (evaluation),
the notebook, exports and dependencies. merge.py, neighbors.py and
anonymization.py are retained unchanged.

## Run

Extract, open the notebook in this directory, install with `%pip install -e .`,
and restart the kernel. Input path is preserved from the supplied notebook.
Use a new output directory. Dependencies include Matplotlib for plotting.

The notebook keeps the supplied one-hour window explicitly as `T_max=3600`.
It adds `sample_users=None` and `random_state=42`. Set sample_users to an integer
to select whole users uniformly without replacement, retaining every observation
within their time window. A fixed seed is reproducible for a fixed input row order.
The manifest retains selected source IDs and contiguous internal indices.
No random sample operation was present in the supplied files; only a time filter.
For full-duration data use `T_max=None`. Select the users ONCE before comparing k.

## Exact definitions

Reference: https://github.com/nds-group/k-scale/blob/7df19be4891c71af55a0f16a99153ed178a2b306/Evaluation/eval_k-scale.ipynb
Methods: spatialAccuracy, temporalAccuracy, getPosError, getTimeError and
plot_dist_mult_all (the three-panel plotting code). Repository commit checked:
7df19be4891c71af55a0f16a99153ed178a2b306.

For every generalized box, with coordinates in metres and timestamps in seconds:

- Area A = (x_max-x_min)*(y_max-y_min)/1e6, in km².
- Duration T = (timestamp_max-timestamp_min)/60, in minutes.
- M = total number of generalized boxes across all output records.
- m = floor(0.95*M), independently sorting areas and durations.
- Mean spatial error [km] = sqrt(mean(sorted(A)[:m])).
- Mean time error [min] = mean(sorted(T)[:m]).

These match the evaluation notebook's formulas, not a derivation from a fully
specified Table II protocol. In particular the spatial quantity is NOT the
mean width+height, NOT mean(sqrt(A)), and NOT point-to-center Euclidean error.
Each generalized box has one vote, without weighting by N_i_segment or taking
an unweighted mean across users. Each published record counts even if another
record has identical geometry. The two upper tails are trimmed independently.

trim_fraction defaults to .95; set it to 1 for the full distributions. Untrimmed
values are always reported separately. If floor(trim_fraction*M)=0, headline
errors are undefined and exported as None/empty, never silently substituted.
Zero-area and zero-duration boxes are preserved. No artificial 500 m footprint
is added to point-center bounds.

## CDFs

The main empirical CDF is F(a)=number(A <= a)/M. All observations, including
zero-area samples, are included and there is no 95% truncation.
`spatial_thresholds_km2` specifies any desired nonnegative finite thresholds.
The full ECDF groups repeated values and ends at 1.

The original plot uses histogram edges starting at .1 km², then .2,.3,.5,1,2,4,
7,10,25,50,100,500,max(area). It drops values below .1 from its denominator and
counts values strictly below each internal upper edge. We export this as a
SEPARATE compatibility curve; it is not the full empirical CDF. If max(area)<=500,
the original edge list is invalid or problematic; our fixed-threshold conditional
CDF remains defined. If all values are below .1, that conditional CDF is undefined.
Its numerical threshold values are preserved, but plotting uses a numerical axis
rather than the original categorical spacing. symlog supports zero on the ECDF.

## Outputs in each metrics directory

- paper_metrics.csv: named errors, retained fraction and sample counts.
- metadata.json: headline errors, untrimmed errors, legacy metrics, source commit,
  data selection parameters and generalizer identity.
- trajectory_metrics.parquet and segments/: existing per-user/per-box metrics.
- spatial_accuracy_ecdf.parquet / .csv: full exact empirical CDF.
- spatial_accuracy_thresholds.csv: count and probability at requested thresholds.
- spatial_accuracy_repository_histogram.csv: explicitly conditional legacy curve.

The notebook runs k=2,5,10 on the same Sigma/KNN/C_tilde, exports
paper_metrics_comparison.csv, and plots spatial_accuracy_cdf.png/.pdf and a
separately named repository histogram curve. It validates K and population size.

## Recompute only evaluation for an existing result

```python
from pathlib import Path
from kscale import calculate_metrics, plot_spatial_accuracy_cdf

summary = calculate_metrics(
    Path("run_01/Sigma"), Path("run_01/Sigma_tilde"),
    Path("run_01/metrics_v2"),
    spatial_thresholds_km2=[0, .1, .5, 1, 2, 5, 10, 25, 50, 100],
    trim_fraction=.95,
)
print(summary["mean_spatial_error_km"], summary["mean_time_error_min"])
ax = plot_spatial_accuracy_cdf({"k=2": Path("run_01/metrics_v2")})
```

No need to rerun KNN or K-Merge just to update these metrics. Changing the sampled
users or time window does require rebuilding all stages. Metrics retain all
area/duration scalars in memory for sorting (O(M) storage); full containment
verification can take time and is enabled by default.

## Reproduction limits

Table II reports Senegal datasets with k=2; it is not a YJMob reference table.
Figure 5 includes mob/YJMob with k=2,5,10; Table I reports 75 days for mob.
A one-hour window or a subset of users changes the experiment. K=10, exact KNN,
reciprocal candidate expansion, corrected RG and the quadratic partition DP
also differ from the original pipeline. Matching definitions does not imply
matching the paper's numeric results. The DP is exact only within its chosen
contiguous-partition family and sum-of-mean-cost objective. No new loss function
or linear original k-merge implementation is introduced by this metric update.

Tests: `python -m pytest tests -q`.
