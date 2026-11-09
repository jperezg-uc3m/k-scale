"""Independent accuracy metrics for saved generalized trajectories (Section IV)."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from ._io import new_stage, read_metadata, write_metadata, require_dataset
from .data import load_S_i
from .merge import BOUND_COLUMNS


def _contains_all(G, samples):
    """Check geometric containment, including shared temporal boundaries."""
    for sample in samples:
        start = np.searchsorted(G[:, 5], sample[2], side="left")
        stop = np.searchsorted(G[:, 2], sample[2], side="right")
        candidates = G[start:stop]
        if not np.any(np.all(sample >= candidates[:, :3], axis=1)
                      & np.all(sample <= candidates[:, 3:], axis=1)):
            return False
    return True


def calculate_metrics(Sigma, Sigma_tilde, output_path, *, verify_containment=True,
                      spatial_thresholds_km2=None, trim_fraction=0.95):
    """Save area, spatial span, duration, Eq. (4) cost and coverage checks.

    Generalized-sample averages weight each box equally. Original-sample
    averages weight a box by the number of S_i samples mapped into it.
    Geometry uses point coordinates in metres, not full cell footprints.
    Repository-compatible headline errors retain floor(trim_fraction * M)
    smallest generalized-sample areas/durations, sorted independently. Spatial
    error is sqrt(mean(area)), NOT mean(sqrt(area)) or width+height.
    Exact ECDF uses ALL generalized samples, including zeros, without trimming.
    Also exports the repository's histogram convention separately.
    Does not reproduce the paper's entropy/predictability/mobility-law studies.
    """
    if not np.isfinite(trim_fraction) or not 0 < trim_fraction <= 1:
        raise ValueError("trim_fraction must be in (0, 1].")
    thresholds = np.asarray(DEFAULT_SPATIAL_THRESHOLDS if spatial_thresholds_km2 is None
                            else spatial_thresholds_km2, dtype=float)
    if thresholds.ndim != 1 or not len(thresholds) or not np.isfinite(thresholds).all() or (thresholds < 0).any():
        raise ValueError("Spatial thresholds must be a nonempty list of finite nonnegative areas.")
    thresholds = np.unique(thresholds)
    metadata = read_metadata(Sigma_tilde)
    require_dataset(Sigma, metadata)
    Sigma_tilde = Path(Sigma_tilde)
    manifest = pd.read_parquet(Sigma_tilde / "manifest.parquet")
    original_manifest = pd.read_parquet(Path(Sigma) / "manifest.parquet").set_index("i")
    if metadata.get("coordinate_unit") != "m" or metadata.get("time_unit") != "s":
        raise ValueError("Metrics require coordinates in metres and timestamps in seconds.")
    A = load_npz(Sigma_tilde / "A.npz").tocsr()
    U = metadata["U"]
    coverage = np.asarray(A.sum(axis=0)).ravel() + 1
    records = []
    areas, durations = [], []
    total_segments = total_samples = 0
    box_sum = np.zeros(3)
    sample_sum = np.zeros(3)
    with new_stage(output_path) as output:
        (output / "segments").mkdir()
        for row in manifest.itertuples(index=False):
            i = int(row.i)
            frame = pd.read_parquet(Sigma_tilde / row.path)
            G = frame[BOUND_COLUMNS].to_numpy(dtype=float)
            raw_counts = frame.N_i_segment.to_numpy(dtype=float)
            if not np.isfinite(raw_counts).all() or (raw_counts % 1 != 0).any():
                raise ValueError(f"Invalid sample counts for G_{i}.")
            counts = raw_counts.astype(np.int64)
            N_i = int(original_manifest.loc[i, "N_i"])
            if (len(G) == 0 or not np.isfinite(G).all() or np.any(G[:, 3:] < G[:, :3])
                    or np.any(G[:-1, 5] > G[1:, 2]) or np.any(counts < 1)
                    or counts.sum() != N_i):
                raise ValueError(f"Invalid generalized trajectory G_{i}.")
            if verify_containment:
                for j in [i, *A.indices[A.indptr[i]:A.indptr[i + 1]]]:
                    S_j = load_S_i(Sigma, int(j))[["x", "y", "timestamp"]].to_numpy(dtype=float)
                    if not _contains_all(G, S_j):
                        raise ValueError(f"G_{i} does not contain every sample of S_{j}.")
            delta_x, delta_y, delta_t = (G[:, 3:] - G[:, :3]).T
            measures = np.column_stack([delta_x * delta_y / 1e6,
                                        (delta_x + delta_y) / 1000, delta_t / 60])
            areas.append(measures[:, 0])
            durations.append(measures[:, 2])
            C_G_i = float(np.dot(delta_t * (delta_x + delta_y), counts) / N_i)
            if not np.isclose(C_G_i, row.C_G_i, rtol=1e-10, atol=1e-10):
                raise ValueError(f"Stored and recomputed costs disagree for G_{i}.")
            pd.DataFrame({"segment": np.arange(len(G)), "area_km2": measures[:, 0],
                          "spatial_span_km": measures[:, 1], "duration_min": measures[:, 2],
                          "N_i_segment": counts}).to_parquet(
                output / "segments" / f"G_{i:08d}.parquet", index=False)
            means = np.average(measures, axis=0, weights=counts)
            records.append({"i": i, "N_i": N_i, "M_i": len(G), "C_G_i": C_G_i,
                            "sample_weighted_area_km2": means[0],
                            "sample_weighted_spatial_span_km": means[1],
                            "sample_weighted_duration_min": means[2],
                            "generalization_count": int(coverage[i])})
            box_sum += measures.sum(axis=0)
            sample_sum += (measures * counts[:, None]).sum(axis=0)
            total_segments += len(G)
            total_samples += N_i
        metrics = pd.DataFrame(records)
        metrics.to_parquet(output / "trajectory_metrics.parquet", index=False)
        summary = {"U": U, "k": metadata["k"], "original_samples": total_samples,
                   "generalized_samples": total_segments,
                   "mean_C_G_i_m_s": float(metrics.C_G_i.mean()),
                   "minimum_generalization_count": int(coverage.min()),
                   "nu_k_pick_satisfied": bool(np.all(coverage >= metadata["k"])),
                   "assigned_sample_containment_verified": bool(verify_containment)}
        for column, position in [("area_km2", 0), ("spatial_span_km", 1), ("duration_min", 2)]:
            summary[f"generalized_sample_mean_{column}"] = float(box_sum[position] / total_segments)
            summary[f"original_sample_mean_{column}"] = float(sample_sum[position] / total_samples)
        area = np.concatenate(areas)
        duration = np.concatenate(durations)
        paper = repository_error_summary(area, duration, trim_fraction)
        summary.update(paper)
        summary.update({
            "metric_weighting": "one_vote_per_generalized_sample",
            "spatial_error_definition": "sqrt(mean(lowest floor(trim_fraction*M) areas_km2))",
            "time_error_definition": "mean(lowest floor(trim_fraction*M) durations_min)",
            "evaluation_source": EVALUATION_SOURCE,
            "evaluation_commit": EVALUATION_COMMIT,
            "dataset_config": read_metadata(Sigma),
            "gen_implementation": metadata.get("gen_implementation", "unknown"),
        })
        ecdf = empirical_cdf(area)
        ecdf.to_parquet(output / "spatial_accuracy_ecdf.parquet", index=False)
        ecdf.to_csv(output / "spatial_accuracy_ecdf.csv", index=False)
        sorted_area = np.sort(area)
        threshold_table = pd.DataFrame({
            "spatial_accuracy_km2": thresholds,
            "count_le": np.searchsorted(sorted_area, thresholds, side="right"),
        })
        threshold_table["cdf"] = threshold_table.count_le / len(area)
        threshold_table.to_csv(output / "spatial_accuracy_thresholds.csv", index=False)
        repository_histogram_cdf(area).to_csv(output / "spatial_accuracy_repository_histogram.csv", index=False)
        pd.DataFrame([{
            "k": metadata["k"],
            "Mean spatial error [km]": paper["mean_spatial_error_km"],
            "Mean time error [min]": paper["mean_time_error_min"],
            "Retained fraction": trim_fraction,
            "Generalized samples": len(area),
            "Retained samples per metric": paper["trimmed_sample_count"],
        }]).to_csv(output / "paper_metrics.csv", index=False)
        write_metadata(output, stage="metrics", dataset_id=metadata["dataset_id"], **summary)
    return summary


EVALUATION_COMMIT = "7df19be4891c71af55a0f16a99153ed178a2b306"
EVALUATION_SOURCE = f"https://github.com/nds-group/k-scale/blob/{EVALUATION_COMMIT}/Evaluation/eval_k-scale.ipynb"
DEFAULT_SPATIAL_THRESHOLDS = [0, 0.1, 0.2, 0.3, 0.5, 1, 2, 4, 7, 10, 25, 50, 100, 500]


def _valid_measurements(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Measurements must be a nonempty finite nonnegative vector.")
    return values


def repository_error_summary(area_km2, duration_min, trim_fraction=0.95):
    """Match getPosError/getTimeError in the authors' evaluation notebook.

    Trim spatial and temporal samples independently; do not filter one using
    the other's mask. A single box at fraction .95 retains zero samples, so
    the trimmed metrics are undefined (None), rather than silently untrimmed.
    """
    area = _valid_measurements(area_km2)
    duration = _valid_measurements(duration_min)
    if len(area) != len(duration):
        raise ValueError("Area and duration arrays must have equal lengths.")
    if not np.isfinite(trim_fraction) or not 0 < trim_fraction <= 1:
        raise ValueError("trim_fraction must be in (0, 1].")
    m = int(trim_fraction * len(area))
    return {
        "mean_spatial_error_km": float(np.sqrt(np.sort(area)[:m].mean())) if m else None,
        "mean_time_error_min": float(np.sort(duration)[:m].mean()) if m else None,
        "trim_fraction": float(trim_fraction),
        "trimmed_sample_count": m,
        "untrimmed_spatial_error_km": float(np.sqrt(area.mean())),
        "untrimmed_mean_time_error_min": float(duration.mean()),
        "zero_area_fraction": float(np.mean(area == 0)),
        "zero_duration_fraction": float(np.mean(duration == 0)),
    }


def empirical_cdf(area_km2):
    """Exact right-continuous F(a) = count(area <= a)/M, including zeros."""
    values = _valid_measurements(area_km2)
    unique, counts = np.unique(values, return_counts=True)
    cumulative = np.cumsum(counts)
    return pd.DataFrame({"spatial_accuracy_km2": unique, "count": counts,
                         "count_le": cumulative, "cdf": cumulative / len(values)})


def repository_histogram_cdf(area_km2):
    """Repository plot convention at its fixed internal bin boundaries.

    Its histogram starts at 0.1 km², excluding smaller areas from the denominator.
    At each internal edge a, cumulative bins count 0.1 <= area < a (not <=).
    We preserve this distinction explicitly. For maxima <= 500 the original
    edge list is invalid/nonmonotone; fixed thresholds here remain well-defined.
    All-below-0.1 input gives an undefined conditional CDF, represented as NaN.
    """
    area = np.sort(_valid_measurements(area_km2))
    thresholds = np.asarray(DEFAULT_SPATIAL_THRESHOLDS[2:])
    below = int(np.searchsorted(area, 0.1, side="left"))
    eligible = len(area) - below
    numerator = np.searchsorted(area, thresholds, side="left") - below
    cdf = numerator / eligible if eligible else np.full(len(thresholds), np.nan)
    return pd.DataFrame({"spatial_accuracy_km2": thresholds,
                         "cdf_repository_histogram": cdf,
                         "excluded_below_0_1": below, "eligible_samples": eligible})


def plot_spatial_accuracy_cdf(metric_paths, *, convention="ecdf", xscale="symlog", ax=None):
    """Plot {label: metrics_directory}; use the same sampled Sigma for all k.

    Returns a Matplotlib Axes. Caller controls display and optional savefig.
    symlog retains zero-area observations while showing the broad area range.
    """
    import matplotlib.pyplot as plt
    if not metric_paths:
        raise ValueError("Provide at least one metrics directory.")
    if convention not in {"ecdf", "repository_histogram"}:
        raise ValueError("convention must be 'ecdf' or 'repository_histogram'.")
    if xscale not in {"linear", "symlog"}:
        raise ValueError("Use linear or symlog to retain zero-area samples.")
    metadata = [read_metadata(p) for p in metric_paths.values()]
    if len({m["dataset_id"] for m in metadata}) != 1:
        raise ValueError("Use the same Sigma (same sampled users/time window) to compare k values.")
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    for label, path in metric_paths.items():
        path = Path(path)
        if convention == "ecdf":
            frame = pd.read_parquet(path / "spatial_accuracy_ecdf.parquet")
            x, y = frame.spatial_accuracy_km2.to_numpy(), frame.cdf.to_numpy()
            # Add the vertical jump at the first observed value, including zero.
            ax.step(np.r_[0, x[0], x], np.r_[0, 0, y], where="post", label=str(label))
        else:
            frame = pd.read_csv(path / "spatial_accuracy_repository_histogram.csv")
            ax.plot(frame.spatial_accuracy_km2, frame.cdf_repository_histogram,
                    marker="o", linestyle="--", label=str(label))
    if xscale == "symlog":
        ax.set_xscale("symlog", linthresh=0.1)
    else:
        ax.set_xscale("linear")
    ax.set(xlabel=r"Spatial accuracy (km$^2$)", ylabel="CDF", ylim=(0, 1.02))
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.figure.tight_layout()
    return ax
