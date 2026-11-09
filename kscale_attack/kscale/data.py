"""Read raw records and persist Sigma = (S_1, ..., S_U)."""
from pathlib import Path
from uuid import uuid4
import numpy as np
import pandas as pd
from ._io import new_stage, write_metadata, read_metadata

SAMPLE_COLUMNS = ["x", "y", "timestamp", "site_id"]


def read_and_parse_data(input_path, output_path, *, trajectory_column="uid",
                        source_format="yjmob", time_window_seconds=None,
                        sample_users=None, random_state=42):
    """Write one four-column Parquet per S_i and a separate manifest.

    yjmob: input uid,d,t,x,y; 1-based 500 m grid and 30-minute slots.
    standard: input trajectory_column,x,y,timestamp,site_id; coordinates in
    projected metres, timestamps in seconds (numeric) or ISO date/time strings.
    sample_users selects whole users without replacement, never individual rows.
    random_state makes this selection reproducible for a fixed input row order.
    No time filter is applied unless time_window_seconds is supplied.
    """
    if source_format not in {"yjmob", "standard"}:
        raise ValueError("source_format must be 'yjmob' or 'standard'.")
    if time_window_seconds is not None and (
        not np.isfinite(time_window_seconds) or time_window_seconds < 0
    ):
        raise ValueError("time_window_seconds must be finite and nonnegative.")
    source = str(input_path)
    raw = (pd.read_parquet(source) if source.rstrip('/').endswith('.parquet')
           or Path(source).is_dir() else pd.read_csv(source))
    required = ([trajectory_column, "d", "t", "x", "y"] if source_format == "yjmob"
                else [trajectory_column, *SAMPLE_COLUMNS])
    missing = set(required) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if raw[required].isna().any().any():
        raise ValueError("Required input columns contain missing values.")
    if raw.empty:
        raise ValueError("The input dataset is empty.")
    raw = raw[required].copy()
    input_samples = len(raw)
    available_users = raw[trajectory_column].nunique()
    if sample_users is not None:
        from ._io import positive_integer
        sample_users = positive_integer(sample_users, "sample_users")
        if sample_users > available_users:
            raise ValueError("sample_users exceeds the number of available users.")
        source_users = raw[trajectory_column].drop_duplicates().to_numpy()
        selected = np.random.default_rng(random_state).choice(
            len(source_users), size=sample_users, replace=False)
        raw = raw[raw[trajectory_column].isin(source_users[selected])].copy()
    samples_after_user_sampling = len(raw)
    if source_format == "yjmob":
        for name in ["d", "t", "x", "y"]:
            raw[name] = pd.to_numeric(raw[name], errors="raise")
            if not np.isfinite(raw[name]).all() or (raw[name] % 1 != 0).any():
                raise ValueError(f"YJMob {name} must contain finite integers.")
        if (raw[["x", "y"]] < 1).any().any() or not raw.t.between(0, 47).all():
            raise ValueError("YJMob requires x,y >= 1 and t in [0, 47].")
        # Site identity is a grid cell, never a user identifier.
        raw["site_id"] = raw.x.astype('int64').astype(str) + ':' + raw.y.astype('int64').astype(str)
        raw["timestamp"] = raw.d * 86400 + raw.t * 1800
        raw[["x", "y"]] = (raw[["x", "y"]] - 1) * 500 + 250
    elif not pd.api.types.is_numeric_dtype(raw.timestamp):
        timestamps = pd.to_datetime(raw.timestamp, utc=True, errors="raise")
        if timestamps.isna().any():
            raise ValueError("timestamp contains missing dates.")
        raw["timestamp"] = timestamps.astype('datetime64[ns, UTC]').astype('int64') / 1e9
    for name in ["x", "y", "timestamp"]:
        raw[name] = pd.to_numeric(raw[name], errors="raise").astype('float64')
        if not np.isfinite(raw[name]).all():
            raise ValueError(f"{name} contains non-finite values.")
    # Preserve distinct source IDs even when their string representations match.
    raw["_i"], source_ids = pd.factorize(raw[trajectory_column], sort=False)
    raw = raw.sort_values(["_i", "timestamp"], kind="stable")
    original_samples = len(raw)
    if time_window_seconds is not None:
        first = raw.groupby("_i").timestamp.transform("min")
        raw = raw[raw.timestamp - first <= time_window_seconds]
    with new_stage(output_path) as output:
        (output / "Sigma").mkdir()
        manifest = []
        for i, S_i in raw.groupby("_i", sort=True):
            relative = f"Sigma/S_{i:08d}.parquet"
            S_i[SAMPLE_COLUMNS].to_parquet(output / relative, index=False)
            manifest.append({"i": int(i), "source_id": str(source_ids[i]),
                             "path": relative, "N_i": len(S_i)})
        pd.DataFrame(manifest).to_parquet(output / "manifest.parquet", index=False)
        write_metadata(output, stage="Sigma", dataset_id=str(uuid4()), U=len(manifest),
                       source_format=source_format, coordinate_unit="m", time_unit="s",
                       time_window_seconds=time_window_seconds,
                       input_samples=input_samples, retained_samples=len(raw),
                       available_users=int(available_users), sample_users=sample_users,
                       random_state=random_state,
                       samples_after_user_sampling=samples_after_user_sampling,
                       samples_removed_by_time_window=original_samples-len(raw),
                       sample_columns=SAMPLE_COLUMNS)
    return Path(output_path).resolve()


def load_S_i(Sigma, i):
    """Read S_i; integer indices are zero-based in Python."""
    U = read_metadata(Sigma)["U"]
    if not isinstance(i, (int, np.integer)) or not 0 <= i < U:
        raise IndexError(i)
    return pd.read_parquet(Path(Sigma) / "Sigma" / f"S_{i:08d}.parquet")
