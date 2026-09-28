"""Merge the midnight ERA5 files with the CLARA observations that motivated them.

The other two subsets keep every ERA5 hour x longitude x latitude row and attach
CLARA where it exists. Here only the CLARA-matched rows are kept, so the output
is about a thousand rows and there is no need to stream: each request file is
merged in turn and the pieces are concatenated at the end.

Every request covers its own small box, so each file is snapped to its own grid.
Observations are routed to files by the request planner, which assigns each one
to exactly one request, so no observation can be counted twice.
"""

import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "paths.py").exists():
        sys.path.insert(0, str(_p))
        break
from paths import subset_data_path, subset_results_path

import argparse

import numpy as np
import pandas as pd

from read_in.midnight_subset.midnight import (
    CLARA_LOCAL_TIME_COLUMN,
    CLARA_TIME_COLUMN,
    HOUR_COLUMN,
    REQUEST_COLUMN,
    SUBSET,
    plan_requests,
)
from read_in.temporal_subset.data_preprocessing import (
    CLARA_LATITUDE_COLUMN,
    CLARA_RADIANCE_COLUMN,
    KEY_COLUMNS,
    aggregate_clara_hourly,
    radiance_quality_mask,
    build_era5_index,
    get_common_grid,
    load_clara,
    load_era5,
    make_grid_frame,
    make_merged_hour_frame,
    nearest_grid_values,
    nearest_longitude_values,
    plot_clara_timeseries,
)


DEFAULT_CLARA_PATH = subset_data_path("CLARA_matched.pkl", subset=SUBSET)
DEFAULT_ERA5_DIR = subset_data_path(subset=SUBSET)
DEFAULT_CLARA_HOURLY_PATH = subset_data_path("CLARA_hourly_by_grid.pkl", subset=SUBSET)
DEFAULT_MERGED_PATH = subset_data_path("CLARA_ERA5_merged.parquet", subset=SUBSET)
DEFAULT_PLOT_PATH = subset_results_path("preprocessing/clara_raw_and_hourly_mean.png", subset=SUBSET)

LOCAL_TIME_COLUMN = "local_time"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Merge each midnight ERA5 request file with its CLARA observations "
            "and write the CLARA-matched rows to one parquet."
        )
    )
    parser.add_argument("--clara-path", type=Path, default=DEFAULT_CLARA_PATH)
    parser.add_argument("--era5-dir", type=Path, default=DEFAULT_ERA5_DIR)
    parser.add_argument("--clara-hourly-output", type=Path, default=DEFAULT_CLARA_HOURLY_PATH)
    parser.add_argument("--merged-output", type=Path, default=DEFAULT_MERGED_PATH)
    parser.add_argument("--plot-output", type=Path, default=DEFAULT_PLOT_PATH)
    parser.add_argument(
        "--allow-missing-files",
        action="store_true",
        help="Merge whatever has been downloaded instead of stopping. For smoke tests.",
    )
    return parser.parse_args()


def circular_mean_hours(values):
    """Mean clock time of values that may sit on either side of midnight.

    A plain mean of 23.9 and 0.1 is 12.0, the opposite time of day, so the
    values are averaged as angles and mapped back into [0, 24).
    """
    angles = 2.0 * np.pi * np.asarray(values, dtype=float) / 24.0
    mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
    return float(np.mod(mean_angle * 24.0 / (2.0 * np.pi), 24.0))


def local_time_by_cell(observations, latitude, longitude, longitude_column):
    """Mean CLARA local clock time per (hour, longitude, latitude) grid cell.

    ``aggregate_clara_hourly`` returns radiance only, and the merged local time
    is what the models read (modelling/covariate_preprocessing.py). Without it
    they fall back to CET for every cell, which is wrong for a global subset.

    The same radiance filter applies here, so a cell's local time is the mean
    over the observations that produced its radiance, not over ones that were
    rejected as calibration or off-nominal views.
    """
    frame = observations.loc[
        radiance_quality_mask(observations[CLARA_RADIANCE_COLUMN]),
        [CLARA_TIME_COLUMN, CLARA_LOCAL_TIME_COLUMN, CLARA_LATITUDE_COLUMN, longitude_column],
    ].dropna().copy()

    frame["hour"] = frame[CLARA_TIME_COLUMN].dt.floor("h")
    frame["latitude"] = nearest_grid_values(frame[CLARA_LATITUDE_COLUMN], latitude).astype(np.float32)
    frame["longitude"] = nearest_longitude_values(frame[longitude_column], longitude).astype(np.float32)

    return (
        frame.groupby(KEY_COLUMNS, as_index=False)[CLARA_LOCAL_TIME_COLUMN]
        .agg(lambda values: circular_mean_hours(values))
        .rename(columns={CLARA_LOCAL_TIME_COLUMN: LOCAL_TIME_COLUMN})
    )


def merge_request(path, observations):
    """CLARA-matched rows for one request file, or None if nothing matched."""
    era5_datasets = [dataset.load() for dataset in load_era5(path)]
    try:
        latitude, longitude = get_common_grid(era5_datasets)
        dataset_indexes, era5_hours, variable_names = build_era5_index(era5_datasets)

        clara_hourly, longitude_column = aggregate_clara_hourly(observations, latitude, longitude)
        if clara_hourly.empty:
            print(f"  no CLARA observation fell on this grid ({len(observations)} candidates)")
            return None, clara_hourly

        local_times = local_time_by_cell(observations, latitude, longitude, longitude_column)
        grid_frame = make_grid_frame(latitude, longitude)

        available_hours = set(era5_hours)
        wanted_hours = sorted(pd.to_datetime(clara_hourly["hour"].unique()))
        missing_hours = [hour for hour in wanted_hours if hour not in available_hours]
        if missing_hours:
            print(f"  WARNING: ERA5 has no data for {len(missing_hours)} CLARA hour(s): {missing_hours}")

        frames = []
        for hour in wanted_hours:
            if hour not in available_hours:
                continue
            frame = make_merged_hour_frame(
                hour=hour,
                grid_frame=grid_frame,
                dataset_indexes=dataset_indexes,
                variable_names=variable_names,
                clara_hourly=clara_hourly,
            )
            frames.append(frame[frame["clara_n_datapoints"] > 0])

        if not frames:
            return None, clara_hourly

        merged = pd.concat(frames, ignore_index=True)
        merged = merged.merge(local_times, how="left", on=KEY_COLUMNS)
        return merged, clara_hourly
    finally:
        for dataset in era5_datasets:
            dataset.close()


def restore_dtypes(frame):
    """Undo the widening that concatenating files with different columns causes."""
    for column in frame.columns:
        if column.endswith("_n_datapoints"):
            frame[column] = frame[column].fillna(0)
            frame[column] = frame[column].astype(
                np.int32 if column == "clara_n_datapoints" else np.int16
            )
        elif column in {"longitude", "latitude", LOCAL_TIME_COLUMN} or column.startswith("era5_"):
            frame[column] = frame[column].astype(np.float32)

    frame["clara_radiance_hourly_mean"] = frame["clara_radiance_hourly_mean"].astype(np.float32)
    return frame


def main():
    args = parse_args()

    clara = load_clara(args.clara_path)
    observations, requests = plan_requests(clara)
    plot_clara_timeseries(clara, args.plot_output)

    missing_files = [
        row.filename for row in requests.itertuples(index=False)
        if not (args.era5_dir / row.filename).exists()
    ]
    if missing_files and not args.allow_missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} of {len(requests)} ERA5 request files are missing, "
            f"e.g. {missing_files[:3]}. Run data_fetch.py, or pass --allow-missing-files."
        )
    if missing_files:
        print(f"skipping {len(missing_files)} request(s) that are not downloaded yet")

    merged_frames = []
    hourly_frames = []

    for number, row in enumerate(requests.itertuples(index=False), start=1):
        path = args.era5_dir / row.filename
        if not path.exists():
            continue

        print(f"[{number}/{len(requests)}] {row.filename} ({row.n_observations} CLARA obs)")
        request_observations = observations[observations[REQUEST_COLUMN] == row.request_id]
        merged, clara_hourly = merge_request(path, request_observations)

        if clara_hourly is not None and not clara_hourly.empty:
            hourly_frames.append(clara_hourly)
        if merged is not None and not merged.empty:
            merged_frames.append(merged)
            print(f"  {len(merged):,} matched grid cell(s)")

    if not merged_frames:
        raise ValueError("No CLARA observation could be matched to ERA5.")

    merged = pd.concat(merged_frames, ignore_index=True).sort_values(KEY_COLUMNS)
    merged = restore_dtypes(merged.reset_index(drop=True))
    clara_hourly = (
        pd.concat(hourly_frames, ignore_index=True)
        .sort_values(KEY_COLUMNS)
        .reset_index(drop=True)
    )

    args.merged_output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.merged_output, compression="zstd", index=False)
    args.clara_hourly_output.parent.mkdir(parents=True, exist_ok=True)
    clara_hourly.to_pickle(args.clara_hourly_output)

    matched_observations = int(merged["clara_n_datapoints"].sum())
    print(f"\nSaved CLARA hourly means to {args.clara_hourly_output}")
    print(f"Saved CLARA raw/hourly plot to {args.plot_output}")
    print(f"Saved merged CLARA-matched dataset to {args.merged_output}")
    print(
        f"Merged shape: {len(merged):,} rows x {merged.shape[1]} columns "
        f"({merged['hour'].nunique():,} hours)"
    )
    print(
        f"CLARA observations matched: {matched_observations:,} of "
        f"{len(observations):,} ({len(observations) - matched_observations:,} unmatched)"
    )
    print(
        f"local_time range: {merged[LOCAL_TIME_COLUMN].min():.2f} .. "
        f"{merged[LOCAL_TIME_COLUMN].max():.2f} (expected near 0 or near 24)"
    )


if __name__ == "__main__":
    main()
