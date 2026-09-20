import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "paths.py").exists():
        sys.path.insert(0, str(_p))
        break
from paths import subset_data_path, subset_results_path

import argparse
import os
import tempfile
import warnings

import cfgrib

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# The helpers below are shared with read_in/spatial_subset/data_preprocessing.py;
# only these defaults are specific to the temporal subset.
SUBSET = "temporal_subset"

DEFAULT_CLARA_PATH = subset_data_path("CLARA_matched.pkl", subset=SUBSET)
DEFAULT_ERA5_PATH = subset_data_path("ERA5_matched.grib", subset=SUBSET)
DEFAULT_CLARA_HOURLY_PATH = subset_data_path("CLARA_hourly_by_grid.pkl", subset=SUBSET)
DEFAULT_MERGED_PATH = subset_data_path("CLARA_ERA5_merged.parquet", subset=SUBSET)
DEFAULT_PLOT_PATH = subset_results_path("preprocessing/clara_raw_and_hourly_mean.png", subset=SUBSET)

CLARA_TIME_COLUMN = "TimeJD"
CLARA_RADIANCE_COLUMN = "CLARA_radiance"
CLARA_LATITUDE_COLUMN = "CLARA_fov_latitude"
CLARA_LONGITUDE_COLUMN = "CLARA_fov_longitude"
CLARA_POSITIVE_LONGITUDE_COLUMN = "CLARA_fov_longitude_positive"

KEY_COLUMNS = ["hour", "longitude", "latitude"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate CLARA and ERA5 to hourly grid cells, then keep every "
            "ERA5 hour/longitude/latitude row and attach CLARA radiance where present."
        )
    )
    parser.add_argument("--clara-path", type=Path, default=DEFAULT_CLARA_PATH)
    parser.add_argument("--era5-path", type=Path, default=DEFAULT_ERA5_PATH)
    parser.add_argument("--clara-hourly-output", type=Path, default=DEFAULT_CLARA_HOURLY_PATH)
    parser.add_argument("--merged-output", type=Path, default=DEFAULT_MERGED_PATH)
    parser.add_argument("--plot-output", type=Path, default=DEFAULT_PLOT_PATH)
    parser.add_argument(
        "--limit-hours",
        type=int,
        default=None,
        help="Only process the first N ERA5 hours. Useful for quick checks.",
    )
    return parser.parse_args()


def load_clara(path):
    clara = pd.read_pickle(path)
    clara[CLARA_TIME_COLUMN] = pd.to_datetime(clara[CLARA_TIME_COLUMN])
    return clara


# Let cfgrib write one <file>.<hash>.idx beside each GRIB and reuse it. Building
# that index means scanning every message in the file, which takes ~30 minutes for
# a month of the spatial subset; with the index on disk later opens are seconds.
# It is one extra file per GRIB, which the Euler inode quota does not notice.
BACKEND_KWARGS = {}


def load_era5(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return cfgrib.open_datasets(path, backend_kwargs=BACKEND_KWARGS)


def get_common_grid(era5_datasets):
    latitude = None
    longitude = None

    for dataset in era5_datasets:
        if latitude is None and "latitude" in dataset.coords:
            latitude = dataset.latitude.values
        if longitude is None and "longitude" in dataset.coords:
            longitude = dataset.longitude.values

    if latitude is None or longitude is None:
        raise ValueError("Could not find latitude and longitude coordinates in the ERA5 datasets.")

    return np.asarray(latitude), np.asarray(longitude)


def choose_clara_longitude_column(clara, era5_longitude):
    if (
        era5_longitude.min() >= 0
        and era5_longitude.max() > 180
        and CLARA_POSITIVE_LONGITUDE_COLUMN in clara.columns
    ):
        return CLARA_POSITIVE_LONGITUDE_COLUMN

    return CLARA_LONGITUDE_COLUMN


def nearest_grid_values(values, grid):
    values = np.asarray(values, dtype=float)
    grid = np.asarray(grid, dtype=float)
    ascending_grid = np.sort(grid)

    right_index = np.searchsorted(ascending_grid, values, side="left")
    right_index = np.clip(right_index, 0, len(ascending_grid) - 1)
    left_index = np.clip(right_index - 1, 0, len(ascending_grid) - 1)

    right_distance = np.abs(ascending_grid[right_index] - values)
    left_distance = np.abs(values - ascending_grid[left_index])
    nearest_index = np.where(left_distance <= right_distance, left_index, right_index)
    return ascending_grid[nearest_index]


def is_global_longitude_grid(grid):
    sorted_grid = np.sort(np.asarray(grid, dtype=float))
    if len(sorted_grid) <= 1:
        return False

    resolution = float(np.median(np.diff(sorted_grid)))
    return np.isclose(resolution * len(sorted_grid), 360.0)


def nearest_longitude_values(values, grid):
    values = np.asarray(values, dtype=float)
    grid = np.asarray(grid, dtype=float)
    sorted_grid = np.sort(grid)

    if is_global_longitude_grid(sorted_grid):
        resolution = float(np.median(np.diff(sorted_grid)))
        normalized_values = np.mod(values - sorted_grid[0], 360.0) + sorted_grid[0]
        index = np.rint((normalized_values - sorted_grid[0]) / resolution).astype(int) % len(sorted_grid)
        return sorted_grid[index]
    return nearest_grid_values(values, sorted_grid)


def plot_clara_timeseries(clara, output_path):
    plot_data = clara[[CLARA_TIME_COLUMN, CLARA_RADIANCE_COLUMN]].dropna()
    plot_data = plot_data.sort_values(CLARA_TIME_COLUMN)
    hourly_mean = (
        plot_data.assign(hour=plot_data[CLARA_TIME_COLUMN].dt.floor("h"))
        .groupby("hour", as_index=False)[CLARA_RADIANCE_COLUMN]
        .mean()
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        plot_data[CLARA_TIME_COLUMN],
        plot_data[CLARA_RADIANCE_COLUMN],
        color="#9ca3af",
        linewidth=0.7,
        alpha=0.55,
        label="CLARA raw",
    )
    ax.plot(
        hourly_mean["hour"],
        hourly_mean[CLARA_RADIANCE_COLUMN],
        color="#2563eb",
        linewidth=2,
        marker="o",
        markersize=3,
        label="CLARA hourly mean",
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("CLARA radiance")
    ax.set_title("CLARA raw radiance and hourly mean")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)

    if plt.get_backend().lower() == "agg":
        plt.close(fig)
    else:
        plt.show()


def aggregate_clara_hourly(clara, era5_latitude, era5_longitude):
    longitude_column = choose_clara_longitude_column(clara, era5_longitude)
    needed_columns = [
        CLARA_TIME_COLUMN,
        CLARA_RADIANCE_COLUMN,
        CLARA_LATITUDE_COLUMN,
        longitude_column,
    ]
    missing_columns = [column for column in needed_columns if column not in clara.columns]
    if missing_columns:
        raise ValueError(f"CLARA input is missing columns: {missing_columns}")

    clara_grid = clara[needed_columns].dropna().copy()
    if not is_global_longitude_grid(era5_longitude):
        clara_grid = clara_grid[
            clara_grid[longitude_column].between(float(np.min(era5_longitude)), float(np.max(era5_longitude)))
        ]

    clara_grid["hour"] = clara_grid[CLARA_TIME_COLUMN].dt.floor("h")
    clara_grid["latitude"] = nearest_grid_values(clara_grid[CLARA_LATITUDE_COLUMN], era5_latitude)
    clara_grid["longitude"] = nearest_longitude_values(clara_grid[longitude_column], era5_longitude)

    clara_grid = clara_grid[
        clara_grid["latitude"].between(float(np.min(era5_latitude)), float(np.max(era5_latitude)))
    ]
    if np.min(era5_longitude) < 0:
        clara_grid = clara_grid[
            clara_grid["longitude"].between(float(np.min(era5_longitude)), float(np.max(era5_longitude)))
        ]

    clara_hourly = (
        clara_grid.groupby(KEY_COLUMNS, as_index=False)
        .agg(
            clara_radiance_hourly_mean=(CLARA_RADIANCE_COLUMN, "mean"),
            clara_n_datapoints=(CLARA_RADIANCE_COLUMN, "count"),
        )
        .sort_values(KEY_COLUMNS)
        .reset_index(drop=True)
    )
    clara_hourly["latitude"] = clara_hourly["latitude"].astype(np.float32)
    clara_hourly["longitude"] = clara_hourly["longitude"].astype(np.float32)
    clara_hourly["clara_n_datapoints"] = clara_hourly["clara_n_datapoints"].astype(np.int32)

    return clara_hourly, longitude_column


def valid_hours_for_dataset(dataset):
    time_coordinate = dataset["valid_time"] if "valid_time" in dataset.coords else dataset["time"]
    valid_hours = pd.to_datetime(time_coordinate.values.ravel()).floor("h")

    hour_positions = {}
    for flat_index, hour in enumerate(valid_hours):
        position = np.unravel_index(flat_index, time_coordinate.shape)
        hour_positions.setdefault(pd.Timestamp(hour), []).append(position)

    return hour_positions, time_coordinate.dims


def build_era5_index(era5_datasets):
    dataset_indexes = []
    all_hours = set()
    variable_names = []

    for dataset in era5_datasets:
        hour_positions, time_dims = valid_hours_for_dataset(dataset)
        all_hours.update(hour_positions)
        variable_names.extend(dataset.data_vars)
        dataset_indexes.append(
            {
                "dataset": dataset,
                "hour_positions": hour_positions,
                "time_dims": time_dims,
                "variables": list(dataset.data_vars),
            }
        )

    return dataset_indexes, sorted(all_hours), variable_names


def hourly_dataset_values(dataset_index, hour):
    positions = dataset_index["hour_positions"].get(hour, [])
    values = {}
    counts = {}

    if not positions:
        return values, counts

    for variable in dataset_index["variables"]:
        grids = []
        for position in positions:
            selector = dict(zip(dataset_index["time_dims"], position))
            grids.append(dataset_index["dataset"][variable].isel(selector).values)

        if len(grids) == 1:
            hourly_grid = grids[0]
        else:
            hourly_grid = np.nanmean(np.stack(grids, axis=0), axis=0)

        values[variable] = np.asarray(hourly_grid, dtype=np.float32)
        counts[variable] = len(grids)

    return values, counts


def make_grid_frame(latitude, longitude):
    longitude_grid, latitude_grid = np.meshgrid(longitude, latitude)
    return pd.DataFrame(
        {
            "longitude": longitude_grid.ravel().astype(np.float32),
            "latitude": latitude_grid.ravel().astype(np.float32),
        }
    )


def make_merged_hour_frame(hour, grid_frame, dataset_indexes, variable_names, clara_hourly):
    frame = grid_frame.copy()
    frame.insert(0, "hour", hour)

    variable_counts = {}
    for dataset_index in dataset_indexes:
        hourly_values, hourly_counts = hourly_dataset_values(dataset_index, hour)
        for variable in dataset_index["variables"]:
            output_column = f"era5_{variable}"
            count_column = f"era5_{variable}_n_datapoints"

            if variable in hourly_values:
                frame[output_column] = hourly_values[variable].ravel()
                count = hourly_counts[variable]
            else:
                frame[output_column] = np.nan
                count = 0

            frame[count_column] = np.int16(count)
            variable_counts[variable] = count

    present_counts = [count for count in variable_counts.values() if count > 0]
    frame["era5_n_datapoints"] = np.int16(max(present_counts) if present_counts else 0)

    clara_for_hour = clara_hourly[clara_hourly["hour"] == hour]
    frame = frame.merge(clara_for_hour, how="left", on=KEY_COLUMNS)
    frame["clara_n_datapoints"] = frame["clara_n_datapoints"].fillna(0).astype(np.int32)

    for variable in variable_names:
        mean_column = f"era5_{variable}"
        count_column = f"era5_{variable}_n_datapoints"
        frame[mean_column] = frame[mean_column].astype(np.float32)
        frame[count_column] = frame[count_column].astype(np.int16)

    frame["era5_n_datapoints"] = frame["era5_n_datapoints"].astype(np.int16)
    frame["clara_radiance_hourly_mean"] = frame["clara_radiance_hourly_mean"].astype(np.float32)
    return frame


def write_merged_parquet(
    output_path,
    era5_datasets,
    era5_latitude,
    era5_longitude,
    clara_hourly,
    limit_hours=None,
):
    dataset_indexes, hours, variable_names = build_era5_index(era5_datasets)
    if limit_hours is not None:
        hours = hours[:limit_hours]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid_frame = make_grid_frame(era5_latitude, era5_longitude)

    writer = None
    try:
        for hour_number, hour in enumerate(hours, start=1):
            frame = make_merged_hour_frame(
                hour=hour,
                grid_frame=grid_frame,
                dataset_indexes=dataset_indexes,
                variable_names=variable_names,
                clara_hourly=clara_hourly,
            )
            table = pa.Table.from_pandas(frame, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")

            writer.write_table(table)
            print(
                f"Wrote hour {hour_number}/{len(hours)}: {hour} "
                f"({len(frame):,} longitude/latitude rows)"
            )
    finally:
        if writer is not None:
            writer.close()

    return len(hours), len(grid_frame)


def main():
    args = parse_args()

    clara = load_clara(args.clara_path)
    era5_datasets = load_era5(args.era5_path)
    era5_latitude, era5_longitude = get_common_grid(era5_datasets)

    plot_clara_timeseries(clara, args.plot_output)
    clara_hourly, longitude_column = aggregate_clara_hourly(clara, era5_latitude, era5_longitude)
    args.clara_hourly_output.parent.mkdir(parents=True, exist_ok=True)
    clara_hourly.to_pickle(args.clara_hourly_output)

    n_hours, n_grid_cells = write_merged_parquet(
        output_path=args.merged_output,
        era5_datasets=era5_datasets,
        era5_latitude=era5_latitude,
        era5_longitude=era5_longitude,
        clara_hourly=clara_hourly,
        limit_hours=args.limit_hours,
    )

    print(f"Used CLARA longitude column: {longitude_column}")
    print(f"Saved CLARA hourly means to {args.clara_hourly_output}")
    print(f"Saved CLARA raw/hourly plot to {args.plot_output}")
    print(f"Saved merged hourly ERA5-left dataset to {args.merged_output}")
    print(f"Merged shape: {n_hours:,} hours x {n_grid_cells:,} longitude/latitude cells")
    print(f"CLARA hourly matches: {len(clara_hourly):,} hour/longitude/latitude cells")


if __name__ == "__main__":
    main()
