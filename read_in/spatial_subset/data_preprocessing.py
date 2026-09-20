"""Merge the monthly spatial-subset ERA5 GRIBs (Italy, 2020) with CLARA.

Same output layout as read_in/temporal_subset/data_preprocessing.py: one row
per ERA5 hour x longitude x latitude, with the CLARA hourly mean attached where
present. The only difference is the input, which here is twelve monthly GRIBs
instead of a single file. They are streamed one month at a time into a single
parquet, so peak memory stays at roughly one month of ERA5.
"""

import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "paths.py").exists():
        sys.path.insert(0, str(_p))
        break
from paths import subset_data_path, subset_results_path

import argparse
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from read_in.temporal_subset.data_preprocessing import (
    aggregate_clara_hourly,
    build_era5_index,
    get_common_grid,
    load_clara,
    load_era5,
    make_grid_frame,
    make_merged_hour_frame,
    plot_clara_timeseries,
)


SUBSET = "spatial_subset"

DEFAULT_CLARA_PATH = subset_data_path("CLARA_matched.pkl", subset=SUBSET)
DEFAULT_ERA5_DIR = subset_data_path(subset=SUBSET)
DEFAULT_ERA5_PATTERN = "ERA5_matched_2020*.grib"
DEFAULT_CLARA_HOURLY_PATH = subset_data_path("CLARA_hourly_by_grid.pkl", subset=SUBSET)
DEFAULT_MERGED_PATH = subset_data_path("CLARA_ERA5_merged.parquet", subset=SUBSET)
DEFAULT_PLOT_PATH = subset_results_path("preprocessing/clara_raw_and_hourly_mean.png", subset=SUBSET)

MONTH_FILE_PATTERN = re.compile(r"_(\d{4})(\d{2})\.grib$")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stream the monthly ERA5 GRIBs into one hourly parquet, keeping every "
            "ERA5 hour/longitude/latitude row and attaching CLARA radiance where present."
        )
    )
    parser.add_argument("--clara-path", type=Path, default=DEFAULT_CLARA_PATH)
    parser.add_argument("--era5-dir", type=Path, default=DEFAULT_ERA5_DIR)
    parser.add_argument("--era5-pattern", default=DEFAULT_ERA5_PATTERN)
    parser.add_argument("--clara-hourly-output", type=Path, default=DEFAULT_CLARA_HOURLY_PATH)
    parser.add_argument("--merged-output", type=Path, default=DEFAULT_MERGED_PATH)
    parser.add_argument("--plot-output", type=Path, default=DEFAULT_PLOT_PATH)
    parser.add_argument(
        "--months",
        nargs="+",
        default=None,
        help="Only process these months, e.g. --months 01 02. Useful for quick checks.",
    )
    parser.add_argument(
        "--limit-hours",
        type=int,
        default=None,
        help="Only process the first N hours of each month. Useful for quick checks.",
    )
    return parser.parse_args()


def monthly_grib_files(era5_dir, pattern, months=None):
    files = []
    for path in sorted(Path(era5_dir).glob(pattern)):
        match = MONTH_FILE_PATTERN.search(path.name)
        if match is None:
            print(f"skipping {path.name}: no _YYYYMM suffix")
            continue
        year, month = match.groups()
        if months is not None and month not in months:
            continue
        files.append((int(year), int(month), path))

    if not files:
        raise FileNotFoundError(f"No ERA5 GRIBs matching {pattern!r} in {era5_dir}")
    return files


def close_datasets(datasets):
    for dataset in datasets:
        dataset.close()


def check_same_grid(path, latitude, longitude, reference_latitude, reference_longitude):
    if not (
        np.array_equal(latitude, reference_latitude)
        and np.array_equal(longitude, reference_longitude)
    ):
        raise ValueError(
            f"{path.name} has a different lat/lon grid "
            f"({len(latitude)}x{len(longitude)}) from the first month "
            f"({len(reference_latitude)}x{len(reference_longitude)})."
        )


def conform_to_schema(frame, schema):
    """Give every month the first month's columns, in the same order and type.

    A variable missing from one month becomes NaN with a zero count; a variable
    only present in a later month is dropped (the first month defines the file).
    """
    extra = sorted(set(frame.columns) - set(schema.names))
    if extra:
        print(f"  dropping columns absent from the first month: {extra}")

    for name in schema.names:
        if name not in frame.columns:
            frame[name] = 0 if name.endswith("_n_datapoints") else np.nan

    table = pa.Table.from_pandas(frame[schema.names], preserve_index=False)
    return table.cast(schema)


def write_monthly_merged_parquet(output_path, grib_files, clara, limit_hours=None):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reference_latitude = reference_longitude = None
    grid_frame = clara_hourly = longitude_column = None
    writer = None
    schema = None
    n_hours_total = 0

    try:
        for year, month, path in grib_files:
            print(f"opening {path.name}")
            era5_datasets = load_era5(path)
            try:
                # One month is a few hundred MB, and holding it in memory turns the
                # per-hour lookups below into array slicing instead of ~30k separate
                # GRIB message reads.
                era5_datasets = [dataset.load() for dataset in era5_datasets]
                latitude, longitude = get_common_grid(era5_datasets)

                if reference_latitude is None:
                    # The grid is fixed by the first month; CLARA is snapped to it once.
                    reference_latitude, reference_longitude = latitude, longitude
                    grid_frame = make_grid_frame(latitude, longitude)
                    clara_hourly, longitude_column = aggregate_clara_hourly(
                        clara, latitude, longitude
                    )
                else:
                    check_same_grid(path, latitude, longitude, reference_latitude, reference_longitude)

                dataset_indexes, hours, variable_names = build_era5_index(era5_datasets)

                # Forecast-based fields (the avg_* fluxes) can carry valid times
                # outside the requested month; keep each hour in its own file only.
                in_month = [hour for hour in hours if hour.year == year and hour.month == month]
                if len(in_month) != len(hours):
                    print(f"  ignoring {len(hours) - len(in_month)} hours outside {year}-{month:02d}")
                hours = in_month[:limit_hours] if limit_hours is not None else in_month

                for hour_number, hour in enumerate(hours, start=1):
                    frame = make_merged_hour_frame(
                        hour=hour,
                        grid_frame=grid_frame,
                        dataset_indexes=dataset_indexes,
                        variable_names=variable_names,
                        clara_hourly=clara_hourly,
                    )

                    if writer is None:
                        table = pa.Table.from_pandas(frame, preserve_index=False)
                        schema = table.schema
                        writer = pq.ParquetWriter(output_path, schema, compression="zstd")
                    else:
                        table = conform_to_schema(frame, schema)

                    writer.write_table(table)
                    if hour_number == 1 or hour_number % 24 == 0 or hour_number == len(hours):
                        print(
                            f"  wrote hour {hour_number}/{len(hours)}: {hour} "
                            f"({len(frame):,} longitude/latitude rows)"
                        )

                n_hours_total += len(hours)
            finally:
                close_datasets(era5_datasets)
    finally:
        if writer is not None:
            writer.close()

    return n_hours_total, len(grid_frame), clara_hourly, longitude_column


def main():
    args = parse_args()
    months = None if args.months is None else [f"{int(m):02d}" for m in args.months]
    grib_files = monthly_grib_files(args.era5_dir, args.era5_pattern, months)
    print(f"ERA5 months: {', '.join(f'{y}-{m:02d}' for y, m, _ in grib_files)}")

    clara = load_clara(args.clara_path)
    plot_clara_timeseries(clara, args.plot_output)

    n_hours, n_grid_cells, clara_hourly, longitude_column = write_monthly_merged_parquet(
        output_path=args.merged_output,
        grib_files=grib_files,
        clara=clara,
        limit_hours=args.limit_hours,
    )

    args.clara_hourly_output.parent.mkdir(parents=True, exist_ok=True)
    clara_hourly.to_pickle(args.clara_hourly_output)

    print(f"Used CLARA longitude column: {longitude_column}")
    print(f"Saved CLARA hourly means to {args.clara_hourly_output}")
    print(f"Saved CLARA raw/hourly plot to {args.plot_output}")
    print(f"Saved merged hourly ERA5-left dataset to {args.merged_output}")
    print(f"Merged shape: {n_hours:,} hours x {n_grid_cells:,} longitude/latitude cells")
    print(f"CLARA hourly matches: {len(clara_hourly):,} hour/longitude/latitude cells")


if __name__ == "__main__":
    main()
