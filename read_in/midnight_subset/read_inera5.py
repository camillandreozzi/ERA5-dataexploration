"""Inventory of the downloaded midnight ERA5 files.

Each file covers one UTC hour over a handful of days in its own small box, so
unlike the other subsets there is no common grid to check. What matters instead
is that every request arrived, that the files agree on the variable set, and
that each file really contains the hours its CLARA observations need.
"""

import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "paths.py").exists():
        sys.path.insert(0, str(_p))
        break
from paths import subset_data_path

import pandas as pd

from read_in.midnight_subset.midnight import HOUR_COLUMN, REQUEST_COLUMN, SUBSET, plan_requests
from read_in.temporal_subset.data_preprocessing import get_common_grid, load_era5


CLARA_PATH = subset_data_path("CLARA_matched.pkl", subset=SUBSET)
ERA5_DIR = subset_data_path(subset=SUBSET)


def era5_hours(datasets):
    hours = set()
    for dataset in datasets:
        coordinate = dataset["valid_time"] if "valid_time" in dataset.coords else dataset["time"]
        hours.update(pd.to_datetime(coordinate.values.ravel()).floor("h"))
    return hours


def main():
    observations, requests = plan_requests(pd.read_pickle(CLARA_PATH))
    needed_hours = observations.groupby(REQUEST_COLUMN)[HOUR_COLUMN].apply(set)

    missing_files = []
    reference_variables = None
    n_checked = 0

    for row in requests.itertuples(index=False):
        path = ERA5_DIR / row.filename
        if not path.exists():
            missing_files.append(row.filename)
            continue

        datasets = load_era5(path)
        try:
            variables = sorted(name for dataset in datasets for name in dataset.data_vars)
            latitude, longitude = get_common_grid(datasets)
            hours = era5_hours(datasets)
        finally:
            for dataset in datasets:
                dataset.close()

        n_checked += 1
        missing_hours = sorted(needed_hours[row.request_id] - hours)

        print(
            f"{row.filename}: {len(variables)} variables, "
            f"grid {len(latitude)}x{len(longitude)}, {len(hours)} valid hours"
        )
        if missing_hours:
            print(f"  WARNING: no ERA5 data for {len(missing_hours)} CLARA hour(s): {missing_hours[:3]}")

        if reference_variables is None:
            reference_variables = variables
            print(f"  variables: {', '.join(variables)}")
        elif variables != reference_variables:
            missing = sorted(set(reference_variables) - set(variables))
            extra = sorted(set(variables) - set(reference_variables))
            print(f"  WARNING: variable set differs; missing {missing}, extra {extra}")

    print(f"\nchecked {n_checked}/{len(requests)} requested files")
    if missing_files:
        print(f"{len(missing_files)} not downloaded yet, e.g. {missing_files[:3]}")
        print("Run read_in/midnight_subset/data_fetch.py again to resume.")


if __name__ == "__main__":
    main()
