"""Download one ERA5 file per (month, UTC hour) that has midnight CLARA data.

The other two subsets ask for a fixed box over a fixed period. Here only the
CLARA-matched rows are kept, so downloading 24 global hours a day would fetch
hundreds of GB to use about a thousand of them. Instead each request covers a
single hour and the bounding box of the observations made in it, which is about
100 requests and well under a GB for 2020.

Requests whose file already exists are skipped, so an interrupted run resumes
where it stopped.
"""

import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "paths.py").exists():
        sys.path.insert(0, str(_p))
        break
from paths import subset_data_path

import argparse

import cdsapi
import pandas as pd

from read_in.midnight_subset.midnight import SUBSET, VARIABLES, plan_requests


DATASET = "reanalysis-era5-single-levels"

DEFAULT_CLARA_PATH = subset_data_path("CLARA_matched.pkl", subset=SUBSET)
DEFAULT_OUT_DIR = subset_data_path(subset=SUBSET)
DEFAULT_REQUEST_TABLE = subset_data_path("era5_requests.csv", subset=SUBSET)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch the ERA5 hours covering the midnight CLARA observations."
    )
    parser.add_argument("--clara-path", type=Path, default=DEFAULT_CLARA_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--request-table", type=Path, default=DEFAULT_REQUEST_TABLE)
    parser.add_argument(
        "--limit-requests",
        type=int,
        default=None,
        help="Only submit the first N missing requests. Useful for a smoke test.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the request table and print what would be fetched.",
    )
    return parser.parse_args()


def build_request(row):
    return {
        "product_type": ["reanalysis"],
        "year": [f"{int(row.year)}"],
        "month": [f"{int(row.month):02d}"],
        "day": [f"{int(day):02d}" for day in row.days],
        "time": [f"{int(row.hour_of_day):02d}:00"],
        "variable": VARIABLES,
        "area": [row.north, row.west, row.south, row.east],
        "data_format": "grib",
        "download_format": "unarchived",
    }


def main():
    args = parse_args()

    clara = pd.read_pickle(args.clara_path)
    _, requests = plan_requests(clara)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.request_table.parent.mkdir(parents=True, exist_ok=True)
    requests.to_csv(args.request_table, index=False)
    print(f"{len(requests)} requests ({int(requests['n_items'].sum()):,} CDS items)")
    print(f"request table -> {args.request_table}")

    pending = [
        row for row in requests.itertuples(index=False)
        if not (args.out_dir / row.filename).exists()
    ]
    print(f"{len(requests) - len(pending)} already downloaded, {len(pending)} to fetch")

    if args.limit_requests is not None:
        pending = pending[: args.limit_requests]
        print(f"limited to the first {len(pending)} request(s)")

    client = None if args.dry_run else cdsapi.Client()

    for number, row in enumerate(pending, start=1):
        target = args.out_dir / row.filename
        print(
            f"[{number}/{len(pending)}] {row.filename}: "
            f"{int(row.year)}-{int(row.month):02d} {int(row.hour_of_day):02d}:00 UTC, "
            f"{len(row.days)} day(s), area {[row.north, row.west, row.south, row.east]}, "
            f"{row.n_observations} CLARA obs"
        )
        if args.dry_run:
            continue

        client.retrieve(DATASET, build_request(row)).download(str(target))

    print("done" if not args.dry_run else "dry run, nothing downloaded")


if __name__ == "__main__":
    main()
