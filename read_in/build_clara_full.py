"""Build data/CLARA_full.parquet from the raw CLARA delivery.

    python3 read_in/build_clara_full.py                    # needs the complete CSV set
    python3 read_in/build_clara_full.py --no-geolocation   # OLR only, no footprints
    python3 read_in/build_clara_full.py --start 2020-01-01 --end 2021-01-01

The OLR and the geolocation arrive from different routines, so the usable record
is their intersection and is not known until both are on disk. This script works
it out and writes it down twice: as ``data/clara_coverage.csv``, one row per day,
and as key-value metadata on the parquet. Everything downstream reads the span
from there rather than hardcoding a year.

Joining the two halves is done a month at a time. Loading every geolocation CSV
at once would be the simpler code, but at a one-second cadence the full record
would be over a hundred million rows; a month plus a day of overlap on each side
keeps the footprint bounded while still letting a sample just after midnight
interpolate against the previous day's track.
"""

import argparse
import hashlib
import sys
from pathlib import Path

for _p in Path(__file__).resolve().parents:
    if (_p / "paths.py").exists():
        sys.path.insert(0, str(_p))
        break
from paths import results_path

import numpy as np
import pandas as pd

from read_in.clara_source import (
    COVERAGE_PATH,
    FULL_PATH,
    GEOLOCATION_DIR,
    GEO_GAP_COLUMN,
    LATITUDE_COLUMN,
    LONGITUDE_COLUMN,
    OLR_DIR,
    RADIANCE_COLUMN,
    SCHEMA_COLUMNS,
    STATUS_DEAD_OLR,
    STATUS_GAP_TOO_LARGE,
    STATUS_NO_EARTH_VIEW,
    STATUS_NO_GEOLOCATION,
    STATUS_NO_OLR,
    STATUS_OK,
    TIME_COLUMN,
    VIEW_ANGLE_COLUMN,
    add_longitude_and_local_time,
    geolocation_path,
    interpolate_geolocation,
    load_geolocation,
    load_olr,
)

# Same bounds as read_in/temporal_subset/data_preprocessing.py. Reported here so
# the build prints how much of the record is an Earth view, but NOT applied:
# the radiance filter stays at the merge step, where its rejection report is read.
RADIANCE_MIN = 0.0
RADIANCE_MAX = 500.0

DEFAULT_RETENTION_PATH = results_path("exploratory/clara_full/build_retention.csv")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--olr-dir", type=Path, default=OLR_DIR)
    parser.add_argument("--geolocation-dir", type=Path, default=GEOLOCATION_DIR)
    parser.add_argument("--output", type=Path, default=FULL_PATH)
    parser.add_argument("--coverage", type=Path, default=COVERAGE_PATH)
    parser.add_argument("--retention", type=Path, default=DEFAULT_RETENTION_PATH)
    parser.add_argument("--start", default=None, help="first timestamp to keep")
    parser.add_argument("--end", default=None, help="first timestamp to drop")
    parser.add_argument(
        "--no-geolocation", action="store_true",
        help="build from the OLR alone; footprints are NaN and the subsets refuse it",
    )
    parser.add_argument(
        "--allow-missing-days", action="store_true",
        help="tolerate days with no CSV, for inspecting an incomplete delivery; "
             "the result is marked incomplete and the subsets refuse it",
    )
    parser.add_argument(
        "--max-gap-seconds", type=float, default=90.0,
        help="furthest a footprint may be interpolated, in seconds "
             "(default 90, three OLR samples); pass 0 to use 3x the CSV cadence",
    )
    parser.add_argument("--force", action="store_true", help="overwrite the output")
    return parser.parse_args(argv)


def _month_periods(times):
    return pd.PeriodIndex(pd.DatetimeIndex(times).to_period("M")).unique().sort_values()


def geolocate(olr, geolocation_dir, max_gap_seconds, allow_missing, verbose=True):
    """Interpolate footprints onto every OLR timestamp, a month at a time."""
    located = []
    missing_days = []

    for period in _month_periods(olr[TIME_COLUMN]):
        chunk = olr[olr[TIME_COLUMN].dt.to_period("M") == period]
        if chunk.empty:
            continue

        days = pd.DatetimeIndex(chunk[TIME_COLUMN].dt.normalize().unique())
        # One day of overlap each side, so a sample minutes after midnight can
        # still be bracketed by the previous day's track.
        padded = days.union(days - pd.Timedelta("1D")).union(days + pd.Timedelta("1D"))
        available = pd.DatetimeIndex([d for d in padded if geolocation_path(d, geolocation_dir)])

        missing_days.extend(d for d in days if d not in available)

        if len(available) < 1:
            if not allow_missing:
                raise FileNotFoundError(
                    f"No geolocation CSV for any day of {period}. The delivery is "
                    f"incomplete; wait for the rest, or pass --allow-missing-days."
                )
            blank = chunk.copy()
            blank[LATITUDE_COLUMN] = np.nan
            blank[LONGITUDE_COLUMN] = np.nan
            blank[VIEW_ANGLE_COLUMN] = np.float32("nan")
            blank[GEO_GAP_COLUMN] = np.float32("inf")
            located.append(blank)
            continue

        geo, _ = load_geolocation(
            available, geolocation_dir, allow_missing=True, verbose=False
        )
        located.append(interpolate_geolocation(chunk, geo, max_gap_seconds))
        if verbose:
            print(f"  {period}: {len(chunk):,} samples against {len(geo):,} footprints")

    frame = pd.concat(located, ignore_index=True)
    return frame, pd.DatetimeIndex(sorted(set(missing_days)))


def build_coverage(olr, located, dead_days, geolocated):
    """One row per calendar day of the OLR span, with why each day is unusable.

    ``n_valid <= n_earth_view <= n_geolocated <= n_olr`` holds by construction,
    so the manifest can be checked without re-deriving it.
    """
    times = pd.DatetimeIndex(olr[TIME_COLUMN])
    days = pd.date_range(times.min().normalize(), times.max().normalize(), freq="D")

    n_olr = times.normalize().value_counts().reindex(days, fill_value=0)

    coverage = pd.DataFrame({"day": days, "n_olr": n_olr.to_numpy()})

    if geolocated:
        day_of = pd.DatetimeIndex(located[TIME_COLUMN]).normalize()
        bracketed = np.isfinite(located[GEO_GAP_COLUMN].to_numpy())
        valid = located[LATITUDE_COLUMN].notna().to_numpy()
        has_csv = bracketed | valid  # a day with a CSV but no usable bracket

        coverage["n_geolocated"] = (
            pd.Series(has_csv).groupby(day_of).sum().reindex(days, fill_value=0).to_numpy()
        )
        coverage["n_earth_view"] = (
            pd.Series(bracketed).groupby(day_of).sum().reindex(days, fill_value=0).to_numpy()
        )
        coverage["n_valid"] = (
            pd.Series(valid).groupby(day_of).sum().reindex(days, fill_value=0).to_numpy()
        )
    else:
        coverage["n_geolocated"] = 0
        coverage["n_earth_view"] = 0
        coverage["n_valid"] = 0

    status = np.full(len(coverage), STATUS_OK, dtype=object)
    status[coverage["n_valid"].to_numpy() == 0] = STATUS_GAP_TOO_LARGE
    status[coverage["n_earth_view"].to_numpy() == 0] = STATUS_NO_EARTH_VIEW
    status[coverage["n_geolocated"].to_numpy() == 0] = STATUS_NO_GEOLOCATION
    status[coverage["n_olr"].to_numpy() == 0] = STATUS_NO_OLR
    status[coverage["day"].isin(dead_days).to_numpy()] = STATUS_DEAD_OLR
    coverage["status"] = status

    return coverage


def manifest_sha(coverage_path):
    return hashlib.sha256(Path(coverage_path).read_bytes()).hexdigest()[:16]


def write_parquet(frame, path, metadata):
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(frame, preserve_index=False)
    existing = table.schema.metadata or {}
    merged = {**existing, **{k.encode(): str(v).encode() for k, v in metadata.items()}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.replace_schema_metadata(merged), path, compression="zstd")


def main(argv=None):
    args = parse_args(argv)

    if Path(args.output).exists() and not args.force:
        raise SystemExit(
            f"{args.output} already exists. Pass --force to rebuild it."
        )

    diagnostics = {}
    print(f"reading OLR from {args.olr_dir}")
    olr = load_olr(args.olr_dir, args.start, args.end, diagnostics=diagnostics)
    print(f"{len(olr):,} OLR samples "
          f"{olr[TIME_COLUMN].min()} .. {olr[TIME_COLUMN].max()}")

    max_gap = args.max_gap_seconds or None

    if args.no_geolocation:
        located = olr.copy()
        located[LATITUDE_COLUMN] = np.nan
        located[LONGITUDE_COLUMN] = np.nan
        located[VIEW_ANGLE_COLUMN] = np.float32("nan")
        located[GEO_GAP_COLUMN] = np.float32("inf")
        missing_days = pd.DatetimeIndex([])
        final = add_longitude_and_local_time(located)
        print("built without geolocation: footprints are NaN")
    else:
        print(f"interpolating footprints from {args.geolocation_dir}")
        located, missing_days = geolocate(
            olr, args.geolocation_dir, max_gap, args.allow_missing_days
        )
        final = add_longitude_and_local_time(located)
        n_before = len(final)
        final = final[final[LATITUDE_COLUMN].notna()].reset_index(drop=True)
        print(f"dropped {n_before - len(final):,} samples with no usable footprint")

    # Documented schema first, diagnostics after.
    ordered = [c for c in SCHEMA_COLUMNS if c in final.columns]
    final = final[ordered + [c for c in final.columns if c not in ordered]]

    coverage = build_coverage(
        olr, located, diagnostics.get("dead_days", pd.DatetimeIndex([])),
        geolocated=not args.no_geolocation,
    )
    Path(args.coverage).parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(args.coverage, index=False)
    sha = manifest_sha(args.coverage)

    ok = coverage[coverage["status"] == STATUS_OK]
    complete = (not args.no_geolocation) and len(missing_days) == 0

    metadata = {
        "clara_geolocated": str(not args.no_geolocation).lower(),
        "clara_complete": str(complete).lower(),
        "clara_covered_days": len(ok),
        "clara_first_day": str(ok["day"].min().date()) if len(ok) else "",
        "clara_last_day": str(ok["day"].max().date()) if len(ok) else "",
        "clara_manifest_sha": sha,
    }
    write_parquet(final, args.output, metadata)

    in_range = (
        (final[RADIANCE_COLUMN] > RADIANCE_MIN) & (final[RADIANCE_COLUMN] <= RADIANCE_MAX)
    ).sum()
    steps = [
        ("save files read", diagnostics["n_files"]),
        ("placeholder files dropped", len(diagnostics["dead_files"])),
        ("rows read", diagnostics["n_read"]),
        ("duplicate timestamps dropped", diagnostics["n_duplicates"]),
        ("rows after deduplication", len(olr)),
    ]
    if not args.no_geolocation:
        steps += [
            ("rows with a usable footprint", len(final)),
            ("days covered", len(ok)),
            ("days without a geolocation CSV",
             int((coverage["status"] == STATUS_NO_GEOLOCATION).sum())),
        ]
    steps.append((f"rows in ({RADIANCE_MIN:g}, {RADIANCE_MAX:g}]", int(in_range)))
    retention = pd.DataFrame(steps, columns=["step", "n"])
    Path(args.retention).parent.mkdir(parents=True, exist_ok=True)
    retention.to_csv(args.retention, index=False)

    print()
    print(retention.to_string(index=False))
    print()
    print(f"coverage  -> {args.coverage}  (sha {sha})")
    print(f"retention -> {args.retention}")
    print(f"record    -> {args.output}")
    if not complete:
        print("NOTE: marked incomplete; the subsets will refuse this build.")
    return 0


if __name__ == "__main__":
    # A delivery that has not arrived yet is an expected state, not a crash.
    try:
        raise SystemExit(main())
    except FileNotFoundError as error:
        raise SystemExit(f"\n{error}")
