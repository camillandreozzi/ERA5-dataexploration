"""Read the raw CLARA delivery into the frame the rest of the pipeline expects.

This module replaces ``data/CLARA.pkl``, whose provenance was undocumented and
which arrived already filtered and already geolocated by someone else's routine.
The raw delivery comes in two halves that have to be joined here:

``data/CLARA_OLR/``
    Monthly IDL save files ``CLARA_OLR_{YYYY}_{M}.save`` (month not zero-padded)
    holding an ``olr_all`` structure with ``JDAY`` (Julian Day, UTC) and ``OLR``
    (W/m2/sr), sampled every 30 s. These carry no geolocation at all.

``data/CLARA_geolocation/``
    Daily CSVs ``CLARA_lat_lon_alt-{yyyy}-{m}-{d}_V50_1_4.csv`` from a different
    routine on the satellite side. Their timestamps are not synchronised with the
    OLR ones, so the footprint has to be interpolated onto the OLR timestamps.

The usable record is the *intersection* of the two: an OLR sample with no
footprint cannot be matched to an ERA5 grid cell, so it contributes nothing. That
intersection is a property of what has been delivered, not a constant, so it is
computed at build time and written to ``data/clara_coverage.csv``. Nothing in
this repo should hardcode a year or an end date; read the manifest instead.

Output columns keep the names the old pickle used, so nothing downstream moves:

    TimeJD                        datetime64[ns], UTC-naive
    CLARA_radiance                float64, the raw OLR with no offset applied
    CLARA_fov_latitude            float64, degrees
    CLARA_fov_longitude           float64, degrees in [-180, 180)
    CLARA_fov_longitude_positive  float64, degrees in [0, 360)
    CLARA_local_time              float64, mean solar hours in [0, 24)

The 52 housekeeping columns the old pickle carried are not reproduced; nothing
in this repo ever read them.
"""

import json
import re
import sys
from functools import lru_cache
from pathlib import Path

for _p in Path(__file__).resolve().parents:
    if (_p / "paths.py").exists():
        sys.path.insert(0, str(_p))
        break
from paths import data_path

import numpy as np
import pandas as pd


OLR_DIR = data_path("CLARA_OLR")
GEOLOCATION_DIR = data_path("CLARA_geolocation")
FULL_PATH = data_path("CLARA_full.parquet")
COVERAGE_PATH = data_path("clara_coverage.csv")

TIME_COLUMN = "TimeJD"
RADIANCE_COLUMN = "CLARA_radiance"
LATITUDE_COLUMN = "CLARA_fov_latitude"
LONGITUDE_COLUMN = "CLARA_fov_longitude"
POSITIVE_LONGITUDE_COLUMN = "CLARA_fov_longitude_positive"
LOCAL_TIME_COLUMN = "CLARA_local_time"
VIEW_ANGLE_COLUMN = "CLARA_view_angle"
GEO_GAP_COLUMN = "CLARA_geo_gap_seconds"

SCHEMA_COLUMNS = [
    TIME_COLUMN,
    RADIANCE_COLUMN,
    LATITUDE_COLUMN,
    LONGITUDE_COLUMN,
    POSITIVE_LONGITUDE_COLUMN,
    LOCAL_TIME_COLUMN,
]

OLR_FILE_PATTERN = re.compile(r"^CLARA_OLR_(\d{4})_(\d{1,2})\.save$")

# Julian Day of the Unix epoch, so JDAY - this is days since 1970-01-01.
JULIAN_DAY_UNIX_EPOCH = 2440587.5

# A monthly save file with fewer rows than this and nothing but zeros is a
# placeholder, not data. The 2021-09 .. 2022-05 files are all like this: 25-31
# rows each, every OLR exactly 0.0. Tested by content rather than by a filename
# blocklist, which would go stale the moment a new delivery lands.
DEAD_FILE_MAX_ROWS = 1000

# How far two OLR values sharing a timestamp may differ before the build refuses
# to pick one. See _deduplicate_timestamps.
DUPLICATE_RTOL = 0.01
DUPLICATE_ATOL = 1.0

GEOLOCATION_FILE_TEMPLATE = "CLARA_lat_lon_alt-{year}-{month}-{day}_V50_1_4.csv"
GEOLOCATION_FILE_GLOB = "CLARA_lat_lon_alt-{year}-{month}-{day}_*.csv"

# Columns of the geolocation CSV, by position. Columns 2-4 (satellite lat/lon/alt)
# and 7 (clara distance) are deliberately not read.
GEO_TIME_INDEX = 0
GEO_VIEW_ANGLE_INDEX = 1
GEO_LATITUDE_INDEX = 5
GEO_LONGITUDE_INDEX = 6

# Statuses recorded per day in the coverage manifest.
STATUS_OK = "ok"
STATUS_NO_OLR = "no_olr_file"
STATUS_DEAD_OLR = "dead_olr_file"
STATUS_NO_GEOLOCATION = "no_geolocation_csv"
STATUS_NO_EARTH_VIEW = "no_earth_view"
STATUS_GAP_TOO_LARGE = "gap_too_large"


# --------------------------------------------------------------------------
# OLR side
# --------------------------------------------------------------------------


def jday_to_datetime(jday):
    """Julian Day (UTC) to naive datetime64[ns]."""
    return pd.to_datetime(np.asarray(jday, dtype="<f8") - JULIAN_DAY_UNIX_EPOCH, unit="D")


def list_olr_files(directory=OLR_DIR):
    """The ``CLARA_OLR_{year}_{month}.save`` files present, sorted by month."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"No CLARA OLR directory at {directory}. Expected monthly IDL save "
            f"files named CLARA_OLR_YYYY_M.save (month not zero-padded)."
        )

    rows = []
    for path in directory.iterdir():
        match = OLR_FILE_PATTERN.match(path.name)
        if match:
            rows.append({
                "year": int(match.group(1)),
                "month": int(match.group(2)),
                "path": path,
            })
    if not rows:
        raise FileNotFoundError(f"No CLARA_OLR_YYYY_M.save files under {directory}.")

    files = pd.DataFrame(rows).sort_values(["year", "month"]).reset_index(drop=True)
    files["period"] = pd.PeriodIndex.from_fields(
        year=files["year"], month=files["month"], freq="M"
    )
    return files


def read_olr_file(path):
    """One monthly save file as a ``[TimeJD, CLARA_radiance]`` frame.

    The IDL fields come back big-endian (``>f8``). Left that way they make a
    later ``groupby`` raise "Big-endian buffer not supported on little-endian
    compiler", so they are cast to native order here, once, and no code
    downstream has to know about it.
    """
    from scipy.io import readsav

    saved = readsav(str(path), verbose=False)["olr_all"]
    jday = np.asarray(saved["JDAY"][0], dtype="<f8")
    olr = np.asarray(saved["OLR"][0], dtype="<f8")

    return pd.DataFrame({TIME_COLUMN: jday_to_datetime(jday), RADIANCE_COLUMN: olr})


def _is_dead_olr_frame(frame):
    """A placeholder file: too small to be a month, and nothing but zeros."""
    return len(frame) < DEAD_FILE_MAX_ROWS and bool((frame[RADIANCE_COLUMN] == 0).all())


def load_olr(directory=OLR_DIR, start=None, end=None, verbose=True, diagnostics=None):
    """Every real OLR sample on disk, deduplicated and sorted by time.

    ``start``/``end`` clip the result (``end`` exclusive). Months absent from the
    delivery are reported but not an error: the record genuinely has holes.
    Pass a dict as ``diagnostics`` to receive the per-step counts the build
    script turns into its retention table.
    """
    files = list_olr_files(directory)

    expected = pd.period_range(files["period"].min(), files["period"].max(), freq="M")
    absent = expected.difference(pd.PeriodIndex(files["period"]))
    if verbose and len(absent):
        print(f"OLR months absent from the delivery ({len(absent)}): "
              f"{', '.join(str(p) for p in absent)}")

    frames = []
    dead = []
    dead_days = []
    for row in files.itertuples():
        frame = read_olr_file(row.path)
        if _is_dead_olr_frame(frame):
            dead.append((row.path.name, len(frame)))
            dead_days.append(frame[TIME_COLUMN].dt.normalize())
            continue
        frames.append(frame)

    if verbose and dead:
        total = sum(n for _, n in dead)
        print(f"dropped {len(dead)} placeholder save files ({total} all-zero rows): "
              f"{', '.join(name for name, _ in dead)}")

    if not frames:
        raise ValueError(f"Every save file under {directory} is a placeholder.")

    olr = pd.concat(frames, ignore_index=True).sort_values(TIME_COLUMN, kind="stable")

    n_read = len(olr)
    olr = _deduplicate_timestamps(olr)
    n_duplicates = n_read - len(olr)
    if verbose and n_duplicates:
        print(f"dropped {n_duplicates:,} duplicate timestamps "
              f"(monthly files overlap by a few samples at each boundary)")

    if start is not None:
        olr = olr[olr[TIME_COLUMN] >= pd.Timestamp(start)]
    if end is not None:
        olr = olr[olr[TIME_COLUMN] < pd.Timestamp(end)]

    if diagnostics is not None:
        diagnostics.update({
            "n_files": len(files),
            "absent_months": [str(p) for p in absent],
            "dead_files": dead,
            "dead_days": (
                pd.DatetimeIndex(pd.concat(dead_days).unique()) if dead_days
                else pd.DatetimeIndex([])
            ),
            "n_read": n_read,
            "n_duplicates": n_duplicates,
        })

    return olr.reset_index(drop=True)


def _deduplicate_timestamps(olr, rtol=DUPLICATE_RTOL, atol=DUPLICATE_ATOL, verbose=True):
    """Collapse repeated timestamps, refusing to guess when they really disagree.

    Within a file ~110 rows are exact ``(time, radiance)`` repeats, and each
    monthly file spills a few samples into the next month's file; those agree
    exactly, so dropping the later one is lossless.

    One pair in the current delivery does not agree exactly: 2024-09-20 00:01:28
    carries 381.562 and 380.909, two adjacent samples that collided on a
    timestamp. A 0.17% spread is far below anything the hourly grid-cell mean
    downstream can resolve, so refusing to build over it would be theatre. It is
    reported rather than hidden, and a spread wide enough to change which side of
    the Earth-view filter a sample falls on still raises.
    """
    duplicated = olr.duplicated(subset=[TIME_COLUMN], keep=False)
    if duplicated.any():
        grouped = olr.loc[duplicated].groupby(TIME_COLUMN)[RADIANCE_COLUMN]
        spread = grouped.max() - grouped.min()
        allowed = np.maximum(atol, rtol * grouped.mean().abs())

        conflicting = spread[spread > allowed]
        if len(conflicting):
            first = conflicting.index[0]
            raise ValueError(
                f"{len(conflicting)} timestamps carry OLR values that differ by "
                f"more than {rtol:.0%} (or {atol:g} W/m2/sr), first at {first} "
                f"with a spread of {conflicting.iloc[0]:g}. That is too wide to "
                f"resolve by keeping one arbitrarily; check it upstream."
            )

        inexact = spread[spread > 0]
        if verbose and len(inexact):
            print(f"{len(inexact)} duplicate timestamps disagree slightly "
                  f"(max spread {inexact.max():g} W/m2/sr); keeping the first of each")

    return olr.drop_duplicates(subset=[TIME_COLUMN], keep="first")


# --------------------------------------------------------------------------
# Geolocation side
# --------------------------------------------------------------------------


def parse_vector_column(series, source=""):
    """A column of ``"(x,y,z)"`` strings as a float frame with x/y/z columns.

    Vectorised on purpose: the record is ~3 M rows and a per-row
    ``ast.literal_eval`` would dominate the build.
    """
    text = series.astype("string").str.strip().str.strip("()")

    # Check every row, not just the frame's width. str.split(expand=True) sizes
    # itself to the widest row and pads the rest with None, so a short row would
    # otherwise slip through as a silent NaN component.
    bad = text[text.str.count(",") != 2]
    if len(bad):
        where = f" in {source}" if source else ""
        raise ValueError(
            f"Expected '(x,y,z)' triples{where}; {len(bad)} rows are not, "
            f"first at position {bad.index[0]}: {bad.iloc[0]!r}"
        )

    parts = text.str.split(",", expand=True)
    return parts.astype("float64").set_axis(["x", "y", "z"], axis=1)


def _looks_like_scalar(parts):
    """True when only the first component is used, i.e. a scalar in parentheses."""
    return bool((parts["y"] == 0).all() and (parts["z"] == 0).all())


def _vector_to_degrees(latitude_parts, longitude_parts, source=""):
    """Footprint latitude and longitude in degrees, whichever form the CSV uses.

    The delivery note describes columns 6 and 7 as "clara latitude" and "clara
    longitude" but stores both as ``"(x,y,z)"``, which leaves two readings: a
    scalar wrapped in parentheses, or a Cartesian direction. Both are handled,
    and anything else raises rather than being guessed at, because a silently
    wrong footprint would only show up much later as a bad ERA5 match.
    """
    where = f" in {source}" if source else ""

    if _looks_like_scalar(latitude_parts) and _looks_like_scalar(longitude_parts):
        return latitude_parts["x"].to_numpy(), longitude_parts["x"].to_numpy()

    norm = np.sqrt(
        latitude_parts["x"] ** 2 + latitude_parts["y"] ** 2 + latitude_parts["z"] ** 2
    ).to_numpy()
    if np.allclose(norm[np.isfinite(norm) & (norm > 0)], 1.0, atol=1e-3):
        with np.errstate(invalid="ignore", divide="ignore"):
            latitude = np.degrees(np.arcsin(np.clip(latitude_parts["z"] / norm, -1.0, 1.0)))
            longitude = np.degrees(np.arctan2(latitude_parts["y"], latitude_parts["x"]))
        return np.asarray(latitude), np.asarray(longitude)

    raise ValueError(
        f"Cannot tell how the clara latitude/longitude columns are encoded{where}. "
        f"They are neither scalars in parentheses (components 2 and 3 all zero) nor "
        f"unit vectors (norm 1). First latitude triple: "
        f"{tuple(latitude_parts.iloc[0])}, first longitude triple: "
        f"{tuple(longitude_parts.iloc[0])}. Pin this down against the delivered "
        f"file before building anything."
    )


def geolocation_path(day, directory=GEOLOCATION_DIR):
    """The CSV for one day, tolerating a version suffix other than V50_1_4."""
    day = pd.Timestamp(day)
    directory = Path(directory)
    fields = {"year": day.year, "month": day.month, "day": day.day}

    exact = directory / GEOLOCATION_FILE_TEMPLATE.format(**fields)
    if exact.exists():
        return exact

    matches = sorted(directory.glob(GEOLOCATION_FILE_GLOB.format(**fields)))
    if len(matches) > 1:
        raise ValueError(
            f"{len(matches)} geolocation files for {day.date()}: "
            f"{', '.join(m.name for m in matches)}. Expected exactly one."
        )
    return matches[0] if matches else None


@lru_cache(maxsize=8)
def _read_geolocation_file(path):
    """Cached per-file read, so a day boundary inside a chunk costs one read."""
    frame = pd.read_csv(
        path,
        usecols=[
            GEO_TIME_INDEX,
            GEO_VIEW_ANGLE_INDEX,
            GEO_LATITUDE_INDEX,
            GEO_LONGITUDE_INDEX,
        ],
        dtype="string",
    )
    frame.columns = ["time", "view_angle", "latitude_raw", "longitude_raw"]

    source = Path(path).name
    latitude_parts = parse_vector_column(frame["latitude_raw"], source)
    longitude_parts = parse_vector_column(frame["longitude_raw"], source)

    # A zero vector means the instrument was not pointed at Earth. This is the
    # main quality gate of the new pipeline: it rejects calibration views on a
    # geometric fact rather than on a threshold over the radiance.
    earth_view = ~(
        (latitude_parts == 0).all(axis=1).to_numpy()
        | (longitude_parts == 0).all(axis=1).to_numpy()
    )

    latitude, longitude = _vector_to_degrees(latitude_parts, longitude_parts, source)

    geo = pd.DataFrame({
        "time": pd.to_datetime(frame["time"]),
        "view_angle": pd.to_numeric(frame["view_angle"], errors="coerce"),
        "latitude": latitude,
        "longitude": longitude,
        "earth_view": earth_view,
    })
    geo = geo.loc[geo["earth_view"] & geo["time"].notna()].drop(columns="earth_view")
    geo = geo.sort_values("time", kind="stable").drop_duplicates("time", keep="first")
    return geo.reset_index(drop=True)


def read_geolocation_day(day, directory=GEOLOCATION_DIR):
    """Earth-viewing footprints for one day, or None when the CSV is absent."""
    path = geolocation_path(day, directory)
    return None if path is None else _read_geolocation_file(path)


def load_geolocation(days, directory=GEOLOCATION_DIR, allow_missing=False, verbose=True):
    """Footprints for ``days``, concatenated and sorted.

    Returns ``(frame, missing_days)``. With ``allow_missing=False`` a gap raises,
    listing every missing day rather than only the first: an incomplete delivery
    needs a shopping list, not one date at a time.
    """
    directory = Path(directory)
    days = pd.DatetimeIndex(days).normalize().unique().sort_values()

    if not directory.is_dir() or not any(directory.glob("CLARA_lat_lon_alt-*.csv")):
        raise FileNotFoundError(
            f"No CLARA geolocation CSVs under {directory}. Expected files named "
            f"CLARA_lat_lon_alt-YYYY-M-D_V50_1_4.csv (month and day not "
            f"zero-padded). To work with the OLR alone for now, run "
            f"read_in/build_clara_full.py --no-geolocation."
        )

    frames = []
    missing = []
    for day in days:
        frame = read_geolocation_day(day, directory)
        if frame is None:
            missing.append(day)
        else:
            frames.append(frame)

    if missing and not allow_missing:
        shown = ", ".join(str(d.date()) for d in missing[:5])
        tail = ", ".join(str(d.date()) for d in missing[-5:])
        raise FileNotFoundError(
            f"{len(missing)} of {len(days)} days have no geolocation CSV under "
            f"{directory}. First: {shown}. Last: {tail}. The delivery is "
            f"incomplete; wait for the rest, or pass --allow-missing-days to "
            f"inspect what is there (that build is marked incomplete and the "
            f"subsets will refuse it)."
        )

    if verbose and missing:
        print(f"no geolocation CSV for {len(missing)} of {len(days)} days; "
              f"their OLR samples will be dropped")

    if not frames:
        raise ValueError(f"None of the {len(days)} requested days has a geolocation CSV.")

    geo = pd.concat(frames, ignore_index=True)
    geo = geo.sort_values("time", kind="stable").drop_duplicates("time", keep="first")
    return geo.reset_index(drop=True), pd.DatetimeIndex(missing)


# --------------------------------------------------------------------------
# Joining the two
# --------------------------------------------------------------------------


def _epoch_seconds(times):
    return pd.DatetimeIndex(times).astype("int64").to_numpy() / 1e9


def interpolate_geolocation(olr, geo, max_gap_seconds=None):
    """Footprint at each OLR timestamp, by interpolating the position vector.

    The angles are not interpolated directly. Linear interpolation of longitude
    averages -179.9 and 179.9 to 0.0, which moves a Pacific footprint to West
    Africa; the track crosses the antimeridian about fifteen times a day, and
    those rows would go on to produce nonsense request boxes in plan_requests.
    Interpolating the unit vector and converting back is correct across the
    antimeridian and at the poles, where the track reaches beyond 80 degrees.

    ``np.unwrap`` would be cheaper but accumulates an arbitrary multiple of 2*pi
    across the several hundred multi-minute gaps per month, so it would need the
    track segmented first.

    Rows further than ``max_gap_seconds`` from the nearer bracketing sample, or
    outside the CSV's span, come back NaN. The default is three times the CSV's
    own median cadence (floor 60 s), measured rather than assumed.
    """
    t_olr = _epoch_seconds(olr[TIME_COLUMN])
    t_geo = _epoch_seconds(geo["time"])

    if len(t_geo) < 2:
        raise ValueError("Need at least two geolocation samples to interpolate.")

    if max_gap_seconds is None:
        max_gap_seconds = max(3.0 * float(np.median(np.diff(t_geo))), 60.0)

    latitude = np.radians(geo["latitude"].to_numpy())
    longitude = np.radians(geo["longitude"].to_numpy())
    ux = np.cos(latitude) * np.cos(longitude)
    uy = np.cos(latitude) * np.sin(longitude)
    uz = np.sin(latitude)

    ix = np.interp(t_olr, t_geo, ux)
    iy = np.interp(t_olr, t_geo, uy)
    iz = np.interp(t_olr, t_geo, uz)
    norm = np.sqrt(ix ** 2 + iy ** 2 + iz ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        ix, iy, iz = ix / norm, iy / norm, iz / norm

    interpolated_latitude = np.degrees(np.arcsin(np.clip(iz, -1.0, 1.0)))
    interpolated_longitude = np.degrees(np.arctan2(iy, ix))

    gap = _distance_to_nearest_sample(t_olr, t_geo)
    view_angle = np.interp(t_olr, t_geo, geo["view_angle"].to_numpy())

    unusable = ~np.isfinite(gap) | (gap > max_gap_seconds) | ~np.isfinite(norm)
    interpolated_latitude[unusable] = np.nan
    interpolated_longitude[unusable] = np.nan
    view_angle[unusable] = np.nan

    result = olr.copy()
    result[LATITUDE_COLUMN] = interpolated_latitude
    result[LONGITUDE_COLUMN] = interpolated_longitude
    result[VIEW_ANGLE_COLUMN] = view_angle.astype("float32")
    result[GEO_GAP_COLUMN] = gap.astype("float32")
    return result


def _distance_to_nearest_sample(t_target, t_source):
    """Seconds from each target time to the nearer bracketing source sample.

    Targets outside the source span get ``inf``: extrapolating a satellite
    footprint past the end of the track it was measured on is not interpolation.
    """
    index = np.searchsorted(t_source, t_target)
    inside = (index > 0) & (index < len(t_source))

    gap = np.full(t_target.shape, np.inf)
    left = t_target[inside] - t_source[index[inside] - 1]
    right = t_source[index[inside]] - t_target[inside]
    gap[inside] = np.minimum(left, right)

    # An exact hit on the first or last sample is a match, not an extrapolation.
    gap[np.isin(t_target, t_source[[0, -1]])] = 0.0
    return gap


def add_longitude_and_local_time(frame):
    """Add the 0-360 longitude and the local mean solar time.

    Local time is derived here rather than inherited. The old pickle's
    ``CLARA_local_time`` behaved like the satellite subpoint's clock time and sat
    about 1.5 h ahead of the footprint's own solar time on the rows the midnight
    rule selected; this is the footprint's, which is what "the ERA5 cell was at
    local midnight" actually means.

    It is *mean* solar time: no equation-of-time correction, so it differs from
    apparent solar time by up to 16 minutes, a third of the midnight window's
    half-width.
    """
    result = frame.copy()

    longitude = ((result[LONGITUDE_COLUMN] + 180.0) % 360.0) - 180.0
    result[LONGITUDE_COLUMN] = longitude
    result[POSITIVE_LONGITUDE_COLUMN] = longitude % 360.0

    # From the timedelta, not hour + minute/60 + second/3600: these timestamps
    # carry microseconds and the arithmetic form would silently drop them.
    utc_hours = (
        result[TIME_COLUMN] - result[TIME_COLUMN].dt.normalize()
    ) / pd.Timedelta("1h")
    result[LOCAL_TIME_COLUMN] = (utc_hours + longitude / 15.0) % 24.0

    finite = result[LOCAL_TIME_COLUMN].notna()
    if finite.any():
        values = result.loc[finite, LOCAL_TIME_COLUMN]
        if not ((values >= 0.0) & (values < 24.0)).all():
            raise ValueError("Derived CLARA_local_time fell outside [0, 24).")

    return result


# --------------------------------------------------------------------------
# Coverage manifest
# --------------------------------------------------------------------------


def load_coverage(path=COVERAGE_PATH):
    """The per-day coverage manifest written by build_clara_full.py."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No coverage manifest at {path}. Run read_in/build_clara_full.py."
        )
    coverage = pd.read_csv(path, parse_dates=["day"])
    return coverage


def covered_days(coverage=None):
    """The days that have both OLR and geolocation, as a DatetimeIndex."""
    coverage = load_coverage() if coverage is None else coverage
    return pd.DatetimeIndex(coverage.loc[coverage["status"] == STATUS_OK, "day"])


def covered_span(coverage=None):
    """``(first_day, last_day)`` of the usable record."""
    days = covered_days(coverage)
    if not len(days):
        raise ValueError("The coverage manifest has no usable days.")
    return days.min(), days.max()


def format_span(span=None):
    """``2020-01 to 2025-10``, for figure titles that must not outrun the data."""
    first, last = covered_span() if span is None else span
    return f"{first:%Y-%m} to {last:%Y-%m}"


def write_subset_span(path, frame, manifest_sha):
    """Record the span a subset actually used, next to its CLARA_matched.pkl.

    Downstream stages compare this against the manifest they see. That is what
    stops a CLARA_matched.pkl left over from an earlier, narrower delivery from
    silently pinning a later run to it.
    """
    times = pd.to_datetime(frame[TIME_COLUMN])
    payload = {
        "first_day": str(times.min().date()),
        "last_day": str(times.max().date()),
        "n_observations": int(len(frame)),
        "clara_manifest_sha": manifest_sha,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


# --------------------------------------------------------------------------
# Reading the built file
# --------------------------------------------------------------------------


def _parquet_metadata(path):
    import pyarrow.parquet as pq

    metadata = pq.read_schema(path).metadata or {}
    return {
        key.decode(): value.decode()
        for key, value in metadata.items()
        if not key.decode().startswith("pandas")
    }


def load_clara_full(path=FULL_PATH, require_geolocation=True, require_complete=True,
                    columns=None):
    """The built CLARA record.

    ``require_geolocation`` is on by default because every consumer except
    ``exploratory/explore_clara_full.py`` needs a footprint to match ERA5 with.
    ``require_complete`` refuses a build made from a partial delivery, so an
    incomplete set cannot quietly become the basis of a subset.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No built CLARA record at {path}. Run read_in/build_clara_full.py."
        )

    metadata = _parquet_metadata(path)

    if require_geolocation and metadata.get("clara_geolocated") != "true":
        raise ValueError(
            f"{path} was built without geolocation (--no-geolocation), so it has "
            f"no CLARA_fov_latitude/longitude and cannot be matched to ERA5. "
            f"Rebuild once the CSVs are under {GEOLOCATION_DIR}."
        )

    if require_complete and metadata.get("clara_complete") != "true":
        raise ValueError(
            f"{path} was built from an incomplete delivery "
            f"(--allow-missing-days), covering {metadata.get('clara_covered_days')} "
            f"days from {metadata.get('clara_first_day')} to "
            f"{metadata.get('clara_last_day')}. Rebuild from the complete set "
            f"before using it for a subset."
        )

    frame = pd.read_parquet(path, columns=columns)
    frame.attrs.update(metadata)
    return frame
