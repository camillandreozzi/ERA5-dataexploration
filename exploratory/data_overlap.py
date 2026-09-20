import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "paths.py").exists():
        sys.path.insert(0, str(_p))
        break
from paths import SUBSET, subset_data_path, subset_results_path, show_or_save

import warnings

import cfgrib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


CLARA_PATH = subset_data_path("CLARA_matched.pkl")
# temporal_subset is one GRIB; spatial_subset is one GRIB per month.
ERA5_PATHS = (
    [subset_data_path("ERA5_matched.grib")]
    if SUBSET == "temporal_subset"
    else sorted(subset_data_path().glob("ERA5_matched_2020*.grib"))
)
OUTPUT_PATH = subset_results_path("exploratory/clara_era5_overlap.png")


def load_clara_matched(path):
    clara_matched = pd.read_pickle(path)
    clara_matched["TimeJD"] = pd.to_datetime(clara_matched["TimeJD"])
    return clara_matched


def load_era5_matched(paths):
    """Open every GRIB lazily; only coordinates are read below."""
    if not paths:
        raise FileNotFoundError(f"No ERA5 GRIBs found in {subset_data_path()}")
    datasets = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        for path in paths:
            datasets.extend(cfgrib.open_datasets(path))
    return datasets


def get_era5_times(era5_matched):
    times = []

    for ds in era5_matched:
        time_coord = "valid_time" if "valid_time" in ds.coords else "time"
        values = ds[time_coord].values.ravel()
        times.extend(pd.to_datetime(values).floor("min"))

    return pd.Series(times).dropna().drop_duplicates().sort_values().reset_index(drop=True)


def get_era5_extent(era5_matched):
    latitudes = []
    longitudes = []

    for ds in era5_matched:
        if "latitude" in ds.coords:
            latitudes.extend([float(ds.latitude.min()), float(ds.latitude.max())])
        if "longitude" in ds.coords:
            longitudes.extend([float(ds.longitude.min()), float(ds.longitude.max())])

    return {
        "lat_min": min(latitudes),
        "lat_max": max(latitudes),
        "lon_min": min(longitudes),
        "lon_max": max(longitudes),
    }


def clara_longitude_for_era5(clara_matched, era5_extent):
    if era5_extent["lon_min"] >= 0 and "CLARA_fov_longitude_positive" in clara_matched:
        return clara_matched["CLARA_fov_longitude_positive"], "CLARA_fov_longitude_positive"

    return clara_matched["CLARA_fov_longitude"], "CLARA_fov_longitude"


def format_timestamp(timestamp):
    return pd.Timestamp(timestamp).strftime("%Y-%m-%d %H:%M")


def plot_overlap(clara_matched, era5_matched):
    clara_columns = ["TimeJD", "CLARA_fov_latitude", "CLARA_fov_longitude"]
    if "CLARA_fov_longitude_positive" in clara_matched.columns:
        clara_columns.append("CLARA_fov_longitude_positive")

    clara_plot = clara_matched[clara_columns].dropna(
        subset=["TimeJD", "CLARA_fov_latitude", "CLARA_fov_longitude"]
    )
    clara_plot = clara_plot.sort_values("TimeJD")
    clara_minutes = clara_plot["TimeJD"].dt.floor("min")

    era5_times = get_era5_times(era5_matched)
    era5_extent = get_era5_extent(era5_matched)
    clara_lon, clara_lon_name = clara_longitude_for_era5(clara_plot, era5_extent)

    clara_time_start = clara_plot["TimeJD"].min()
    clara_time_end = clara_plot["TimeJD"].max()
    era5_time_start = era5_times.min()
    era5_time_end = era5_times.max()
    overlap_start = max(clara_time_start, era5_time_start)
    overlap_end = min(clara_time_end, era5_time_end)
    has_time_overlap = overlap_start <= overlap_end

    if has_time_overlap:
        in_time_window = clara_plot["TimeJD"].between(overlap_start, overlap_end)
    else:
        in_time_window = pd.Series(False, index=clara_plot.index)
    exact_time_match = clara_minutes.isin(set(era5_times))
    in_space_overlap = (
        clara_plot["CLARA_fov_latitude"].between(era5_extent["lat_min"], era5_extent["lat_max"])
        & clara_lon.between(era5_extent["lon_min"], era5_extent["lon_max"])
    )
    in_time_window_and_space = in_time_window & in_space_overlap
    exact_time_and_space = exact_time_match & in_space_overlap

    fig, (time_ax, space_ax) = plt.subplots(
        2,
        1,
        figsize=(12, 10),
        gridspec_kw={"height_ratios": [1, 2]},
        constrained_layout=True,
    )

    time_ax.hlines(1, clara_time_start, clara_time_end, color="#d97706", linewidth=10, alpha=0.65)
    time_ax.scatter(clara_plot["TimeJD"], np.ones(len(clara_plot)), color="#92400e", s=9, alpha=0.5)
    time_ax.scatter(
        clara_plot.loc[exact_time_match, "TimeJD"],
        np.full(int(exact_time_match.sum()), 1.12),
        color="#dc2626",
        marker="x",
        s=40,
        label="Exact ERA5 valid minute",
    )
    time_ax.hlines(0, era5_time_start, era5_time_end, color="#2563eb", linewidth=10, alpha=0.65)
    time_ax.vlines(era5_times, -0.12, 0.12, color="#1e40af", linewidth=0.9, alpha=0.45)

    if has_time_overlap:
        time_ax.axvspan(overlap_start, overlap_end, color="#22c55e", alpha=0.18, label="Temporal overlap")
        time_overlap_label = f"{format_timestamp(overlap_start)} to {format_timestamp(overlap_end)}"
    else:
        time_overlap_label = "none"

    time_ax.set_yticks([0, 1], ["ERA5 valid times", "CLARA samples"])
    time_window_count = int(in_time_window.sum())
    exact_time_count = int(exact_time_match.sum())

    time_ax.set_title("Temporal coverage and exact valid-minute matches")
    time_ax.set_xlabel("Time")
    time_ax.grid(True, axis="x", alpha=0.3)
    time_ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    time_ax.text(
        0.01,
        0.96,
        (
            f"Coverage overlap: {time_overlap_label}\n"
            f"CLARA rows in ERA5 time range: {time_window_count}/{len(clara_plot)}\n"
            f"CLARA rows exactly on ERA5 valid minute: {exact_time_count}/{len(clara_plot)}"
        ),
        transform=time_ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.85},
    )
    time_ax.legend(loc="lower left")

    space_ax.scatter(
        clara_lon[~in_time_window_and_space],
        clara_plot.loc[~in_time_window_and_space, "CLARA_fov_latitude"],
        color="0.75",
        s=14,
        alpha=0.5,
        label="CLARA outside ERA5 time range or space",
    )
    space_ax.scatter(
        clara_lon[in_time_window_and_space],
        clara_plot.loc[in_time_window_and_space, "CLARA_fov_latitude"],
        color="#16a34a",
        s=18,
        alpha=0.75,
        label="CLARA inside ERA5 time range and space",
    )
    space_ax.scatter(
        clara_lon[exact_time_and_space],
        clara_plot.loc[exact_time_and_space, "CLARA_fov_latitude"],
        color="#dc2626",
        marker="x",
        s=50,
        linewidths=1.4,
        label="CLARA exactly on ERA5 valid minute and in space",
    )

    era5_box = Rectangle(
        (era5_extent["lon_min"], era5_extent["lat_min"]),
        era5_extent["lon_max"] - era5_extent["lon_min"],
        era5_extent["lat_max"] - era5_extent["lat_min"],
        fill=False,
        edgecolor="#2563eb",
        linewidth=2,
        label="ERA5 spatial extent",
    )
    space_ax.add_patch(era5_box)

    lon_padding = max(2, (clara_lon.max() - clara_lon.min()) * 0.04)
    lat_padding = max(2, (clara_plot["CLARA_fov_latitude"].max() - clara_plot["CLARA_fov_latitude"].min()) * 0.04)
    space_ax.set_xlim(
        min(clara_lon.min(), era5_extent["lon_min"]) - lon_padding,
        max(clara_lon.max(), era5_extent["lon_max"]) + lon_padding,
    )
    space_ax.set_ylim(
        min(clara_plot["CLARA_fov_latitude"].min(), era5_extent["lat_min"]) - lat_padding,
        max(clara_plot["CLARA_fov_latitude"].max(), era5_extent["lat_max"]) + lat_padding,
    )
    space_ax.set_title("Spatial overlap")
    space_ax.set_xlabel(f"Longitude ({clara_lon_name})")
    space_ax.set_ylabel("Latitude")
    space_ax.grid(True, alpha=0.3)
    space_ax.legend(loc="upper right")

    time_window_and_space_count = int(in_time_window_and_space.sum())
    exact_time_and_space_count = int(exact_time_and_space.sum())
    spatial_count = int(in_space_overlap.sum())
    total_count = len(clara_plot)
    space_ax.text(
        0.01,
        0.02,
        (
            f"CLARA points in ERA5 space: {spatial_count}/{total_count} "
            f"({spatial_count / total_count:.1%})\n"
            f"CLARA points in time range and space: {time_window_and_space_count}/{total_count} "
            f"({time_window_and_space_count / total_count:.1%})\n"
            f"CLARA points on exact ERA5 minute and in space: {exact_time_and_space_count}/{total_count} "
            f"({exact_time_and_space_count / total_count:.1%})"
        ),
        transform=space_ax.transAxes,
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.85},
    )

    fig.suptitle("CLARA matched vs ERA5 matched overlap", fontsize=14)
    return fig, {
        "clara_time_start": clara_time_start,
        "clara_time_end": clara_time_end,
        "era5_time_start": era5_time_start,
        "era5_time_end": era5_time_end,
        "overlap_start": overlap_start,
        "overlap_end": overlap_end,
        "has_time_overlap": has_time_overlap,
        "time_window_count": time_window_count,
        "exact_time_count": exact_time_count,
        "spatial_count": spatial_count,
        "time_window_and_space_count": time_window_and_space_count,
        "exact_time_and_space_count": exact_time_and_space_count,
        "total_count": total_count,
        "era5_extent": era5_extent,
    }


clara_matched = load_clara_matched(CLARA_PATH)
era5_matched = load_era5_matched(ERA5_PATHS)

fig, summary = plot_overlap(clara_matched, era5_matched)

print(f"CLARA time range: {format_timestamp(summary['clara_time_start'])} to {format_timestamp(summary['clara_time_end'])}")
print(f"ERA5 time range: {format_timestamp(summary['era5_time_start'])} to {format_timestamp(summary['era5_time_end'])}")
if summary["has_time_overlap"]:
    print(f"Temporal range overlap: {format_timestamp(summary['overlap_start'])} to {format_timestamp(summary['overlap_end'])}")
else:
    print("Temporal range overlap: none")
print(
    "ERA5 spatial extent: "
    f"lon {summary['era5_extent']['lon_min']:.2f} to {summary['era5_extent']['lon_max']:.2f}, "
    f"lat {summary['era5_extent']['lat_min']:.2f} to {summary['era5_extent']['lat_max']:.2f}"
)
print(
    f"CLARA points inside ERA5 space: {summary['spatial_count']}/{summary['total_count']} "
    f"({summary['spatial_count'] / summary['total_count']:.1%})"
)
print(
    f"CLARA rows inside ERA5 time range: {summary['time_window_count']}/{summary['total_count']} "
    f"({summary['time_window_count'] / summary['total_count']:.1%})"
)
print(
    f"CLARA rows exactly equal to an ERA5 valid minute: {summary['exact_time_count']}/{summary['total_count']} "
    f"({summary['exact_time_count'] / summary['total_count']:.1%})"
)
print(
    f"CLARA points inside ERA5 time range and space: {summary['time_window_and_space_count']}/{summary['total_count']} "
    f"({summary['time_window_and_space_count'] / summary['total_count']:.1%})"
)
print(
    "CLARA points exactly equal to an ERA5 valid minute and inside ERA5 space: "
    f"{summary['exact_time_and_space_count']}/{summary['total_count']} "
    f"({summary['exact_time_and_space_count'] / summary['total_count']:.1%})"
)

fig.savefig(OUTPUT_PATH, dpi=200)
print(f"Saved overlap figure to {OUTPUT_PATH}")

if plt.get_backend().lower() == "agg":
    plt.close(fig)
else:
    show_or_save(plt, subset_results_path("exploratory/data_overlap_01.png"))
