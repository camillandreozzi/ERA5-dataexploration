from pathlib import Path
import argparse
import os
import tempfile


os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "exploratory" / "clara"

DEFAULT_CLARA_PATH = DATA_DIR / "CLARA_matched.pkl"
DEFAULT_HOURLY_GRID_PATH = DATA_DIR / "CLARA_hourly_by_grid.pkl"

TIME_COLUMN = "TimeJD"
RADIANCE_COLUMN = "CLARA_radiance"
LONGITUDE_COLUMN = "CLARA_fov_longitude"
LATITUDE_COLUMN = "CLARA_fov_latitude"

HOUR_COLUMN = "hour"
GRID_LONGITUDE_COLUMN = "longitude"
GRID_LATITUDE_COLUMN = "latitude"
GRID_RADIANCE_COLUMN = "clara_radiance_hourly_mean"
GRID_COUNT_COLUMN = "clara_n_datapoints"

RANDOM_SEED = 42
MAX_PLOT_POINTS = 50_000
MAX_SPATIAL_PAIRS = 50_000
MAX_TEMPORAL_LAG_HOURS = 48
N_DISTANCE_BINS = 12
MIN_TEMPORAL_PAIRS = 6
MIN_SPATIAL_PAIRS_TOTAL = 30
MIN_SPATIAL_PAIRS_PER_BIN = 6


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Explore CLARA radiance distributions, linear/log-positive summaries, "
            "and temporal/spatial autocorrelation."
        )
    )
    parser.add_argument("--clara-path", type=Path, default=DEFAULT_CLARA_PATH)
    parser.add_argument("--hourly-grid-path", type=Path, default=DEFAULT_HOURLY_GRID_PATH)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--max-plot-points", type=int, default=MAX_PLOT_POINTS)
    parser.add_argument("--max-spatial-pairs", type=int, default=MAX_SPATIAL_PAIRS)
    parser.add_argument("--max-temporal-lag-hours", type=int, default=MAX_TEMPORAL_LAG_HOURS)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def ensure_output_dir(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)


def save_dataframe(frame, output_path):
    frame.to_csv(output_path, index=False)
    print(f"Saved {output_path}")


def save_blank_figure(output_path, title, message):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved {output_path}")


def load_clara(path):
    if not path.exists():
        raise FileNotFoundError(f"CLARA file not found: {path}")

    clara = pd.read_pickle(path)
    required_columns = {TIME_COLUMN, RADIANCE_COLUMN, LONGITUDE_COLUMN, LATITUDE_COLUMN}
    missing_columns = sorted(required_columns - set(clara.columns))
    if missing_columns:
        raise ValueError(f"CLARA file is missing columns: {missing_columns}")

    clara = clara.copy()
    clara[TIME_COLUMN] = pd.to_datetime(clara[TIME_COLUMN])
    return clara


def load_hourly_grid(path):
    if not path.exists():
        print(f"Hourly CLARA grid file not found, skipping gridded outputs: {path}")
        return None

    hourly = pd.read_pickle(path)
    required_columns = {
        HOUR_COLUMN,
        GRID_LONGITUDE_COLUMN,
        GRID_LATITUDE_COLUMN,
        GRID_RADIANCE_COLUMN,
        GRID_COUNT_COLUMN,
    }
    missing_columns = sorted(required_columns - set(hourly.columns))
    if missing_columns:
        raise ValueError(f"Hourly CLARA grid file is missing columns: {missing_columns}")

    hourly = hourly.copy()
    hourly[HOUR_COLUMN] = pd.to_datetime(hourly[HOUR_COLUMN])
    return hourly.reset_index(drop=True)


def finite_values(series):
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def log_positive_values(values):
    values = np.asarray(values, dtype=float)
    positive = values[np.isfinite(values) & (values > 0)]
    return np.log(positive)


def summary_row(source, scale, values, total_rows, nonpositive_excluded=0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    quantiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]

    if len(values):
        quantile_values = np.quantile(values, quantiles)
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
    else:
        quantile_values = [np.nan] * len(quantiles)
        minimum = np.nan
        maximum = np.nan
        mean = np.nan
        std = np.nan

    row = {
        "source": source,
        "scale": scale,
        "total_rows": int(total_rows),
        "n_used": int(len(values)),
        "n_missing_or_invalid": int(total_rows - len(values) - nonpositive_excluded),
        "n_nonpositive_excluded": int(nonpositive_excluded),
        "min": minimum,
        "p01": float(quantile_values[0]),
        "p05": float(quantile_values[1]),
        "p25": float(quantile_values[2]),
        "median": float(quantile_values[3]),
        "mean": mean,
        "std": std,
        "p75": float(quantile_values[4]),
        "p95": float(quantile_values[5]),
        "p99": float(quantile_values[6]),
        "max": maximum,
    }
    return row


def build_radiance_summary(clara, hourly):
    raw_values = finite_values(clara[RADIANCE_COLUMN])
    raw_nonpositive = int(np.sum(raw_values <= 0))

    rows = [
        summary_row("raw_clara", "linear", raw_values, len(clara)),
        summary_row(
            "raw_clara",
            "log_positive",
            log_positive_values(raw_values),
            len(clara),
            nonpositive_excluded=raw_nonpositive,
        ),
    ]

    if hourly is not None:
        hourly_values = finite_values(hourly[GRID_RADIANCE_COLUMN])
        hourly_nonpositive = int(np.sum(hourly_values <= 0))
        rows.extend(
            [
                summary_row("hourly_grid", "linear", hourly_values, len(hourly)),
                summary_row(
                    "hourly_grid",
                    "log_positive",
                    log_positive_values(hourly_values),
                    len(hourly),
                    nonpositive_excluded=hourly_nonpositive,
                ),
            ]
        )

    return pd.DataFrame(rows)


def hourly_radiance_summary(clara, hourly):
    raw_hourly = (
        clara[[TIME_COLUMN, RADIANCE_COLUMN]]
        .dropna()
        .assign(hour=lambda frame: frame[TIME_COLUMN].dt.floor("h"))
        .groupby("hour", as_index=False)
        .agg(
            raw_n_samples=(RADIANCE_COLUMN, "size"),
            raw_mean_radiance=(RADIANCE_COLUMN, "mean"),
            raw_median_radiance=(RADIANCE_COLUMN, "median"),
            raw_min_radiance=(RADIANCE_COLUMN, "min"),
            raw_max_radiance=(RADIANCE_COLUMN, "max"),
        )
    )

    if hourly is None:
        return raw_hourly.sort_values("hour", ignore_index=True)

    gridded_hourly = (
        hourly.groupby(HOUR_COLUMN, as_index=False)
        .agg(
            hourly_grid_cells=(GRID_RADIANCE_COLUMN, "size"),
            hourly_grid_observations=(GRID_COUNT_COLUMN, "sum"),
            hourly_grid_mean_radiance=(GRID_RADIANCE_COLUMN, "mean"),
            hourly_grid_median_radiance=(GRID_RADIANCE_COLUMN, "median"),
            hourly_grid_min_radiance=(GRID_RADIANCE_COLUMN, "min"),
            hourly_grid_max_radiance=(GRID_RADIANCE_COLUMN, "max"),
        )
        .rename(columns={HOUR_COLUMN: "hour"})
    )

    return raw_hourly.merge(gridded_hourly, on="hour", how="outer").sort_values(
        "hour", ignore_index=True
    )


def sample_values(values, max_points, rng):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) <= max_points:
        return values
    selected = rng.choice(np.arange(len(values)), size=max_points, replace=False)
    return values[selected]


def plot_histogram(ax, values, title, x_label, max_points, rng):
    values = sample_values(values, max_points, rng)
    if len(values) == 0:
        ax.text(0.5, 0.5, "No finite values", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.axis("off")
        return

    ax.hist(values, bins=45, color="#2563eb", edgecolor="white", alpha=0.82)
    ax.axvline(np.median(values), color="#b45309", linewidth=1.6, label="median")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right")


def plot_radiance_distributions(clara, hourly, output_dir, max_points, rng):
    raw_values = finite_values(clara[RADIANCE_COLUMN])

    if hourly is not None:
        hourly_values = finite_values(hourly[GRID_RADIANCE_COLUMN])
    else:
        hourly_values = np.array([], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    plot_histogram(
        axes[0, 0],
        raw_values,
        "Raw CLARA radiance",
        "Radiance",
        max_points,
        rng,
    )
    plot_histogram(
        axes[0, 1],
        log_positive_values(raw_values),
        "Raw CLARA log-positive radiance",
        "log(radiance), radiance > 0",
        max_points,
        rng,
    )
    plot_histogram(
        axes[1, 0],
        hourly_values,
        "Hourly gridded CLARA radiance",
        "Radiance",
        max_points,
        rng,
    )
    plot_histogram(
        axes[1, 1],
        log_positive_values(hourly_values),
        "Hourly gridded log-positive radiance",
        "log(radiance), radiance > 0",
        max_points,
        rng,
    )

    fig.suptitle("CLARA radiance distributions: linear vs log-positive", fontsize=14)
    fig.tight_layout()
    output_path = output_dir / "radiance_distribution_log_vs_linear.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_hourly_radiance(hourly_summary, output_dir):
    if hourly_summary.empty:
        save_blank_figure(output_dir / "radiance_by_hour.png", "CLARA hourly radiance", "No rows found")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    if "raw_mean_radiance" in hourly_summary:
        ax.plot(
            hourly_summary["hour"],
            hourly_summary["raw_mean_radiance"],
            color="#2563eb",
            marker="o",
            linewidth=1.8,
            markersize=3,
            label="Raw hourly mean",
        )

    if "hourly_grid_mean_radiance" in hourly_summary:
        ax.plot(
            hourly_summary["hour"],
            hourly_summary["hourly_grid_mean_radiance"],
            color="#16a34a",
            marker="o",
            linewidth=1.8,
            markersize=3,
            label="Gridded hourly mean",
        )

    ax.set_title("CLARA radiance by hour")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Radiance")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    fig.tight_layout()
    output_path = output_dir / "radiance_by_hour.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved {output_path}")


def autocorrelation_rows(series, label, max_lag_hours):
    series = series.sort_index()
    all_hours = pd.date_range(series.index.min(), series.index.max(), freq="h")
    series = series.reindex(all_hours)
    rows = []

    n_non_null = int(series.notna().sum())
    if n_non_null >= MIN_TEMPORAL_PAIRS and series.nunique(dropna=True) > 1:
        rows.append(
            {
                "series": label,
                "lag_hours": 0,
                "n_pairs": n_non_null,
                "autocorrelation": 1.0,
            }
        )

    for lag in range(1, max_lag_hours + 1):
        lagged = pd.concat([series, series.shift(lag)], axis=1).dropna()
        lagged.columns = ["current", "lagged"]

        if (
            len(lagged) < MIN_TEMPORAL_PAIRS
            or lagged["current"].nunique(dropna=True) < 2
            or lagged["lagged"].nunique(dropna=True) < 2
        ):
            continue

        rows.append(
            {
                "series": label,
                "lag_hours": lag,
                "n_pairs": int(len(lagged)),
                "autocorrelation": float(lagged["current"].corr(lagged["lagged"])),
            }
        )

    return rows


def compute_temporal_autocorrelation(clara, hourly, max_lag_hours):
    rows = []
    raw_series = (
        clara[[TIME_COLUMN, RADIANCE_COLUMN]]
        .dropna()
        .assign(hour=lambda frame: frame[TIME_COLUMN].dt.floor("h"))
        .groupby("hour")[RADIANCE_COLUMN]
        .mean()
    )
    if not raw_series.empty:
        rows.extend(autocorrelation_rows(raw_series, "raw_hourly_mean", max_lag_hours))

    if hourly is not None:
        grid_series = hourly.groupby(HOUR_COLUMN)[GRID_RADIANCE_COLUMN].mean()
        if not grid_series.empty:
            rows.extend(autocorrelation_rows(grid_series, "gridded_hourly_mean", max_lag_hours))

    return pd.DataFrame(rows)


def plot_temporal_autocorrelation(autocorrelation, output_dir):
    output_path = output_dir / "temporal_autocorrelation.png"

    if autocorrelation.empty:
        save_blank_figure(
            output_path,
            "CLARA temporal autocorrelation",
            "No temporal autocorrelation lags met sample thresholds",
        )
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for series_name, subset in autocorrelation.groupby("series", sort=False):
        ax.plot(
            subset["lag_hours"],
            subset["autocorrelation"],
            marker="o",
            linewidth=1.8,
            markersize=3.5,
            label=series_name.replace("_", " "),
        )

    ax.axhline(0, color="#6b7280", linewidth=0.8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("CLARA temporal autocorrelation")
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("Autocorrelation")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved {output_path}")


def haversine_km(lat1, lon1, lat2, lon2):
    radius_km = 6371.0088
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    delta_lat = lat2 - lat1
    delta_lon = np.radians(lon2 - lon1)
    a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    return 2 * radius_km * np.arcsin(np.sqrt(a))


def random_pairs_for_indices(indices, n_pairs, rng):
    indices = np.asarray(indices, dtype=np.int64)
    n_indices = len(indices)
    total_pairs = n_indices * (n_indices - 1) // 2

    if n_indices < 2 or n_pairs <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    if n_pairs >= total_pairs:
        left, right = np.triu_indices(n_indices, k=1)
        return indices[left], indices[right]

    left_positions = rng.integers(0, n_indices, size=n_pairs)
    right_positions = rng.integers(0, n_indices - 1, size=n_pairs)
    right_positions = np.where(right_positions >= left_positions, right_positions + 1, right_positions)
    return indices[left_positions], indices[right_positions]


def sample_same_hour_pairs(frame, max_pairs, rng):
    if frame.empty:
        return pd.DataFrame(columns=["left_index", "right_index", "distance_km"])

    frame = frame.reset_index(drop=True)
    group_sizes = frame.groupby(HOUR_COLUMN).size()
    possible_by_hour = (group_sizes * (group_sizes - 1) // 2).loc[lambda values: values > 0]
    total_possible = int(possible_by_hour.sum())

    if total_possible == 0:
        return pd.DataFrame(columns=["left_index", "right_index", "distance_km"])

    use_all_pairs = max_pairs is None or total_possible <= max_pairs
    left_parts = []
    right_parts = []

    for hour, possible_pairs in possible_by_hour.items():
        indices = frame.index[frame[HOUR_COLUMN] == hour].to_numpy()
        if use_all_pairs:
            n_pairs = int(possible_pairs)
        else:
            proportional = max_pairs * float(possible_pairs) / float(total_possible)
            n_pairs = max(1, min(int(possible_pairs), int(round(proportional))))

        left, right = random_pairs_for_indices(indices, n_pairs, rng)
        left_parts.append(left)
        right_parts.append(right)

    left_index = np.concatenate(left_parts)
    right_index = np.concatenate(right_parts)

    if max_pairs is not None and len(left_index) > max_pairs:
        selected = rng.choice(np.arange(len(left_index)), size=max_pairs, replace=False)
        left_index = left_index[selected]
        right_index = right_index[selected]

    lat = frame[GRID_LATITUDE_COLUMN].to_numpy(dtype=float)
    lon = frame[GRID_LONGITUDE_COLUMN].to_numpy(dtype=float)
    distance_km = haversine_km(lat[left_index], lon[left_index], lat[right_index], lon[right_index])

    return pd.DataFrame(
        {
            "left_index": left_index.astype(np.int64),
            "right_index": right_index.astype(np.int64),
            "distance_km": distance_km,
        }
    )


def distance_bin_edges(distances):
    finite_distances = np.asarray(distances[np.isfinite(distances)], dtype=float)
    if len(finite_distances) == 0:
        return None

    max_distance = float(finite_distances.max())
    if max_distance <= 0:
        return None

    edges = np.linspace(0.0, max_distance, N_DISTANCE_BINS + 1)
    edges[-1] = edges[-1] + max(max_distance * 1e-9, 1e-9)
    return edges


def compute_spatial_autocorrelation(hourly, pairs):
    if hourly is None or hourly.empty or pairs.empty:
        return pd.DataFrame()

    left_index = pairs["left_index"].to_numpy(dtype=np.int64)
    right_index = pairs["right_index"].to_numpy(dtype=np.int64)
    distances = pairs["distance_km"].to_numpy(dtype=float)
    values = hourly.reset_index(drop=True)[GRID_RADIANCE_COLUMN].to_numpy(dtype=float)
    left_values = values[left_index]
    right_values = values[right_index]
    usable = np.isfinite(left_values) & np.isfinite(right_values) & np.isfinite(distances)

    if int(usable.sum()) < MIN_SPATIAL_PAIRS_TOTAL:
        return pd.DataFrame()

    edges = distance_bin_edges(distances[usable])
    if edges is None:
        return pd.DataFrame()

    distance_bins = np.digitize(distances, edges, right=False) - 1
    rows = []

    for index in range(len(edges) - 1):
        mask = usable & (distance_bins == index)
        if int(mask.sum()) < MIN_SPATIAL_PAIRS_PER_BIN:
            continue

        left_bin = left_values[mask]
        right_bin = right_values[mask]
        if np.unique(left_bin).size < 2 or np.unique(right_bin).size < 2:
            continue

        rows.append(
            {
                "distance_bin": index + 1,
                "distance_start_km": float(edges[index]),
                "distance_end_km": float(edges[index + 1]),
                "distance_mean_km": float(np.mean(distances[mask])),
                "n_pairs": int(mask.sum()),
                "spatial_autocorrelation": float(np.corrcoef(left_bin, right_bin)[0, 1]),
                "semivariance": float(0.5 * np.mean((left_bin - right_bin) ** 2)),
            }
        )

    return pd.DataFrame(rows)


def plot_spatial_autocorrelation(spatial, output_dir):
    output_path = output_dir / "spatial_autocorrelation.png"

    if spatial.empty:
        save_blank_figure(
            output_path,
            "CLARA spatial autocorrelation",
            "No same-hour spatial bins met sample thresholds",
        )
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        spatial["distance_mean_km"],
        spatial["spatial_autocorrelation"],
        color="#0f766e",
        marker="o",
        linewidth=1.8,
    )
    ax.axhline(0, color="#6b7280", linewidth=0.8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("CLARA same-hour spatial autocorrelation")
    ax.set_xlabel("Distance between gridded CLARA cells (km)")
    ax.set_ylabel("Spatial autocorrelation")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_spatial_variogram(spatial, output_dir):
    output_path = output_dir / "spatial_variogram.png"

    if spatial.empty:
        save_blank_figure(
            output_path,
            "CLARA spatial variogram",
            "No same-hour spatial bins met sample thresholds",
        )
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        spatial["distance_mean_km"],
        spatial["semivariance"],
        color="#7c3aed",
        marker="o",
        linewidth=1.8,
    )
    ax.set_title("CLARA same-hour empirical variogram")
    ax.set_xlabel("Distance between gridded CLARA cells (km)")
    ax.set_ylabel("Semivariance")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_spatial_radiance(hourly, output_dir):
    output_path = output_dir / "spatial_radiance_mean.png"
    if hourly is None or hourly.empty:
        save_blank_figure(output_path, "CLARA spatial radiance", "No gridded CLARA rows found")
        return

    spatial_mean = (
        hourly.groupby([GRID_LONGITUDE_COLUMN, GRID_LATITUDE_COLUMN], as_index=False)
        .agg(
            mean_radiance=(GRID_RADIANCE_COLUMN, "mean"),
            n_grid_hours=(GRID_RADIANCE_COLUMN, "size"),
            n_observations=(GRID_COUNT_COLUMN, "sum"),
        )
        .sort_values("mean_radiance")
    )

    fig, ax = plt.subplots(figsize=(11, 6.5))
    scatter = ax.scatter(
        spatial_mean[GRID_LONGITUDE_COLUMN],
        spatial_mean[GRID_LATITUDE_COLUMN],
        c=spatial_mean["mean_radiance"],
        s=18 + 3 * np.sqrt(spatial_mean["n_observations"]),
        cmap="viridis",
        alpha=0.82,
        edgecolors="none",
    )
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Mean CLARA radiance")
    ax.set_title("Mean gridded CLARA radiance by location")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    args = parse_args()
    ensure_output_dir(args.output_dir)
    rng = np.random.default_rng(args.random_seed)

    clara = load_clara(args.clara_path)
    hourly = load_hourly_grid(args.hourly_grid_path)

    radiance_summary = build_radiance_summary(clara, hourly)
    save_dataframe(radiance_summary, args.output_dir / "radiance_summary.csv")

    hourly_summary = hourly_radiance_summary(clara, hourly)
    save_dataframe(hourly_summary, args.output_dir / "radiance_by_hour.csv")

    plot_radiance_distributions(clara, hourly, args.output_dir, args.max_plot_points, rng)
    plot_hourly_radiance(hourly_summary, args.output_dir)
    plot_spatial_radiance(hourly, args.output_dir)

    temporal = compute_temporal_autocorrelation(clara, hourly, args.max_temporal_lag_hours)
    save_dataframe(temporal, args.output_dir / "temporal_autocorrelation.csv")
    plot_temporal_autocorrelation(temporal, args.output_dir)

    pairs = sample_same_hour_pairs(hourly, args.max_spatial_pairs, rng) if hourly is not None else pd.DataFrame()
    spatial = compute_spatial_autocorrelation(hourly, pairs)
    save_dataframe(spatial, args.output_dir / "spatial_autocorrelation.csv")
    plot_spatial_autocorrelation(spatial, args.output_dir)
    plot_spatial_variogram(spatial, args.output_dir)

    print(f"Exploratory CLARA outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
