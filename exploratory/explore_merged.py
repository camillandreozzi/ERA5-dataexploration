import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "paths.py").exists():
        sys.path.insert(0, str(_p))
        break
from paths import subset_data_path, subset_results_path

from collections import defaultdict
from pathlib import Path
import math
import os
import tempfile


os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq


RESULTS_DIR = subset_results_path("exploratory/merged")

MERGED_PATH = subset_data_path("CLARA_ERA5_merged.parquet")

HOUR_COLUMN = "hour"
LONGITUDE_COLUMN = "longitude"
LATITUDE_COLUMN = "latitude"
RADIANCE_COLUMN = "clara_radiance_hourly_mean"
CLARA_COUNT_COLUMN = "clara_n_datapoints"

RANDOM_SEED = 42
BATCH_SIZE = 1_000_000
MAX_SCATTER_POINTS = 20_000
MAX_PAIR_SAMPLE = 50_000
N_DISTANCE_BINS = 12
MAX_TEMPORAL_LAG_HOURS = 24
MIN_NON_NULL_ROWS = 20
MIN_PAIRS_TOTAL = 50
MIN_PAIRS_PER_BIN = 10
MIN_TEMPORAL_PAIRS = 6

PREFERRED_COVARIATES = [
    "era5_tcc",
    "era5_lcc",
    "era5_mcc",
    "era5_hcc",
    "era5_cloud_layer_sum",
    "era5_ssrd",
    "era5_strd",
    "era5_t2m_c",
    "era5_d2m_c",
    "era5_dewpoint_depression",
    "era5_tcwv",
    "era5_wind10_speed",
    "era5_tp",
    "era5_cp",
    "era5_cape",
    "era5_blh",
    "era5_sp",
    "era5_msl",
    "era5_i10fg",
]

DERIVED_COVARIATE_DEPENDENCIES = {
    "era5_t2m_c": ["era5_t2m"],
    "era5_d2m_c": ["era5_d2m"],
    "era5_dewpoint_depression": ["era5_t2m", "era5_d2m"],
    "era5_wind10_speed": ["era5_u10", "era5_v10"],
    "era5_cloud_layer_sum": ["era5_lcc", "era5_mcc", "era5_hcc"],
}

VARIABLE_LABELS = {
    RADIANCE_COLUMN: "CLARA radiance",
    "era5_tcc": "Total cloud cover",
    "era5_lcc": "Low cloud cover",
    "era5_mcc": "Medium cloud cover",
    "era5_hcc": "High cloud cover",
    "era5_ssrd": "Surface solar radiation downwards",
    "era5_strd": "Surface thermal radiation downwards",
    "era5_t2m": "2 m temperature (K)",
    "era5_t2m_c": "2 m temperature (C)",
    "era5_d2m": "2 m dewpoint (K)",
    "era5_d2m_c": "2 m dewpoint (C)",
    "era5_dewpoint_depression": "Dewpoint depression (K)",
    "era5_tcwv": "Total column water vapour",
    "era5_u10": "10 m east wind",
    "era5_v10": "10 m north wind",
    "era5_wind10_speed": "10 m wind speed",
    "era5_tp": "Total precipitation",
    "era5_cp": "Convective precipitation",
    "era5_cape": "CAPE",
    "era5_blh": "Boundary layer height",
    "era5_sp": "Surface pressure",
    "era5_msl": "Mean sea-level pressure",
    "era5_i10fg": "10 m wind gust",
    "era5_cloud_layer_sum": "Cloud layer sum",
}


def variable_label(variable):
    return VARIABLE_LABELS.get(variable, variable.replace("_", " "))


def ensure_output_dir():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_blank_figure(output_path, title, message):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_dataframe(frame, output_path):
    frame.to_csv(output_path, index=False)
    print(f"Saved {output_path}")


def parquet_schema_names(path):
    return list(pq.ParquetFile(path).schema_arrow.names)


def available_base_covariates(schema_names):
    available = set(schema_names)
    covariates = []

    for variable in PREFERRED_COVARIATES:
        if variable in DERIVED_COVARIATE_DEPENDENCIES:
            dependencies = DERIVED_COVARIATE_DEPENDENCIES[variable]
            if all(dependency in available for dependency in dependencies):
                covariates.append(variable)
        elif variable in available:
            covariates.append(variable)

    return covariates


def columns_needed_for_matched_rows(schema_names, covariates):
    available = set(schema_names)
    required = {
        HOUR_COLUMN,
        LONGITUDE_COLUMN,
        LATITUDE_COLUMN,
        RADIANCE_COLUMN,
        CLARA_COUNT_COLUMN,
    }
    missing_required = sorted(required - available)
    if missing_required:
        raise ValueError(f"Merged parquet is missing required columns: {missing_required}")

    needed = set(required)
    for variable in covariates:
        if variable in available:
            needed.add(variable)
        elif variable in DERIVED_COVARIATE_DEPENDENCIES:
            needed.update(DERIVED_COVARIATE_DEPENDENCIES[variable])

    return [column for column in schema_names if column in needed]


def add_derived_covariates(frame):
    if "era5_t2m" in frame:
        frame["era5_t2m_c"] = frame["era5_t2m"] - 273.15

    if "era5_d2m" in frame:
        frame["era5_d2m_c"] = frame["era5_d2m"] - 273.15

    if {"era5_t2m", "era5_d2m"}.issubset(frame.columns):
        frame["era5_dewpoint_depression"] = frame["era5_t2m"] - frame["era5_d2m"]

    if {"era5_u10", "era5_v10"}.issubset(frame.columns):
        frame["era5_wind10_speed"] = np.sqrt(frame["era5_u10"] ** 2 + frame["era5_v10"] ** 2)

    if {"era5_lcc", "era5_mcc", "era5_hcc"}.issubset(frame.columns):
        frame["era5_cloud_layer_sum"] = frame["era5_lcc"] + frame["era5_mcc"] + frame["era5_hcc"]

    return frame


def load_matched_rows(path, schema_names, covariates):
    columns = columns_needed_for_matched_rows(schema_names, covariates)
    dataset = ds.dataset(path, format="parquet")
    table = dataset.to_table(columns=columns, filter=ds.field(CLARA_COUNT_COLUMN) > 0)
    frame = table.to_pandas().reset_index(drop=True)
    frame[HOUR_COLUMN] = pd.to_datetime(frame[HOUR_COLUMN])
    frame = add_derived_covariates(frame)
    print(f"Loaded {len(frame):,} CLARA-matched grid-hour rows")
    return frame


def finite_pair(frame, x_column, y_column):
    pair = frame[[x_column, y_column]].replace([np.inf, -np.inf], np.nan).dropna()
    return pair


def usable_covariates(frame, covariates):
    usable = []

    for variable in covariates:
        if variable not in frame:
            print(f"Skipping {variable}: column is not available")
            continue

        pair = finite_pair(frame, variable, RADIANCE_COLUMN)
        if len(pair) < MIN_NON_NULL_ROWS:
            print(f"Skipping {variable}: only {len(pair)} non-null radiance/covariate pairs")
            continue

        if pair[variable].nunique(dropna=True) < 2:
            print(f"Skipping {variable}: covariate is constant after filtering")
            continue

        usable.append(variable)

    return usable


def sample_frame(frame, max_rows, rng):
    if max_rows is None or len(frame) <= max_rows:
        return frame

    selected = rng.choice(frame.index.to_numpy(), size=max_rows, replace=False)
    return frame.loc[selected].sort_index()


# ---------------------------------------------------------------------------
# 1. Matched-cell counts: ERA5 rows vs CLARA matched cells
# ---------------------------------------------------------------------------


def era5_hour_counts_from_metadata(parquet_file):
    schema_names = list(parquet_file.schema_arrow.names)
    if HOUR_COLUMN not in schema_names:
        return None

    hour_index = schema_names.index(HOUR_COLUMN)
    records = []

    for row_group_index in range(parquet_file.metadata.num_row_groups):
        row_group = parquet_file.metadata.row_group(row_group_index)
        statistics = row_group.column(hour_index).statistics

        if statistics is None or not statistics.has_min_max:
            return None

        min_hour = pd.Timestamp(statistics.min)
        max_hour = pd.Timestamp(statistics.max)
        if min_hour != max_hour:
            return None

        records.append({HOUR_COLUMN: min_hour, "era5_rows": row_group.num_rows})

    if not records:
        return None

    return (
        pd.DataFrame(records)
        .groupby(HOUR_COLUMN, as_index=False)["era5_rows"]
        .sum()
        .sort_values(HOUR_COLUMN)
        .reset_index(drop=True)
    )


def era5_hour_counts_from_batches(path):
    counts = defaultdict(int)
    dataset = ds.dataset(path, format="parquet")
    scanner = dataset.scanner(columns=[HOUR_COLUMN], batch_size=BATCH_SIZE)

    for batch in scanner.to_batches():
        hours = batch.column(0).to_pandas()
        for hour, count in hours.value_counts(dropna=False).items():
            if pd.notna(hour):
                counts[pd.Timestamp(hour)] += int(count)

    return pd.DataFrame(
        [{HOUR_COLUMN: hour, "era5_rows": count} for hour, count in counts.items()]
    ).sort_values(HOUR_COLUMN, ignore_index=True)


def get_era5_hour_counts(path, parquet_file):
    counts = era5_hour_counts_from_metadata(parquet_file)
    if counts is not None:
        print("Read ERA5 hourly row counts from parquet metadata")
        return counts

    print("Parquet row groups are not single-hour groups; reading hour column in batches")
    return era5_hour_counts_from_batches(path)


def matched_counts_by_hour(matched):
    if matched.empty:
        return pd.DataFrame(
            columns=[
                HOUR_COLUMN,
                "clara_matched_cells",
                "clara_observations",
                "clara_mean_radiance",
                "clara_median_radiance",
            ]
        )

    return (
        matched.groupby(HOUR_COLUMN, as_index=False)
        .agg(
            clara_matched_cells=(RADIANCE_COLUMN, "size"),
            clara_observations=(CLARA_COUNT_COLUMN, "sum"),
            clara_mean_radiance=(RADIANCE_COLUMN, "mean"),
            clara_median_radiance=(RADIANCE_COLUMN, "median"),
        )
        .sort_values(HOUR_COLUMN)
        .reset_index(drop=True)
    )


def build_hourly_match_summary(era5_hourly, matched):
    clara_hourly = matched_counts_by_hour(matched)
    hourly = era5_hourly.merge(clara_hourly, on=HOUR_COLUMN, how="left")

    for column in ["clara_matched_cells", "clara_observations"]:
        hourly[column] = hourly[column].fillna(0).astype(int)

    hourly["clara_cell_match_rate"] = hourly["clara_matched_cells"] / hourly["era5_rows"]
    return hourly


def plot_matched_hourly_counts(hourly, output_path):
    if hourly.empty:
        save_blank_figure(output_path, "ERA5 and CLARA hourly counts", "No hourly rows found")
        return

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

    axes[0].plot(hourly[HOUR_COLUMN], hourly["era5_rows"], color="#2563eb", linewidth=2)
    axes[0].set_ylabel("ERA5 rows")
    axes[0].set_title("ERA5 grid rows by hour")
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(
        hourly[HOUR_COLUMN],
        hourly["clara_matched_cells"],
        width=0.03,
        color="#16a34a",
        alpha=0.75,
        label="Matched grid cells",
    )
    axes[1].plot(
        hourly[HOUR_COLUMN],
        hourly["clara_observations"],
        color="#166534",
        marker="o",
        linewidth=1.4,
        markersize=3,
        label="CLARA observations",
    )
    axes[1].set_ylabel("CLARA count")
    axes[1].set_title("CLARA matched cells and observations by hour")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend(loc="upper right")

    axes[2].plot(
        hourly[HOUR_COLUMN],
        hourly["clara_mean_radiance"],
        color="#dc2626",
        marker="o",
        linewidth=1.6,
        markersize=3,
    )
    axes[2].set_ylabel("Mean radiance")
    axes[2].set_title("CLARA mean radiance by matched hour")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlabel("Hour")
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_matched_spatial_coverage(matched, output_path):
    if matched.empty:
        save_blank_figure(output_path, "Matched-cell spatial coverage", "No matched CLARA rows found")
        return

    fig, ax = plt.subplots(figsize=(11, 7))
    scatter = ax.scatter(
        matched[LONGITUDE_COLUMN],
        matched[LATITUDE_COLUMN],
        c=matched[CLARA_COUNT_COLUMN],
        s=18 + 7 * matched[CLARA_COUNT_COLUMN],
        cmap="viridis",
        alpha=0.78,
        edgecolors="none",
    )
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("CLARA observations in grid-hour cell")

    ax.set_title("CLARA matched-cell spatial coverage")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved {output_path}")


def summarize_matched_cells(parquet_file, era5_hourly, matched):
    hourly = build_hourly_match_summary(era5_hourly, matched)
    save_dataframe(hourly, RESULTS_DIR / "matched_counts_by_hour.csv")

    summary = {
        "total_era5_rows": int(parquet_file.metadata.num_rows),
        "era5_row_groups": int(parquet_file.metadata.num_row_groups),
        "era5_hours": int(era5_hourly[HOUR_COLUMN].nunique()),
        "median_era5_rows_per_hour": float(era5_hourly["era5_rows"].median()),
        "clara_matched_grid_hour_cells": int(len(matched)),
        "clara_observations": int(matched[CLARA_COUNT_COLUMN].sum()) if not matched.empty else 0,
        "clara_matched_hours": int(matched[HOUR_COLUMN].nunique()) if not matched.empty else 0,
        "clara_match_rate_over_era5_rows": float(len(matched) / parquet_file.metadata.num_rows),
        "clara_radiance_min": float(matched[RADIANCE_COLUMN].min()) if not matched.empty else np.nan,
        "clara_radiance_mean": float(matched[RADIANCE_COLUMN].mean()) if not matched.empty else np.nan,
        "clara_radiance_median": float(matched[RADIANCE_COLUMN].median()) if not matched.empty else np.nan,
        "clara_radiance_max": float(matched[RADIANCE_COLUMN].max()) if not matched.empty else np.nan,
        "matched_longitude_min": float(matched[LONGITUDE_COLUMN].min()) if not matched.empty else np.nan,
        "matched_longitude_max": float(matched[LONGITUDE_COLUMN].max()) if not matched.empty else np.nan,
        "matched_latitude_min": float(matched[LATITUDE_COLUMN].min()) if not matched.empty else np.nan,
        "matched_latitude_max": float(matched[LATITUDE_COLUMN].max()) if not matched.empty else np.nan,
    }
    save_dataframe(pd.DataFrame([summary]), RESULTS_DIR / "matched_cell_summary.csv")

    plot_matched_hourly_counts(hourly, RESULTS_DIR / "matched_counts_by_hour.png")
    plot_matched_spatial_coverage(matched, RESULTS_DIR / "matched_spatial_coverage.png")
    return hourly


# ---------------------------------------------------------------------------
# 2. Radiance vs relevant ERA5 covariates
# ---------------------------------------------------------------------------


def covariate_summary_stats(matched, covariates):
    rows = []

    for variable in covariates:
        pair = finite_pair(matched, variable, RADIANCE_COLUMN)
        x = pair[variable]
        y = pair[RADIANCE_COLUMN]

        if len(pair) >= MIN_NON_NULL_ROWS and x.nunique(dropna=True) > 1:
            pearson = y.corr(x, method="pearson")
            spearman = y.corr(x, method="spearman")
        else:
            pearson = np.nan
            spearman = np.nan

        rows.append(
            {
                "variable": variable,
                "label": variable_label(variable),
                "n": int(len(pair)),
                "pearson": pearson,
                "spearman": spearman,
                "covariate_min": float(x.min()) if len(pair) else np.nan,
                "covariate_p05": float(x.quantile(0.05)) if len(pair) else np.nan,
                "covariate_median": float(x.median()) if len(pair) else np.nan,
                "covariate_mean": float(x.mean()) if len(pair) else np.nan,
                "covariate_p95": float(x.quantile(0.95)) if len(pair) else np.nan,
                "covariate_max": float(x.max()) if len(pair) else np.nan,
                "radiance_mean": float(y.mean()) if len(pair) else np.nan,
                "radiance_std": float(y.std()) if len(pair) else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values("variable", ignore_index=True)


def add_linear_trend(ax, x, y):
    if len(x) < MIN_NON_NULL_ROWS or pd.Series(x).nunique(dropna=True) < 2:
        return

    slope, intercept = np.polyfit(x, y, deg=1)
    x_line = np.linspace(np.nanmin(x), np.nanmax(x), 100)
    ax.plot(x_line, slope * x_line + intercept, color="#111827", linewidth=1.5)


def plot_radiance_vs_covariates(matched, covariates, rng):
    stats = covariate_summary_stats(matched, covariates)
    save_dataframe(stats, RESULTS_DIR / "radiance_covariate_stats.csv")

    plot_variables = [
        row.variable
        for row in stats.itertuples()
        if row.n >= MIN_NON_NULL_ROWS and matched[row.variable].nunique(dropna=True) > 1
    ]

    if not plot_variables:
        save_blank_figure(
            RESULTS_DIR / "radiance_vs_covariates.png",
            "CLARA radiance vs ERA5 covariates",
            "No covariates had enough non-null, non-constant rows",
        )
        return stats

    ncols = 3
    nrows = math.ceil(len(plot_variables) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.3 * nrows), squeeze=False)
    stats_by_variable = stats.set_index("variable")

    for ax, variable in zip(axes.ravel(), plot_variables):
        pair = finite_pair(matched, variable, RADIANCE_COLUMN)
        pair = sample_frame(pair, MAX_SCATTER_POINTS, rng)

        x = pair[variable].to_numpy(dtype=float)
        y = pair[RADIANCE_COLUMN].to_numpy(dtype=float)

        ax.scatter(x, y, s=16, alpha=0.45, color="#2563eb", edgecolors="none")
        add_linear_trend(ax, x, y)

        row = stats_by_variable.loc[variable]
        ax.set_title(f"{variable_label(variable)}\nn={int(row['n'])}, Spearman={row['spearman']:.2f}")
        ax.set_xlabel(variable_label(variable))
        ax.set_ylabel("CLARA radiance")
        ax.grid(True, alpha=0.25)

    for ax in axes.ravel()[len(plot_variables):]:
        ax.axis("off")

    fig.suptitle("CLARA radiance vs ERA5 covariates", fontsize=14)
    fig.tight_layout()
    output_path = RESULTS_DIR / "radiance_vs_covariates.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved {output_path}")
    return stats


# ---------------------------------------------------------------------------
# 3. Variograms per covariate
# ---------------------------------------------------------------------------


def haversine_km(lat1, lon1, lat2, lon2):
    earth_radius_km = 6371.0088
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = lat2_rad - lat1_rad
    dlon = np.radians(lon2 - lon1)

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * earth_radius_km * np.arcsin(np.sqrt(a))


def all_pairs_for_indices(indices):
    positions = np.asarray(indices, dtype=np.int64)
    left, right = np.triu_indices(len(positions), k=1)
    return positions[left], positions[right]


def random_pairs_for_indices(indices, n_pairs, rng):
    positions = np.asarray(indices, dtype=np.int64)
    n_positions = len(positions)
    possible_pairs = n_positions * (n_positions - 1) // 2

    if n_pairs >= possible_pairs:
        return all_pairs_for_indices(positions)

    selected = set()
    max_attempts = max(n_pairs * 20, 1_000)
    attempts = 0

    while len(selected) < n_pairs and attempts < max_attempts:
        attempts += 1
        left = int(rng.integers(0, n_positions))
        right = int(rng.integers(0, n_positions))
        if left == right:
            continue
        if left > right:
            left, right = right, left
        selected.add((left, right))

    pair_array = np.asarray(sorted(selected), dtype=np.int64)
    return positions[pair_array[:, 0]], positions[pair_array[:, 1]]


def sample_same_hour_pairs(matched, max_pairs, rng):
    if matched.empty:
        return pd.DataFrame(columns=["left_index", "right_index", "distance_km"])

    group_sizes = matched.groupby(HOUR_COLUMN).size()
    possible_by_hour = (group_sizes * (group_sizes - 1) // 2).loc[lambda values: values > 0]
    total_possible = int(possible_by_hour.sum())

    if total_possible == 0:
        return pd.DataFrame(columns=["left_index", "right_index", "distance_km"])

    left_parts = []
    right_parts = []
    use_all_pairs = max_pairs is None or total_possible <= max_pairs

    for hour, possible_pairs in possible_by_hour.items():
        indices = matched.index[matched[HOUR_COLUMN] == hour].to_numpy()

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

    lat = matched[LATITUDE_COLUMN].to_numpy(dtype=float)
    lon = matched[LONGITUDE_COLUMN].to_numpy(dtype=float)
    distance_km = haversine_km(lat[left_index], lon[left_index], lat[right_index], lon[right_index])

    pairs = pd.DataFrame(
        {
            "left_index": left_index.astype(np.int64),
            "right_index": right_index.astype(np.int64),
            "distance_km": distance_km,
        }
    )

    if use_all_pairs:
        print(f"Using all {len(pairs):,} same-hour spatial pairs")
    else:
        print(f"Sampled {len(pairs):,} same-hour spatial pairs from {total_possible:,} possible pairs")

    return pairs


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


def binned_distance_rows(distances, values, variable, metric_name):
    edges = distance_bin_edges(distances)
    if edges is None:
        return []

    bin_index = np.digitize(distances, edges, right=False) - 1
    rows = []

    for index in range(len(edges) - 1):
        mask = bin_index == index
        if int(mask.sum()) < MIN_PAIRS_PER_BIN:
            continue

        bin_values = values[mask]
        rows.append(
            {
                "variable": variable,
                "label": variable_label(variable),
                "distance_bin": index + 1,
                "distance_start_km": float(edges[index]),
                "distance_end_km": float(edges[index + 1]),
                "distance_mean_km": float(np.mean(distances[mask])),
                "n_pairs": int(mask.sum()),
                metric_name: float(np.mean(bin_values)),
            }
        )

    return rows


def compute_empirical_variogram(matched, variables, pairs):
    if pairs.empty:
        print("Skipping variograms: no same-hour spatial pairs were available")
        return pd.DataFrame()

    left_index = pairs["left_index"].to_numpy(dtype=np.int64)
    right_index = pairs["right_index"].to_numpy(dtype=np.int64)
    distances = pairs["distance_km"].to_numpy(dtype=float)
    rows = []

    for variable in variables:
        if variable not in matched:
            continue

        values = matched[variable].to_numpy(dtype=float)
        left_values = values[left_index]
        right_values = values[right_index]
        mask = np.isfinite(left_values) & np.isfinite(right_values) & np.isfinite(distances)

        if int(mask.sum()) < MIN_PAIRS_TOTAL:
            print(f"Skipping variogram for {variable}: only {int(mask.sum())} usable pairs")
            continue

        semivariance = 0.5 * (left_values[mask] - right_values[mask]) ** 2
        rows.extend(
            binned_distance_rows(
                distances=distances[mask],
                values=semivariance,
                variable=variable,
                metric_name="semivariance",
            )
        )

    return pd.DataFrame(rows)


def plot_variograms(variogram):
    output_path = RESULTS_DIR / "variograms.png"

    if variogram.empty:
        save_blank_figure(output_path, "Empirical variograms", "No variogram bins met sample thresholds")
        return

    variables = list(variogram["variable"].drop_duplicates())
    ncols = 3
    nrows = math.ceil(len(variables) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.8 * nrows), squeeze=False)

    for ax, variable in zip(axes.ravel(), variables):
        subset = variogram[variogram["variable"] == variable]
        ax.plot(
            subset["distance_mean_km"],
            subset["semivariance"],
            marker="o",
            linewidth=1.7,
            color="#7c3aed",
        )
        ax.set_title(variable_label(variable))
        ax.set_xlabel("Distance between matched cells (km)")
        ax.set_ylabel("Semivariance")
        ax.grid(True, alpha=0.25)

    for ax in axes.ravel()[len(variables):]:
        ax.axis("off")

    fig.suptitle("Empirical same-hour spatial variograms", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved {output_path}")


def make_variogram_outputs(matched, covariates, pairs):
    variables = [RADIANCE_COLUMN] + covariates
    variogram = compute_empirical_variogram(matched, variables, pairs)
    save_dataframe(variogram, RESULTS_DIR / "variograms.csv")
    plot_variograms(variogram)
    return variogram


# ---------------------------------------------------------------------------
# 4. Correlation summaries, heatmap, temporal and spatial autocorrelation
# ---------------------------------------------------------------------------


def radiance_correlation_summary(matched, covariates):
    rows = []

    for variable in covariates:
        pair = finite_pair(matched, variable, RADIANCE_COLUMN)
        x = pair[variable]
        y = pair[RADIANCE_COLUMN]

        if (
            len(pair) >= MIN_NON_NULL_ROWS
            and x.nunique(dropna=True) > 1
            and y.nunique(dropna=True) > 1
        ):
            pearson = y.corr(x, method="pearson")
            spearman = y.corr(x, method="spearman")
        else:
            pearson = np.nan
            spearman = np.nan

        rows.append(
            {
                "variable": variable,
                "label": variable_label(variable),
                "n": int(len(pair)),
                "pearson": pearson,
                "spearman": spearman,
                "abs_spearman": abs(spearman) if pd.notna(spearman) else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values("abs_spearman", ascending=False, ignore_index=True)


def correlation_matrix(matched, variables, method="spearman"):
    available = [variable for variable in variables if variable in matched]
    if not available:
        return pd.DataFrame()
    return matched[available].corr(method=method, min_periods=MIN_NON_NULL_ROWS)


def plot_correlation_heatmap(matrix):
    output_path = RESULTS_DIR / "correlation_heatmap.png"

    if matrix.empty:
        save_blank_figure(output_path, "Correlation heatmap", "No variables met correlation thresholds")
        return

    values = np.ma.masked_invalid(matrix.to_numpy(dtype=float))
    n_variables = len(matrix.columns)
    figure_size = max(8.0, 0.52 * n_variables + 4.0)
    fig, ax = plt.subplots(figsize=(figure_size, figure_size))

    cmap = plt.cm.coolwarm.copy()
    cmap.set_bad("#e5e7eb")
    image = ax.imshow(values, cmap=cmap, vmin=-1, vmax=1)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Spearman correlation")

    labels = [variable_label(variable) for variable in matrix.columns]
    ax.set_xticks(np.arange(n_variables), labels=labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(n_variables), labels=labels, fontsize=8)
    ax.set_title("Matched-row Spearman correlation heatmap")

    if n_variables <= 12:
        for i in range(n_variables):
            for j in range(n_variables):
                value = matrix.iloc[i, j]
                if pd.notna(value):
                    color = "white" if abs(value) > 0.55 else "#111827"
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color=color)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved {output_path}")


def compute_temporal_autocorrelation(matched, variables):
    available = [variable for variable in variables if variable in matched]
    if matched.empty or not available:
        return pd.DataFrame()

    hourly = matched.groupby(HOUR_COLUMN)[available].mean().sort_index()
    all_hours = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")
    hourly = hourly.reindex(all_hours)

    rows = []
    for variable in available:
        series = hourly[variable]
        n_non_null = int(series.notna().sum())
        if n_non_null >= MIN_TEMPORAL_PAIRS and series.nunique(dropna=True) > 1:
            rows.append(
                {
                    "variable": variable,
                    "label": variable_label(variable),
                    "lag_hours": 0,
                    "n_pairs": n_non_null,
                    "autocorrelation": 1.0,
                }
            )

        for lag in range(1, MAX_TEMPORAL_LAG_HOURS + 1):
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
                    "variable": variable,
                    "label": variable_label(variable),
                    "lag_hours": lag,
                    "n_pairs": int(len(lagged)),
                    "autocorrelation": float(lagged["current"].corr(lagged["lagged"])),
                }
            )

    return pd.DataFrame(rows)


def compute_spatial_autocorrelation(matched, variables, pairs):
    if pairs.empty:
        print("Skipping spatial autocorrelation: no same-hour spatial pairs were available")
        return pd.DataFrame()

    left_index = pairs["left_index"].to_numpy(dtype=np.int64)
    right_index = pairs["right_index"].to_numpy(dtype=np.int64)
    distances = pairs["distance_km"].to_numpy(dtype=float)
    edges = distance_bin_edges(distances)

    if edges is None:
        return pd.DataFrame()

    distance_bins = np.digitize(distances, edges, right=False) - 1
    rows = []

    for variable in variables:
        if variable not in matched:
            continue

        values = matched[variable].to_numpy(dtype=float)
        left_values = values[left_index]
        right_values = values[right_index]
        usable = np.isfinite(left_values) & np.isfinite(right_values) & np.isfinite(distances)

        if int(usable.sum()) < MIN_PAIRS_TOTAL:
            print(f"Skipping spatial autocorrelation for {variable}: only {int(usable.sum())} usable pairs")
            continue

        for index in range(len(edges) - 1):
            mask = usable & (distance_bins == index)
            if int(mask.sum()) < MIN_PAIRS_PER_BIN:
                continue

            left_bin = left_values[mask]
            right_bin = right_values[mask]
            if np.unique(left_bin).size < 2 or np.unique(right_bin).size < 2:
                continue

            rows.append(
                {
                    "variable": variable,
                    "label": variable_label(variable),
                    "distance_bin": index + 1,
                    "distance_start_km": float(edges[index]),
                    "distance_end_km": float(edges[index + 1]),
                    "distance_mean_km": float(np.mean(distances[mask])),
                    "n_pairs": int(mask.sum()),
                    "spatial_autocorrelation": float(np.corrcoef(left_bin, right_bin)[0, 1]),
                }
            )

    return pd.DataFrame(rows)


def plot_autocorrelation_panels(
    frame,
    x_column,
    y_column,
    output_path,
    title,
    x_label,
    y_label,
):
    if frame.empty:
        save_blank_figure(output_path, title, "No autocorrelation bins met sample thresholds")
        return

    variables = list(frame["variable"].drop_duplicates())
    ncols = 3
    nrows = math.ceil(len(variables) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.8 * nrows), squeeze=False)

    for ax, variable in zip(axes.ravel(), variables):
        subset = frame[frame["variable"] == variable]
        ax.plot(
            subset[x_column],
            subset[y_column],
            marker="o",
            linewidth=1.7,
            color="#0f766e",
        )
        ax.axhline(0, color="#6b7280", linewidth=0.8)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(variable_label(variable))
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.25)

    for ax in axes.ravel()[len(variables):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved {output_path}")


def summarize_correlations_and_autocorrelation(matched, covariates, pairs):
    variables = [RADIANCE_COLUMN] + covariates

    correlations = radiance_correlation_summary(matched, covariates)
    save_dataframe(correlations, RESULTS_DIR / "correlations_vs_radiance.csv")

    matrix = correlation_matrix(matched, variables, method="spearman")
    matrix.to_csv(RESULTS_DIR / "correlation_matrix_spearman.csv")
    print(f"Saved {RESULTS_DIR / 'correlation_matrix_spearman.csv'}")
    plot_correlation_heatmap(matrix)

    temporal = compute_temporal_autocorrelation(matched, variables)
    save_dataframe(temporal, RESULTS_DIR / "temporal_autocorrelation.csv")
    plot_autocorrelation_panels(
        temporal,
        x_column="lag_hours",
        y_column="autocorrelation",
        output_path=RESULTS_DIR / "temporal_autocorrelation.png",
        title="Temporal autocorrelation of matched-row hourly means",
        x_label="Lag (hours)",
        y_label="Autocorrelation",
    )

    spatial = compute_spatial_autocorrelation(matched, variables, pairs)
    save_dataframe(spatial, RESULTS_DIR / "spatial_autocorrelation.csv")
    plot_autocorrelation_panels(
        spatial,
        x_column="distance_mean_km",
        y_column="spatial_autocorrelation",
        output_path=RESULTS_DIR / "spatial_autocorrelation.png",
        title="Same-hour spatial autocorrelation by distance",
        x_label="Distance between matched cells (km)",
        y_label="Spatial autocorrelation",
    )

    return correlations, matrix, temporal, spatial


def main():
    ensure_output_dir()

    if not MERGED_PATH.exists():
        raise FileNotFoundError(f"Merged parquet not found: {MERGED_PATH}")

    parquet_file = pq.ParquetFile(MERGED_PATH)
    schema_names = parquet_schema_names(MERGED_PATH)
    covariates = available_base_covariates(schema_names)

    print(f"Reading merged dataset: {MERGED_PATH}")
    print(f"Total ERA5-left rows from metadata: {parquet_file.metadata.num_rows:,}")
    print(f"Candidate covariates: {', '.join(covariates)}")

    era5_hourly = get_era5_hour_counts(MERGED_PATH, parquet_file)
    matched = load_matched_rows(MERGED_PATH, schema_names, covariates)
    covariates = usable_covariates(matched, covariates)
    print(f"Usable covariates: {', '.join(covariates)}")

    summarize_matched_cells(parquet_file, era5_hourly, matched)
    rng = np.random.default_rng(RANDOM_SEED)
    plot_radiance_vs_covariates(matched, covariates, rng)

    pairs = sample_same_hour_pairs(matched, MAX_PAIR_SAMPLE, rng)
    make_variogram_outputs(matched, covariates, pairs)
    summarize_correlations_and_autocorrelation(matched, covariates, pairs)

    print(f"Exploratory merged outputs written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
