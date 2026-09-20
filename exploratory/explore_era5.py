import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "paths.py").exists():
        sys.path.insert(0, str(_p))
        break
from paths import subset_data_path, subset_results_path

from pathlib import Path
import argparse
import math
import os
import tempfile


os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


RESULTS_DIR = subset_results_path("exploratory/era5")

MERGED_PATH = subset_data_path("CLARA_ERA5_merged.parquet")

BATCH_SIZE = 250_000
SAMPLE_SIZE = 200_000
RANDOM_SEED = 42
ZERO_HEAVY_THRESHOLD = 0.50
ZERO_TOLERANCE = 0.0

DERIVED_COVARIATE_DEPENDENCIES = {
    "era5_t2m_c": ["era5_t2m"],
    "era5_d2m_c": ["era5_d2m"],
    "era5_dewpoint_depression": ["era5_t2m", "era5_d2m"],
    "era5_wind10_speed": ["era5_u10", "era5_v10"],
    "era5_cloud_layer_sum": ["era5_lcc", "era5_mcc", "era5_hcc"],
}

VARIABLE_LABELS = {
    "era5_swvl1": "Volumetric soil water layer 1",
    "era5_cape": "CAPE",
    "era5_sp": "Surface pressure",
    "era5_tcwv": "Total column water vapour",
    "era5_sd": "Snow depth",
    "era5_msl": "Mean sea-level pressure",
    "era5_blh": "Boundary layer height",
    "era5_tcc": "Total cloud cover",
    "era5_u10": "10 m east wind",
    "era5_v10": "10 m north wind",
    "era5_t2m": "2 m temperature (K)",
    "era5_d2m": "2 m dewpoint (K)",
    "era5_lcc": "Low cloud cover",
    "era5_mcc": "Medium cloud cover",
    "era5_hcc": "High cloud cover",
    "era5_cp": "Convective precipitation",
    "era5_ssrd": "Surface solar radiation downwards",
    "era5_strd": "Surface thermal radiation downwards",
    "era5_tp": "Total precipitation",
    "era5_i10fg": "10 m wind gust",
    "era5_t2m_c": "2 m temperature (C)",
    "era5_d2m_c": "2 m dewpoint (C)",
    "era5_dewpoint_depression": "Dewpoint depression (K)",
    "era5_wind10_speed": "10 m wind speed",
    "era5_cloud_layer_sum": "Cloud layer sum",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize ERA5 covariate missingness, distributions, and zero-heavy "
            "variables from the merged ERA5-left parquet."
        )
    )
    parser.add_argument("--merged-path", type=Path, default=MERGED_PATH)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--zero-heavy-threshold", type=float, default=ZERO_HEAVY_THRESHOLD)
    parser.add_argument("--zero-tolerance", type=float, default=ZERO_TOLERANCE)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Only scan the first N parquet batches. Useful for quick checks.",
    )
    return parser.parse_args()


def variable_label(variable):
    return VARIABLE_LABELS.get(variable, variable.replace("_", " "))


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


def parquet_schema_names(path):
    return list(pq.ParquetFile(path).schema_arrow.names)


def base_covariate_columns(schema_names):
    return [
        column
        for column in schema_names
        if column.startswith("era5_")
        and not column.endswith("_n_datapoints")
        and column != "era5_n_datapoints"
    ]


def available_covariates(schema_names):
    base_columns = base_covariate_columns(schema_names)
    available = set(base_columns)
    covariates = list(base_columns)

    for variable, dependencies in DERIVED_COVARIATE_DEPENDENCIES.items():
        if all(dependency in available for dependency in dependencies):
            covariates.append(variable)

    return covariates


def scan_columns(schema_names, covariates):
    available = set(schema_names)
    columns = {"hour", "latitude", "longitude"}

    for variable in covariates:
        if variable in available:
            columns.add(variable)
        elif variable in DERIVED_COVARIATE_DEPENDENCIES:
            columns.update(DERIVED_COVARIATE_DEPENDENCIES[variable])

    missing_columns = sorted(columns - available)
    if missing_columns:
        raise ValueError(f"Merged parquet is missing needed ERA5 columns: {missing_columns}")

    return [column for column in schema_names if column in columns]


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


class SpaceTimeMissingness:
    """Exact per-cell and per-hour counts, independent of parquet batch boundaries."""

    def __init__(self, variables):
        self.variables = list(variables)
        self.cells = pd.MultiIndex.from_arrays([[], []], names=["latitude", "longitude"])
        self.hours = pd.DatetimeIndex([], name="hour")
        self.spatial_counts = np.zeros((0, len(variables) + 1), dtype=np.int64)
        self.temporal_counts = np.zeros((0, len(variables) + 1), dtype=np.int64)
        self.limited_scan = False

    def update(self, frame):
        coordinates = frame[["latitude", "longitude"]].to_numpy(dtype=float)
        hours = pd.DatetimeIndex(pd.to_datetime(frame["hour"]))
        if not np.isfinite(coordinates).all() or hours.isna().any():
            raise ValueError("Space/time missingness requires finite coordinates and valid hours.")
        cells = pd.MultiIndex.from_frame(frame[["latitude", "longitude"]])
        new_cells = cells[self.cells.get_indexer(cells) < 0].unique()
        if len(new_cells):
            self.cells = self.cells.append(new_cells)
            self.spatial_counts = np.concatenate([
                self.spatial_counts,
                np.zeros((len(new_cells), len(self.variables) + 1), dtype=np.int64),
            ])
        new_hours = hours[self.hours.get_indexer(hours) < 0].unique()
        if len(new_hours):
            self.hours = self.hours.append(new_hours)
            self.temporal_counts = np.concatenate([
                self.temporal_counts,
                np.zeros((len(new_hours), len(self.variables) + 1), dtype=np.int64),
            ])
        cell_positions = self.cells.get_indexer(cells)
        hour_positions = self.hours.get_indexer(hours)
        missing = ~np.isfinite(frame[self.variables].to_numpy(dtype=float))
        # Most hourly grid batches contain each cell once. Handle repeats too.
        if cells.is_unique:
            self.spatial_counts[cell_positions, 0] += 1
            self.spatial_counts[cell_positions, 1:] += missing
        else:
            np.add.at(self.spatial_counts[:, 0], cell_positions, 1)
            np.add.at(self.spatial_counts[:, 1:], cell_positions, missing)
        self.temporal_counts[:, 0] += np.bincount(hour_positions, minlength=len(self.hours))
        for index in range(len(self.variables)):
            self.temporal_counts[:, index + 1] += np.bincount(
                hour_positions[missing[:, index]], minlength=len(self.hours)
            )

    def spatial_frame(self, index):
        frame = self.cells.to_frame(index=False)
        frame["variable"] = self.variables[index]
        frame["n_rows"] = self.spatial_counts[:, 0]
        frame["n_missing"] = self.spatial_counts[:, index + 1]
        frame["missing_rate"] = frame["n_missing"] / frame["n_rows"]
        frame["limited_scan"] = self.limited_scan
        return frame

    def temporal_frame(self):
        frames = []
        for index, variable in enumerate(self.variables):
            frame = pd.DataFrame({
                "hour": self.hours,
                "variable": variable,
                "n_rows": self.temporal_counts[:, 0],
                "n_missing": self.temporal_counts[:, index + 1],
            })
            frame["missing_rate"] = frame["n_missing"] / frame["n_rows"]
            frame["limited_scan"] = self.limited_scan
            frames.append(frame)
        return pd.concat(frames, ignore_index=True).sort_values(["variable", "hour"])


class CovariateAccumulator:
    def __init__(self, variable, sample_size, rng):
        self.variable = variable
        self.sample_size = max(0, int(sample_size))
        self.rng = rng

        self.n_rows = 0
        self.n_missing = 0
        self.n_non_missing = 0
        self.n_zero = 0
        self.n_negative = 0
        self.n_positive = 0
        self.value_min = np.nan
        self.value_max = np.nan
        self.value_sum = 0.0
        self.value_sum_sq = 0.0

        self._sample = []
        self._sample_seen = 0

    def update(self, values, zero_tolerance):
        values = np.asarray(values, dtype=float)
        self.n_rows += int(len(values))

        finite = np.isfinite(values)
        finite_values = values[finite]
        self.n_missing += int(len(values) - len(finite_values))

        if len(finite_values) == 0:
            return

        self.n_non_missing += int(len(finite_values))
        if zero_tolerance > 0:
            zero_mask = np.abs(finite_values) <= zero_tolerance
        else:
            zero_mask = finite_values == 0

        self.n_zero += int(np.count_nonzero(zero_mask))
        self.n_negative += int(np.count_nonzero(finite_values < 0))
        self.n_positive += int(np.count_nonzero(finite_values > 0))
        self.value_min = float(np.nanmin(finite_values)) if np.isnan(self.value_min) else min(
            self.value_min, float(np.nanmin(finite_values))
        )
        self.value_max = float(np.nanmax(finite_values)) if np.isnan(self.value_max) else max(
            self.value_max, float(np.nanmax(finite_values))
        )

        finite_values = finite_values.astype(np.float64, copy=False)
        self.value_sum += float(np.sum(finite_values, dtype=np.float64))
        self.value_sum_sq += float(np.sum(finite_values * finite_values, dtype=np.float64))
        self._update_sample(finite_values)

    def _update_sample(self, values):
        if self.sample_size == 0:
            self._sample_seen += int(len(values))
            return

        capacity = self.sample_size - len(self._sample)
        if capacity > 0:
            take = min(capacity, len(values))
            self._sample.extend(values[:take].tolist())
            self._sample_seen += int(take)
            values = values[take:]

        if len(values) == 0:
            return

        counts = np.arange(
            self._sample_seen + 1,
            self._sample_seen + len(values) + 1,
            dtype=np.float64,
        )
        keep_probability = np.minimum(1.0, self.sample_size / counts)
        keep = self.rng.random(len(values)) < keep_probability
        if np.any(keep):
            slots = self.rng.integers(0, self.sample_size, size=int(np.count_nonzero(keep)))
            for slot, value in zip(slots, values[keep]):
                self._sample[int(slot)] = float(value)

        self._sample_seen += int(len(values))

    def sample(self):
        return np.asarray(self._sample, dtype=float)

    def to_row(self, zero_heavy_threshold):
        if self.n_non_missing:
            mean = self.value_sum / self.n_non_missing
            if self.n_non_missing > 1:
                variance = (self.value_sum_sq - (self.value_sum * self.value_sum / self.n_non_missing)) / (
                    self.n_non_missing - 1
                )
                std = math.sqrt(max(0.0, variance))
            else:
                std = np.nan

            sample = self.sample()
            if len(sample):
                quantiles = np.quantile(sample, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
            else:
                quantiles = [np.nan] * 7
        else:
            mean = np.nan
            std = np.nan
            sample = np.array([], dtype=float)
            quantiles = [np.nan] * 7

        missing_rate = self.n_missing / self.n_rows if self.n_rows else np.nan
        zero_rate_all = self.n_zero / self.n_rows if self.n_rows else np.nan
        zero_rate_non_missing = self.n_zero / self.n_non_missing if self.n_non_missing else np.nan

        return {
            "variable": self.variable,
            "label": variable_label(self.variable),
            "n_rows": int(self.n_rows),
            "n_missing": int(self.n_missing),
            "missing_rate": missing_rate,
            "n_non_missing": int(self.n_non_missing),
            "n_zero": int(self.n_zero),
            "zero_rate_all_rows": zero_rate_all,
            "zero_rate_non_missing": zero_rate_non_missing,
            "zero_heavy": bool(
                pd.notna(zero_rate_non_missing) and zero_rate_non_missing >= zero_heavy_threshold
            ),
            "n_negative": int(self.n_negative),
            "n_positive": int(self.n_positive),
            "min": self.value_min,
            "p01_sample": float(quantiles[0]),
            "p05_sample": float(quantiles[1]),
            "p25_sample": float(quantiles[2]),
            "median_sample": float(quantiles[3]),
            "mean": mean,
            "std": std,
            "p75_sample": float(quantiles[4]),
            "p95_sample": float(quantiles[5]),
            "p99_sample": float(quantiles[6]),
            "max": self.value_max,
            "sample_count": int(len(sample)),
        }


def summarize_covariates(path, covariates, columns, args):
    dataset = ds.dataset(path, format="parquet")
    scanner = dataset.scanner(columns=columns, batch_size=args.batch_size,
                              batch_readahead=2, fragment_readahead=1)
    accumulators = {
        variable: CovariateAccumulator(
            variable=variable,
            sample_size=args.sample_size,
            rng=np.random.default_rng(args.random_seed + index),
        )
        for index, variable in enumerate(covariates)
    }
    space_time = SpaceTimeMissingness(covariates)
    space_time.limited_scan = args.max_batches is not None

    batches_scanned = 0
    for batch_number, batch in enumerate(scanner.to_batches(), start=1):
        frame = batch.to_pandas()
        frame = add_derived_covariates(frame)
        space_time.update(frame)

        for variable in covariates:
            if variable in frame:
                values = frame[variable].to_numpy(dtype=float, copy=False)
            else:
                values = np.full(len(frame), np.nan)
            accumulators[variable].update(values, args.zero_tolerance)

        batches_scanned = batch_number
        if batch_number % 10 == 0:
            print(f"Scanned {batch_number:,} parquet batches")

        if args.max_batches is not None and batch_number >= args.max_batches:
            print(f"Stopped after --max-batches={args.max_batches}")
            break

    rows = [accumulators[variable].to_row(args.zero_heavy_threshold) for variable in covariates]
    summary = pd.DataFrame(rows)
    summary.insert(2, "batches_scanned", batches_scanned)
    summary.insert(3, "limited_scan", args.max_batches is not None)

    samples = {variable: accumulators[variable].sample() for variable in covariates}
    return summary, samples, space_time


def save_space_time_missingness(space_time, output_dir):
    temporal = space_time.temporal_frame()
    save_dataframe(temporal, output_dir / "covariate_missingness_by_hour.csv")
    output_path = output_dir / "covariate_missingness_by_location.parquet"
    # Write one variable at a time to avoid materializing the full long table.
    writer = None
    try:
        for index in range(len(space_time.variables)):
            table = pa.Table.from_pandas(space_time.spatial_frame(index), preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    print(f"Saved {output_path}")


def plot_space_time_missingness(space_time, output_dir):
    suffix = " (partial scan)" if space_time.limited_scan else ""
    temporal = space_time.temporal_frame().pivot(
        index="variable", columns="hour", values="missing_rate"
    ).reindex(space_time.variables).sort_index(axis=1)
    if temporal.empty:
        save_blank_figure(output_dir / "covariate_missingness_by_hour.png",
                          "ERA5 missingness over time", "No rows scanned")
        return

    fig, ax = plt.subplots(figsize=(16, max(5, 0.32 * len(temporal) + 2)), constrained_layout=True)
    image = ax.imshow(temporal.to_numpy(), aspect="auto", interpolation="nearest",
                      vmin=0, vmax=1, cmap="magma_r")
    ticks = np.unique(np.linspace(0, len(temporal.columns) - 1, min(10, len(temporal.columns))).astype(int))
    ax.set_xticks(ticks, temporal.columns[ticks].strftime("%Y-%m-%d\n%H:%M"))
    ax.set_yticks(np.arange(len(temporal)), temporal.index)
    ax.set_xlabel("ERA5 hour (UTC); each column is one observed hour")
    ax.set_title("ERA5 missing fraction across grid cells, by hour" + suffix)
    fig.colorbar(image, ax=ax, label="Missing fraction (0 = complete; 1 = all missing)")
    output_path = output_dir / "covariate_missingness_by_hour.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved {output_path}")

    # Paginate native-grid maps to keep every variable readable.
    per_page = 12
    hours = temporal.columns
    for start in range(0, len(space_time.variables), per_page):
        count = min(per_page, len(space_time.variables) - start)
        fig, axes = plt.subplots(math.ceil(count / 3), 3, squeeze=False,
                                 figsize=(18, 3.3 * math.ceil(count / 3) + 1),
                                 constrained_layout=True)
        for ax, index in zip(axes.ravel(), range(start, start + count)):
            frame = space_time.spatial_frame(index)
            grid = frame.pivot(index="latitude", columns="longitude", values="missing_rate")
            grid = grid.sort_index().sort_index(axis=1)
            # Cell-center coordinates determine pixel edges on the ERA5 regular grid.
            dx = float(np.median(np.diff(grid.columns))) if len(grid.columns) > 1 else 0.25
            dy = float(np.median(np.diff(grid.index))) if len(grid.index) > 1 else 0.25
            extent = [grid.columns.min() - dx / 2, grid.columns.max() + dx / 2,
                      grid.index.min() - dy / 2, grid.index.max() + dy / 2]
            cmap = plt.get_cmap("magma_r").copy()
            cmap.set_bad("#d1d5db")
            image = ax.imshow(grid.to_numpy(), origin="lower", extent=extent,
                              aspect="auto", interpolation="nearest", vmin=0, vmax=1, cmap=cmap)
            ax.set_title(space_time.variables[index], fontsize=10)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
        for ax in axes.ravel()[count:]:
            ax.set_visible(False)
        fig.colorbar(image, ax=list(axes.ravel()[:count]), shrink=0.85,
                     label="Missing fraction over scanned hours (grey = no scanned rows)")
        fig.suptitle("ERA5 spatial missingness" + suffix + "\n"
                     + f"{hours.min():%Y-%m-%d %H:%M} to {hours.max():%Y-%m-%d %H:%M} UTC; "
                     + "NaN/infinite values are missing; zeros are valid", fontsize=13)
        output_path = output_dir / f"covariate_missingness_spatial_{start // per_page + 1:02d}.png"
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        print(f"Saved {output_path}")


def plot_missingness(summary, output_dir):
    output_path = output_dir / "covariate_missingness_and_zero_rates.png"

    if summary.empty:
        save_blank_figure(output_path, "ERA5 covariate missingness", "No ERA5 covariates found")
        return

    plot_data = summary.sort_values("missing_rate", ascending=True).reset_index(drop=True)
    y = np.arange(len(plot_data))
    labels = plot_data["variable"].to_numpy()

    fig, ax = plt.subplots(figsize=(12, max(6, 0.36 * len(plot_data) + 2)))
    ax.barh(y - 0.18, plot_data["missing_rate"], height=0.36, color="#dc2626", label="Missing rate")
    ax.barh(
        y + 0.18,
        plot_data["zero_rate_non_missing"],
        height=0.36,
        color="#2563eb",
        label="Zero rate among non-missing",
    )
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Rate")
    ax.set_title("ERA5 covariate missingness and zero mass")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="lower right")

    for index, row in plot_data.iterrows():
        if row["zero_heavy"]:
            ax.text(
                min(0.98, row["zero_rate_non_missing"] + 0.02),
                index + 0.18,
                "zero-heavy",
                va="center",
                fontsize=8,
                color="#1e3a8a",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved {output_path}")


def axis_title(row):
    title = row["label"]
    if row["zero_heavy"]:
        title = f"{title} (zero-heavy)"
    return title


def plot_covariate_distributions(summary, samples, output_dir):
    output_path = output_dir / "covariate_distributions.png"

    if summary.empty:
        save_blank_figure(output_path, "ERA5 covariate distributions", "No ERA5 covariates found")
        return

    ncols = 4
    nrows = math.ceil(len(summary) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, max(4, 3.1 * nrows)), squeeze=False)

    for ax, (_, row) in zip(axes.ravel(), summary.iterrows()):
        variable = row["variable"]
        values = samples.get(variable, np.array([], dtype=float))

        if len(values) == 0:
            ax.text(0.5, 0.5, "No finite values", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(axis_title(row), fontsize=9)
            ax.axis("off")
            continue

        ax.hist(values, bins=50, color="#2563eb", edgecolor="white", alpha=0.82)
        if row["n_zero"] > 0:
            ax.axvline(0, color="#dc2626", linewidth=1.1, alpha=0.9)
        if row["zero_heavy"]:
            ax.set_yscale("log")

        ax.set_title(axis_title(row), fontsize=9)
        ax.set_xlabel(variable, fontsize=8)
        ax.set_ylabel("Sample count", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(True, axis="y", alpha=0.2)
        ax.text(
            0.98,
            0.95,
            f"missing {row['missing_rate']:.1%}\nzero {row['zero_rate_non_missing']:.1%}",
            ha="right",
            va="top",
            fontsize=7,
            transform=ax.transAxes,
            bbox={"facecolor": "white", "edgecolor": "0.85", "alpha": 0.85, "pad": 2},
        )

    for ax in axes.ravel()[len(summary) :]:
        ax.axis("off")

    fig.suptitle("ERA5 covariate distributions from reservoir samples", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    args = parse_args()
    ensure_output_dir(args.output_dir)

    if not args.merged_path.exists():
        raise FileNotFoundError(f"Merged parquet not found: {args.merged_path}")

    schema_names = parquet_schema_names(args.merged_path)
    covariates = available_covariates(schema_names)
    if not covariates:
        raise ValueError("No ERA5 covariates found in the merged parquet.")
    columns = scan_columns(schema_names, covariates)

    print(f"Reading merged ERA5-left parquet: {args.merged_path}")
    print(f"Covariates: {', '.join(covariates)}")
    print(f"Scanning columns: {', '.join(columns)}")

    summary, samples, space_time = summarize_covariates(args.merged_path, covariates, columns, args)
    summary = summary.sort_values("variable", ignore_index=True)

    missingness = summary[
        [
            "variable",
            "label",
            "n_rows",
            "n_missing",
            "missing_rate",
            "n_non_missing",
        ]
    ]
    save_dataframe(missingness, args.output_dir / "covariate_missingness.csv")
    save_dataframe(summary, args.output_dir / "covariate_distribution_summary.csv")

    zero_heavy = summary[summary["zero_heavy"]].sort_values(
        "zero_rate_non_missing", ascending=False, ignore_index=True
    )
    save_dataframe(zero_heavy, args.output_dir / "zero_heavy_covariates.csv")

    plot_missingness(summary, args.output_dir)
    plot_covariate_distributions(summary, samples, args.output_dir)
    save_space_time_missingness(space_time, args.output_dir)
    plot_space_time_missingness(space_time, args.output_dir)

    print(f"Exploratory ERA5 outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
