"""Explore how the raw CLARA record is distributed in time and in quality.

exploratory/explore_clara_full.py covers the radiance series itself: trend,
residuals, autocorrelation. This script covers the two things the raw delivery
made newly relevant and that the old pre-filtered CLARA.pkl hid:

  * where the record actually has data, day by day, over six years;
  * how much of it is an Earth view rather than a calibration view, and how that
    fraction drifts.

Spatial homogeneity is deliberately absent. It needs CLARA_fov_latitude and
CLARA_fov_longitude, which come from the geolocation CSVs; until those are
delivered the record has no footprints and any "spatial" plot would be fiction.
exploratory/explore_clara.py draws those views once a subset has been matched.

    python3 exploratory/explore_clara_coverage.py
"""

import sys
from pathlib import Path

for _p in Path(__file__).resolve().parents:
    if (_p / "paths.py").exists():
        sys.path.insert(0, str(_p))
        break
from paths import results_path

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

from read_in.clara_source import (
    FULL_PATH,
    RADIANCE_COLUMN,
    TIME_COLUMN,
    load_clara_full,
)

RESULTS_DIR = results_path("exploratory/clara_full")

# Same bounds the merge step applies, so "Earth view" here means the same thing
# it means in read_in/temporal_subset/data_preprocessing.py.
RADIANCE_MIN = 0.0
RADIANCE_MAX = 500.0

# The repo's existing figure palette, reused so these plots sit beside the ones
# explore_clara_full.py already writes. The blue/amber pair passes the
# colourblind-separation checks (worst adjacent dE 31.3 protan, 34.6 normal).
KEPT_COLOR = "#2563eb"
REJECTED_COLOR = "#b45309"
GRID_COLOR = "#d4d4d8"
INK_MUTED = "#6b7280"
SEQUENTIAL_CMAP = "Blues"

# Each monthly save file spills a few samples past midnight into the next month,
# so a month the instrument never observed can still hold a handful of rows. At
# 30 s cadence this is under three hours of sampling: too little to be a monthly
# statistic, and plotted raw it draws a spurious dive to 0% at every gap edge.
MIN_SAMPLES_PER_MONTH = 300


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clara-path", type=Path, default=FULL_PATH)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def recede(ax):
    """Push grid and spines behind the data, where they belong."""
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=INK_MUTED, labelcolor="black")


def save(fig, output_dir, name):
    path = Path(output_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    return path


def plot_coverage_calendar(clara, output_dir):
    """Samples per day as a year-by-day-of-year grid.

    Magnitude over an ordered two-dimensional key, so a single-hue sequential
    ramp, not a rainbow. Days with no data stay unfilled rather than being drawn
    as zero, so a gap reads as absence instead of as a measurement of nothing.
    """
    day = clara[TIME_COLUMN].dt.normalize()
    per_day = day.value_counts().sort_index()

    years = np.arange(per_day.index.year.min(), per_day.index.year.max() + 1)
    grid = np.full((len(years), 366), np.nan)
    for timestamp, count in per_day.items():
        grid[timestamp.year - years[0], timestamp.dayofyear - 1] = count

    fig, ax = plt.subplots(figsize=(13, 2.6 + 0.42 * len(years)))
    # Cell *edges*, one more than there are rows and columns. Passing centres
    # instead silently shifts every row half a cell against its tick label.
    mesh = ax.pcolormesh(
        np.arange(1, 368) - 0.5,
        np.append(years, years[-1] + 1) - 0.5,
        grid,
        cmap=SEQUENTIAL_CMAP, shading="flat", vmin=0,
    )
    ax.set_yticks(years)
    ax.invert_yaxis()

    month_starts = pd.date_range("2020-01-01", "2020-12-01", freq="MS").dayofyear
    ax.set_xticks(month_starts)
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_xlim(1, 366)

    bar = fig.colorbar(mesh, ax=ax, pad=0.015)
    bar.set_label("OLR samples that day", color=INK_MUTED)
    bar.outline.set_visible(False)

    covered = int(per_day.gt(0).sum())
    span = f"{per_day.index.min():%Y-%m-%d} to {per_day.index.max():%Y-%m-%d}"
    ax.set_title(
        f"CLARA sampling calendar — {covered:,} days with data, {span}\n"
        f"Blank cells are days the record does not cover",
        loc="left", fontsize=11,
    )
    ax.tick_params(colors=INK_MUTED, labelcolor="black")
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    return save(fig, output_dir, "clara_coverage_calendar.png")


def plot_radiance_population(clara, output_dir):
    """Where the radiance actually sits, and how far the calibration cluster is.

    Two panels rather than one: the Earth-view population and the calibration
    cluster are five orders of magnitude apart, and a single axis that holds both
    renders the part we model as one bar.
    """
    values = clara[RADIANCE_COLUMN].to_numpy()
    kept = (values > RADIANCE_MIN) & (values <= RADIANCE_MAX)

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.2))

    window = values[(values > -200) & (values < 700)]
    bins = np.linspace(-200, 700, 180)
    left.hist(window[(window > RADIANCE_MIN) & (window <= RADIANCE_MAX)],
              bins=bins, color=KEPT_COLOR, label=f"kept: ({RADIANCE_MIN:g}, {RADIANCE_MAX:g}]")
    left.hist(window[(window <= RADIANCE_MIN) | (window > RADIANCE_MAX)],
              bins=bins, color=REJECTED_COLOR, label="rejected")
    left.axvline(RADIANCE_MAX, color=INK_MUTED, linewidth=1, linestyle="--")
    left.set_xlabel("OLR (W m$^{-2}$ sr$^{-1}$)")
    left.set_ylabel("samples")
    left.set_title("Earth-view range", loc="left", fontsize=11)
    left.legend(frameon=False, fontsize=9)
    recede(left)

    # Log-modulus keeps the negatives visible; a plain log axis would drop them.
    transformed = np.sign(values) * np.log10(1.0 + np.abs(values))
    right.hist(transformed[kept], bins=200, color=KEPT_COLOR)
    right.hist(transformed[~kept], bins=200, color=REJECTED_COLOR)
    right.set_yscale("log")
    right.set_xlabel("sign(OLR) · log$_{10}$(1 + |OLR|)")
    right.set_ylabel("samples (log)")
    right.set_title("Full range — the calibration cluster near 8.7e4", loc="left", fontsize=11)
    recede(right)

    fig.suptitle(
        f"CLARA radiance population — {kept.sum():,} of {len(values):,} samples "
        f"({100 * kept.mean():.1f}%) are Earth views",
        fontsize=12, x=0.008, ha="left",
    )
    fig.tight_layout()
    return save(fig, output_dir, "clara_radiance_population.png")


def daily_summary(clara):
    """Per-day counts and Earth-view radiance quantiles.

    explore_clara_full.py writes a daily mean, but only of the de-trended
    residuals. This is the level the raw series is actually read at.
    """
    day = clara[TIME_COLUMN].dt.normalize()
    values = clara[RADIANCE_COLUMN]
    kept = (values > RADIANCE_MIN) & (values <= RADIANCE_MAX)

    frame = pd.DataFrame({"day": day, "radiance": values, "kept": kept})
    grouped = frame.groupby("day", observed=True)
    summary = pd.DataFrame({
        "n_samples": grouped.size(),
        "n_earth_view": grouped["kept"].sum(),
    })

    earth = frame[frame["kept"]].groupby("day", observed=True)["radiance"]
    summary["mean"] = earth.mean()
    summary["q25"] = earth.quantile(0.25)
    summary["median"] = earth.median()
    summary["q75"] = earth.quantile(0.75)

    full = pd.date_range(day.min(), day.max(), freq="D")
    return summary.reindex(full).rename_axis("day").reset_index()


def plot_raw_series(clara, daily, output_dir):
    """Every Earth-view sample over time, as a density, with the daily median.

    A scatter of 1.4 M points is a black rectangle, and a daily mean alone hides
    that the population is wide and skewed. Binning the observations and shading
    by count shows the whole distribution at every date, which is what "the raw
    series" is actually being asked for; the daily median then rides on top of
    the density instead of standing in for it.
    """
    values = clara[RADIANCE_COLUMN]
    kept = (values > RADIANCE_MIN) & (values <= RADIANCE_MAX)
    times = clara.loc[kept, TIME_COLUMN]

    fig, ax = plt.subplots(figsize=(13, 5))
    counts = ax.hexbin(
        mdates.date2num(times), values[kept],
        gridsize=(320, 90), cmap=SEQUENTIAL_CMAP, bins="log", mincnt=1, linewidths=0,
    )
    ax.xaxis_date()

    ax.plot(daily["day"], daily["median"], color=REJECTED_COLOR, linewidth=1.2,
            label="daily median")
    ax.set_ylim(0, RADIANCE_MAX)
    ax.set_ylabel("OLR (W m$^{-2}$ sr$^{-1}$)")
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    bar = fig.colorbar(counts, ax=ax, pad=0.012)
    bar.set_label("samples per bin (log)", color=INK_MUTED)
    bar.outline.set_visible(False)

    ax.set_title(
        f"Every CLARA Earth-view sample, {kept.sum():,} observations at 30 s cadence\n"
        "Shading is observation density; the line is the daily median",
        loc="left", fontsize=11,
    )
    recede(ax)
    fig.tight_layout()
    return save(fig, output_dir, "clara_raw_series.png")


def plot_daily_mean(daily, output_dir):
    """Daily mean and median with the interquartile band, Earth views only."""
    fig, ax = plt.subplots(figsize=(13, 4.2))

    ax.fill_between(daily["day"], daily["q25"], daily["q75"], color=KEPT_COLOR,
                    alpha=0.18, linewidth=0, label="interquartile range")
    ax.plot(daily["day"], daily["median"], color=KEPT_COLOR, linewidth=1.1,
            label="daily median")
    ax.plot(daily["day"], daily["mean"], color=REJECTED_COLOR, linewidth=1.1,
            label="daily mean")

    ax.set_ylabel("OLR (W m$^{-2}$ sr$^{-1}$)")
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper right")
    ax.set_title(
        "Daily CLARA Earth-view radiance\n"
        "The skew reverses: mean below median on 89% of days to 2021, above on "
        "90% from 2022",
        loc="left", fontsize=11,
    )
    recede(ax)
    fig.tight_layout()
    return save(fig, output_dir, "clara_daily_series.png")


def monthly_summary(clara):
    """Per-month sample counts, Earth-view fraction and radiance quartiles."""
    month = clara[TIME_COLUMN].dt.to_period("M")
    values = clara[RADIANCE_COLUMN]
    kept = (values > RADIANCE_MIN) & (values <= RADIANCE_MAX)

    frame = pd.DataFrame({"month": month, "radiance": values, "kept": kept})
    grouped = frame.groupby("month", observed=True)

    summary = pd.DataFrame({
        "n_samples": grouped.size(),
        "n_earth_view": grouped["kept"].sum(),
        "n_days": frame.assign(day=clara[TIME_COLUMN].dt.normalize())
                       .groupby("month", observed=True)["day"].nunique(),
    })
    summary["earth_view_fraction"] = summary["n_earth_view"] / summary["n_samples"]

    earth = frame[frame["kept"]].groupby("month", observed=True)["radiance"]
    summary["q25"] = earth.quantile(0.25)
    summary["median"] = earth.median()
    summary["q75"] = earth.quantile(0.75)

    full = pd.period_range(month.min(), month.max(), freq="M")
    summary = summary.reindex(full).rename_axis("month").reset_index()

    # Keep the raw counts, but blank the derived statistics for months that are
    # only boundary spill, so the plots break the line instead of reporting a
    # month-long figure computed from a few minutes of data.
    thin = summary["n_samples"].fillna(0) < MIN_SAMPLES_PER_MONTH
    summary["is_thin"] = thin
    summary.loc[thin, ["earth_view_fraction", "q25", "median", "q75"]] = np.nan
    return summary


def plot_earth_view_fraction(summary, output_dir):
    """The share of each month that is an Earth view.

    One series, so no legend box: the title names it. Months absent from the
    delivery break the line rather than being interpolated across.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    x = summary["month"].dt.to_timestamp()

    ax.plot(x, 100 * summary["earth_view_fraction"], color=KEPT_COLOR, linewidth=2)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of samples in (0, 500]")
    ax.set_title(
        "Share of CLARA samples that are Earth views, by month\n"
        f"Gaps are months with no data or under {MIN_SAMPLES_PER_MONTH} samples; "
        "the rise from 2023 is the instrument, not the filter",
        loc="left", fontsize=11,
    )
    recede(ax)
    fig.tight_layout()
    return save(fig, output_dir, "clara_earth_view_fraction.png")


def plot_monthly_radiance(summary, output_dir):
    """Monthly median Earth-view radiance with the interquartile band."""
    fig, ax = plt.subplots(figsize=(12, 4))
    x = summary["month"].dt.to_timestamp()

    ax.fill_between(x, summary["q25"], summary["q75"], color=KEPT_COLOR,
                    alpha=0.18, linewidth=0)
    ax.plot(x, summary["median"], color=KEPT_COLOR, linewidth=2)
    ax.set_ylabel("OLR (W m$^{-2}$ sr$^{-1}$)")
    ax.set_title(
        "Monthly median CLARA Earth-view radiance, with the interquartile band\n"
        f"Earth views only (0, 500]; gaps are months with no data or under "
        f"{MIN_SAMPLES_PER_MONTH} samples",
        loc="left", fontsize=11,
    )
    recede(ax)
    fig.tight_layout()
    return save(fig, output_dir, "clara_monthly_radiance.png")


def plot_sampling_by_hour(clara, output_dir):
    """When in the UTC day CLARA sampled, one panel per year.

    Small multiples rather than six overlaid lines: the years are an ordered key
    and the question is the shape of each, not a comparison of six values at a
    point.
    """
    hours = (
        clara[TIME_COLUMN] - clara[TIME_COLUMN].dt.normalize()
    ) / pd.Timedelta("1h")
    years = np.sort(clara[TIME_COLUMN].dt.year.unique())

    columns = 3
    rows = int(np.ceil(len(years) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(12, 2.6 * rows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, year in zip(axes, years):
        ax.hist(hours[clara[TIME_COLUMN].dt.year == year],
                bins=np.arange(0, 25, 0.5), color=KEPT_COLOR)
        ax.set_title(str(year), loc="left", fontsize=10)
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        recede(ax)
    for ax in axes[len(years):]:
        ax.set_visible(False)

    for ax in axes[-columns:]:
        ax.set_xlabel("UTC hour of day")
    for index in range(0, len(axes), columns):
        axes[index].set_ylabel("samples")

    fig.suptitle(
        "When CLARA sampled, by UTC hour of day\n"
        "Uneven coverage here becomes uneven coverage of the diurnal cycle",
        fontsize=12, x=0.008, ha="left",
    )
    fig.tight_layout()
    return save(fig, output_dir, "clara_sampling_by_hour.png")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    clara = load_clara_full(
        args.clara_path,
        require_geolocation=False,
        require_complete=False,
        columns=[TIME_COLUMN, RADIANCE_COLUMN],
    ).sort_values(TIME_COLUMN, ignore_index=True)

    print(f"{len(clara):,} samples "
          f"{clara[TIME_COLUMN].min()} .. {clara[TIME_COLUMN].max()}")

    output_dir.mkdir(parents=True, exist_ok=True)

    daily = daily_summary(clara)
    daily_path = output_dir / "clara_daily_summary.csv"
    daily.to_csv(daily_path, index=False)
    print(f"Saved {daily_path}")

    summary = monthly_summary(clara)
    summary_path = output_dir / "clara_monthly_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")

    plot_raw_series(clara, daily, output_dir)
    plot_daily_mean(daily, output_dir)
    plot_coverage_calendar(clara, output_dir)
    plot_radiance_population(clara, output_dir)
    plot_earth_view_fraction(summary, output_dir)
    plot_monthly_radiance(summary, output_dir)
    plot_sampling_by_hour(clara, output_dir)

    print()
    print("Spatial homogeneity needs CLARA_fov_latitude/longitude, which arrive "
          "with the geolocation CSVs. Until then use exploratory/explore_clara.py "
          "on a matched subset for the spatial views.")
    print(f"Coverage outputs written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
