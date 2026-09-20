import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "paths.py").exists():
        sys.path.insert(0, str(_p))
        break
from paths import data_path, results_path

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


# The full CLARA record is shared by every subset, so it stays unscoped.
RESULTS_DIR = results_path("exploratory/clara_full")

DEFAULT_CLARA_PATH = data_path("CLARA.pkl")

TIME_COLUMN = "TimeJD"
RADIANCE_COLUMN = "CLARA_radiance"

DEFAULT_START = "2020-01-01"
DEFAULT_END = "2021-12-31"
DEFAULT_RADIANCE_MIN = 0.0
DEFAULT_RADIANCE_MAX = 500.0
DEFAULT_TREND_DEGREE = 1

MIN_ACF_PAIRS = 6
SERIES_COLOR = "#2563eb"
TREND_COLOR = "#b45309"
ACF_COLOR = "#0f766e"
REFERENCE_COLOR = "#6b7280"

# Granularity name -> (pandas resample rule, matplotlib date format, default max lag).
GRANULARITIES = {
    "daily": ("D", "%Y-%m", 90),
    "weekly": ("W-MON", "%Y-%m", 52),
    "monthly": ("MS", "%Y-%m", 24),
    "yearly": ("YS", "%Y", 3),
}

# Granularities coarse enough that the automatic locator would repeat tick labels.
AXIS_LOCATORS = {
    "monthly": lambda: mdates.MonthLocator(interval=3),
    "yearly": mdates.YearLocator,
}

LAG_UNIT_LABELS = {
    "daily": "days",
    "weekly": "weeks",
    "monthly": "months",
    "yearly": "years",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Explore the full CLARA radiance record: linear/polynomial de-trending "
            "and yearly/monthly/weekly/daily residual series with autocorrelation."
        )
    )
    parser.add_argument("--clara-path", type=Path, default=DEFAULT_CLARA_PATH)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--start",
        type=str,
        default=DEFAULT_START,
        help="First timestamp to keep (inclusive). Use 'none' to keep the whole record.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=DEFAULT_END,
        help=(
            "Last timestamp to keep (inclusive; a bare date keeps that whole day). "
            "Use 'none' to run to the end of the record."
        ),
    )
    parser.add_argument(
        "--radiance-min",
        type=float,
        default=DEFAULT_RADIANCE_MIN,
        help="Drop radiance values at or below this bound (physically implausible OLR).",
    )
    parser.add_argument(
        "--radiance-max",
        type=float,
        default=DEFAULT_RADIANCE_MAX,
        help="Drop radiance values above this bound (calibration/solar-view spikes).",
    )
    parser.add_argument(
        "--no-radiance-filter",
        action="store_true",
        help="Keep every finite radiance value, including negatives and spikes.",
    )
    parser.add_argument(
        "--trend-degree",
        type=int,
        default=DEFAULT_TREND_DEGREE,
        help="Polynomial degree of the de-trending fit (1 = linear).",
    )
    parser.add_argument(
        "--trend-fit",
        choices=("daily", "observations"),
        default="daily",
        help=(
            "Fit the trend on daily means (default, unbiased by sampling density) "
            "or on individual observations."
        ),
    )
    parser.add_argument(
        "--connect-gaps",
        action="store_true",
        help=(
            "Drop empty periods so the plotted lines connect straight across them, "
            "instead of breaking at each gap. Autocorrelation is unaffected either way."
        ),
    )
    for name, (_, _, default_lag) in GRANULARITIES.items():
        parser.add_argument(f"--max-lag-{name}", type=int, default=default_lag)
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
    missing_columns = sorted({TIME_COLUMN, RADIANCE_COLUMN} - set(clara.columns))
    if missing_columns:
        raise ValueError(f"CLARA file is missing columns: {missing_columns}")

    clara = clara[[TIME_COLUMN, RADIANCE_COLUMN]].copy()
    clara[TIME_COLUMN] = pd.to_datetime(clara[TIME_COLUMN])
    clara[RADIANCE_COLUMN] = pd.to_numeric(clara[RADIANCE_COLUMN], errors="coerce")
    return clara.sort_values(TIME_COLUMN, ignore_index=True)


def filter_clara(clara, args):
    """Restrict to the requested window and to plausible radiance, reporting each step."""
    steps = [{"step": "loaded", "n_rows": int(len(clara))}]

    start = None if args.start is None or args.start.lower() == "none" else pd.Timestamp(args.start)
    if args.end is None or args.end.lower() == "none":
        end = None
    else:
        end = pd.Timestamp(args.end)
        if end == end.normalize():
            end = end + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)

    if start is not None:
        clara = clara.loc[clara[TIME_COLUMN] >= start]
        steps.append({"step": f"time >= {start.date()}", "n_rows": int(len(clara))})
    if end is not None:
        clara = clara.loc[clara[TIME_COLUMN] <= end]
        steps.append({"step": f"time <= {end}", "n_rows": int(len(clara))})

    clara = clara.loc[np.isfinite(clara[RADIANCE_COLUMN])]
    steps.append({"step": "finite radiance", "n_rows": int(len(clara))})

    if not args.no_radiance_filter:
        clara = clara.loc[
            (clara[RADIANCE_COLUMN] > args.radiance_min)
            & (clara[RADIANCE_COLUMN] <= args.radiance_max)
        ]
        steps.append(
            {
                "step": f"{args.radiance_min} < radiance <= {args.radiance_max}",
                "n_rows": int(len(clara)),
            }
        )

    if clara.empty:
        raise ValueError("No CLARA rows remain after filtering; relax --start/--radiance bounds.")

    retention = pd.DataFrame(steps)
    retention["fraction_of_loaded"] = retention["n_rows"] / retention.loc[0, "n_rows"]
    return clara.reset_index(drop=True), retention


def decimal_years(timestamps, origin):
    """Time since `origin` in years, the regressor used for the trend fit."""
    seconds = (timestamps - origin).dt.total_seconds().to_numpy(dtype=float)
    return seconds / (365.25 * 24 * 3600)


def fit_trend(clara, degree, trend_fit):
    origin = clara[TIME_COLUMN].min()

    if trend_fit == "daily":
        fit_frame = (
            clara.assign(day=clara[TIME_COLUMN].dt.floor("D"))
            .groupby("day", as_index=False)[RADIANCE_COLUMN]
            .mean()
            .rename(columns={"day": TIME_COLUMN})
        )
    else:
        fit_frame = clara

    fit_x = decimal_years(fit_frame[TIME_COLUMN], origin)
    fit_y = fit_frame[RADIANCE_COLUMN].to_numpy(dtype=float)

    if len(fit_y) <= degree:
        raise ValueError(f"Not enough points ({len(fit_y)}) to fit a degree-{degree} trend.")

    coefficients = np.polyfit(fit_x, fit_y, degree)
    fitted_on_fit_points = np.polyval(coefficients, fit_x)

    observation_x = decimal_years(clara[TIME_COLUMN], origin)
    detrended = clara.assign(
        years_since_start=observation_x,
        trend=np.polyval(coefficients, observation_x),
    )
    detrended["residual"] = detrended[RADIANCE_COLUMN] - detrended["trend"]

    residual_variance = float(np.var(fit_y - fitted_on_fit_points))
    total_variance = float(np.var(fit_y))
    summary = pd.DataFrame(
        [
            {
                "trend_fit": trend_fit,
                "degree": int(degree),
                "n_fit_points": int(len(fit_y)),
                "origin": origin,
                # coefficients[-1] is the intercept, [-2] the linear slope per year, ...
                "intercept": float(coefficients[-1]),
                "slope_per_year": float(coefficients[-2]) if degree >= 1 else np.nan,
                "coefficients_high_to_low_order": ";".join(f"{c:.10g}" for c in coefficients),
                "r_squared": float(1 - residual_variance / total_variance) if total_variance else np.nan,
                "residual_mean": float(detrended["residual"].mean()),
                "residual_std": float(detrended["residual"].std(ddof=1)),
            }
        ]
    )
    return detrended, coefficients, origin, summary


def resample_residuals(detrended, rule):
    """Bucket residuals onto a regular grid; empty buckets stay NaN so gaps stay visible."""
    indexed = detrended.set_index(TIME_COLUMN).sort_index()
    grouped = indexed.resample(rule)
    resampled = pd.DataFrame(
        {
            "n_observations": grouped[RADIANCE_COLUMN].size(),
            "mean_radiance": grouped[RADIANCE_COLUMN].mean(),
            "mean_residual": grouped["residual"].mean(),
            "median_residual": grouped["residual"].median(),
            "std_residual": grouped["residual"].std(ddof=1),
        }
    )
    resampled.index.name = "period_start"
    return resampled


def autocorrelation(series, max_lag):
    """Pearson autocorrelation per lag on pairwise-complete observations."""
    rows = []
    n_non_null = int(series.notna().sum())
    usable_lags = min(max_lag, max(len(series) - 1, 0))

    for lag in range(0, usable_lags + 1):
        if lag == 0:
            if n_non_null < MIN_ACF_PAIRS or series.nunique(dropna=True) < 2:
                continue
            rows.append({"lag": 0, "n_pairs": n_non_null, "autocorrelation": 1.0})
            continue

        paired = pd.concat([series, series.shift(lag)], axis=1).dropna()
        paired.columns = ["current", "lagged"]
        if (
            len(paired) < MIN_ACF_PAIRS
            or paired["current"].nunique() < 2
            or paired["lagged"].nunique() < 2
        ):
            continue

        rows.append(
            {
                "lag": lag,
                "n_pairs": int(len(paired)),
                "autocorrelation": float(paired["current"].corr(paired["lagged"])),
            }
        )

    frame = pd.DataFrame(rows, columns=["lag", "n_pairs", "autocorrelation"])
    if not frame.empty:
        # Bartlett-style white-noise band, widened where a lag has fewer usable pairs.
        frame["significance_band"] = 1.96 / np.sqrt(frame["n_pairs"])
    return frame


def plot_full_series(detrended, coefficients, origin, output_dir, connect_gaps):
    output_path = output_dir / "clara_full_series_with_trend.png"
    daily = (
        detrended.assign(day=detrended[TIME_COLUMN].dt.floor("D"))
        .groupby("day")[RADIANCE_COLUMN]
        .mean()
    )
    if not connect_gaps:
        # Reindexing onto every calendar day inserts NaN, which breaks the line at gaps.
        daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"))

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(daily.index, daily.to_numpy(), color=SERIES_COLOR, linewidth=1.2)

    trend_x = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    trend_y = np.polyval(coefficients, decimal_years(pd.Series(trend_x), origin))
    ax.plot(trend_x, trend_y, color=TREND_COLOR, linewidth=2.0, label="Fitted trend")

    ax.set_title(
        f"CLARA radiance, daily mean, {daily.index.min().date()} to {daily.index.max().date()}"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Radiance")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_granularity(name, resampled, acf, date_format, output_dir, connect_gaps):
    """One figure per granularity: residual series on top, its autocorrelation below."""
    output_path = output_dir / f"clara_residual_{name}.png"
    lag_unit = LAG_UNIT_LABELS[name]
    series = resampled["mean_residual"]
    # Only the plot drops empty periods; `acf` was computed on the regular gapped
    # grid, where a lag of k still means k whole periods.
    plotted = series.dropna() if connect_gaps else series

    if series.notna().sum() == 0:
        save_blank_figure(
            output_path,
            f"CLARA de-trended residuals, {name}",
            "No non-empty periods after filtering",
        )
        return

    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    top, bottom = axes

    marker = "o" if plotted.notna().sum() <= 200 else None
    top.plot(
        plotted.index,
        plotted.to_numpy(dtype=float),
        color=SERIES_COLOR,
        linewidth=1.6,
        marker=marker,
        markersize=4,
    )
    top.axhline(0, color=REFERENCE_COLOR, linewidth=0.9)
    top.set_title(f"De-trended CLARA radiance residuals, {name} mean")
    top.set_xlabel("Period start")
    top.set_ylabel("Mean residual radiance")
    top.grid(True, alpha=0.25)
    if name in AXIS_LOCATORS:
        top.xaxis.set_major_locator(AXIS_LOCATORS[name]())
    top.xaxis.set_major_formatter(mdates.DateFormatter(date_format))

    if acf.empty:
        bottom.axis("off")
        bottom.text(
            0.5,
            0.5,
            f"Only {int(series.notna().sum())} non-empty {name} periods in the record; "
            f"no lag reached the {MIN_ACF_PAIRS}-pair minimum",
            ha="center",
            va="center",
            transform=bottom.transAxes,
        )
        bottom.set_title(f"Autocorrelation of {name} residuals")
    else:
        bottom.vlines(acf["lag"], 0, acf["autocorrelation"], color=ACF_COLOR, linewidth=1.6)
        bottom.plot(acf["lag"], acf["autocorrelation"], "o", color=ACF_COLOR, markersize=4)
        bottom.plot(
            acf["lag"],
            acf["significance_band"],
            color=REFERENCE_COLOR,
            linewidth=1.0,
            linestyle="--",
            label="95% white-noise band",
        )
        bottom.plot(
            acf["lag"],
            -acf["significance_band"],
            color=REFERENCE_COLOR,
            linewidth=1.0,
            linestyle="--",
        )
        bottom.axhline(0, color=REFERENCE_COLOR, linewidth=0.9)
        bottom.set_ylim(-1.05, 1.05)
        bottom.set_title(f"Autocorrelation of {name} residuals")
        bottom.set_xlabel(f"Lag ({lag_unit})")
        bottom.set_ylabel("Autocorrelation")
        bottom.grid(True, alpha=0.25)
        bottom.legend(loc="upper right")

    fig.suptitle(f"CLARA residuals and autocorrelation: {name}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    args = parse_args()
    ensure_output_dir(args.output_dir)

    clara = load_clara(args.clara_path)
    clara, retention = filter_clara(clara, args)
    save_dataframe(retention, args.output_dir / "row_retention.csv")
    print(
        f"Using {len(clara)} CLARA rows from {clara[TIME_COLUMN].min()} to {clara[TIME_COLUMN].max()}"
    )

    detrended, coefficients, origin, trend_summary = fit_trend(
        clara, args.trend_degree, args.trend_fit
    )
    save_dataframe(trend_summary, args.output_dir / "trend_fit_summary.csv")
    plot_full_series(detrended, coefficients, origin, args.output_dir, args.connect_gaps)

    max_lags = {name: getattr(args, f"max_lag_{name}") for name in GRANULARITIES}
    for name, (rule, date_format, _) in GRANULARITIES.items():
        resampled = resample_residuals(detrended, rule)
        save_dataframe(
            resampled.reset_index(), args.output_dir / f"residual_series_{name}.csv"
        )

        acf = autocorrelation(resampled["mean_residual"], max_lags[name])
        acf_out = acf.assign(granularity=name, lag_unit=LAG_UNIT_LABELS[name])
        save_dataframe(acf_out, args.output_dir / f"residual_autocorrelation_{name}.csv")

        plot_granularity(name, resampled, acf, date_format, args.output_dir, args.connect_gaps)

    print(f"Full-record CLARA outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
