import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "paths.py").exists():
        sys.path.insert(0, str(_p))
        break
from paths import subset_data_path, subset_results_path

import os
import tempfile


os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
if not os.environ.get("LOKY_MAX_CPU_COUNT"):
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


try:
    from covariate_preprocessing import (
        LOCAL_TIME_COLUMN, LOG_RADIANCE_COLUMN, add_local_time,
        add_log_radiance, make_covariate_preprocessor,
    )
    from fold_config import MIN_TEST_ROWS, MIN_TRAIN_ROWS, N_SPACE_FOLDS, N_TIME_FOLDS
except ModuleNotFoundError:
    from modelling.covariate_preprocessing import (
        LOCAL_TIME_COLUMN, LOG_RADIANCE_COLUMN, add_local_time,
        add_log_radiance, make_covariate_preprocessor,
    )
    from modelling.fold_config import MIN_TEST_ROWS, MIN_TRAIN_ROWS, N_SPACE_FOLDS, N_TIME_FOLDS


RANDOM_SEED = 14122

VALIDATION_STRATEGY = "space_time_blocked"

N_ESTIMATORS = 500
MIN_SAMPLES_LEAF = 5
MAX_FEATURES = "sqrt"

DATA_PATH = subset_data_path("CLARA_ERA5_merged.parquet")

OUT_DIR = subset_results_path("modelling/rf")

Y_COLUMN = "clara_radiance_hourly_mean"
COUNT_COLUMN = "clara_n_datapoints"
TIME_COLUMN = "hour"
LAT_COLUMN = "latitude"
LON_COLUMN = "longitude"
SPACE_FOLD_COLUMN = "space_fold"
TIME_FOLD_COLUMN = "time_fold"
FOLD_ID_COLUMN = "fold_id"
IMPORTANCE_COLUMN = "impurity_importance"

PREFERRED_COVARIATES = [
    "era5_swvl1",
    "era5_stl1",
    "era5_cl",
    "era5_cvl",
    "era5_cvh",
    "era5_tvl",
    "era5_tvh",
    "era5_siconc",
    "era5_asn",
    "era5_sst",
    "era5_slt",
    "era5_lai_lv",
    "era5_lai_hv",
    "era5_sp",
    "era5_tcw",
    "era5_tcwv",
    "era5_msl",
    "era5_t2m",
    "era5_lsm",
    "era5_lcc",
    "era5_mcc",
    "era5_hcc",
    "era5_tco3",
    "era5_skt",
    "era5_tsn",
    "era5_fal",
    "era5_lmlt",
    "era5_lict",
    "era5_ssr",
    "era5_tisr",
    "era5_deg0l",
    "era5_avg_sdswrf",
    "era5_avg_sdlwrf",
    "era5_avg_snswrf",
    "era5_avg_snlwrf",
    "era5_avg_snswrfcs",
    "era5_avg_snlwrfcs",
    "era5_avg_sdswrfcs",
]

DERIVED_DEPENDENCIES = {
    "era5_t2m_c": ["era5_t2m"],
    "era5_d2m_c": ["era5_d2m"],
    "era5_dewpoint_depression": ["era5_t2m", "era5_d2m"],
    "era5_wind10_speed": ["era5_u10", "era5_v10"],
    "era5_cloud_layer_sum": ["era5_lcc", "era5_mcc", "era5_hcc"],
}

TIME_FEATURES = [
    LOCAL_TIME_COLUMN,
    TIME_COLUMN,
]

SPACE_FEATURES = [
    LAT_COLUMN,
    LON_COLUMN,
]


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


def add_time_space_features(frame):
    frame = add_local_time(frame)
    frame[TIME_COLUMN] = pd.to_datetime(frame[TIME_COLUMN])

    frame["time_hours_since_start"] = (
        frame[TIME_COLUMN] - frame[TIME_COLUMN].min()
    ).dt.total_seconds() / 3600.0
    frame["hour_sin"] = np.sin(2 * np.pi * frame[TIME_COLUMN].dt.hour / 24.0)
    frame["hour_cos"] = np.cos(2 * np.pi * frame[TIME_COLUMN].dt.hour / 24.0)
    frame["dayofyear_sin"] = np.sin(2 * np.pi * frame[TIME_COLUMN].dt.dayofyear / 366.0)
    frame["dayofyear_cos"] = np.cos(2 * np.pi * frame[TIME_COLUMN].dt.dayofyear / 366.0)

    frame["lat2"] = frame[LAT_COLUMN] ** 2
    frame["lon2"] = frame[LON_COLUMN] ** 2
    frame["lat_lon"] = frame[LAT_COLUMN] * frame[LON_COLUMN]

    return frame


def needed_columns(schema_names):
    available = set(schema_names)
    needed = {Y_COLUMN, COUNT_COLUMN, TIME_COLUMN, LAT_COLUMN, LON_COLUMN}

    for variable in PREFERRED_COVARIATES + [LOCAL_TIME_COLUMN]:
        if variable in available:
            needed.add(variable)
        elif variable in DERIVED_DEPENDENCIES:
            needed.update(DERIVED_DEPENDENCIES[variable])

    missing = sorted({Y_COLUMN, COUNT_COLUMN, TIME_COLUMN, LAT_COLUMN, LON_COLUMN} - available)
    if missing:
        raise ValueError(f"Merged parquet is missing required columns: {missing}")

    return [column for column in schema_names if column in needed]


def load_model_data(data_path):
    schema_names = pq.ParquetFile(data_path).schema_arrow.names
    columns = needed_columns(schema_names)

    table = ds.dataset(data_path, format="parquet").to_table(
        columns=columns,
        filter=ds.field(COUNT_COLUMN) > 0,
    )

    frame = table.to_pandas().reset_index(drop=True)
    frame = add_derived_covariates(frame)
    frame = add_time_space_features(frame)
    frame = add_log_radiance(frame, Y_COLUMN)
    frame = frame.replace([np.inf, -np.inf], np.nan)

    candidate_features = [
        column
        for column in PREFERRED_COVARIATES + TIME_FEATURES + SPACE_FEATURES
        if column in frame.columns
    ]

    feature_columns = []
    for column in candidate_features:
        values = pd.to_numeric(frame[column], errors="coerce")
        if column == LOCAL_TIME_COLUMN or (values.notna().sum() >= 20 and values.nunique(dropna=True) > 1):
            frame[column] = values
            feature_columns.append(column)

    if not feature_columns:
        raise ValueError("No usable random forest feature columns were found.")

    return frame, feature_columns


def spherical_coordinates(frame):
    lat = np.deg2rad(frame[LAT_COLUMN].to_numpy(dtype=float))
    lon = np.deg2rad(frame[LON_COLUMN].to_numpy(dtype=float))

    return np.column_stack(
        [
            np.cos(lat) * np.cos(lon),
            np.cos(lat) * np.sin(lon),
            np.sin(lat),
        ]
    )


def assign_space_folds(frame, n_folds):
    locations = frame[[LAT_COLUMN, LON_COLUMN]].drop_duplicates().reset_index(drop=True)
    n_folds = min(n_folds, len(locations))

    if n_folds < 2:
        locations[SPACE_FOLD_COLUMN] = 0
    else:
        coordinates = spherical_coordinates(locations)
        locations[SPACE_FOLD_COLUMN] = KMeans(
            n_clusters=n_folds,
            random_state=RANDOM_SEED,
            n_init=20,
        ).fit_predict(coordinates)

    return frame.merge(locations, on=[LAT_COLUMN, LON_COLUMN], how="left")


def assign_time_folds(frame, n_folds):
    hours = pd.Series(np.sort(frame[TIME_COLUMN].dropna().unique()), name=TIME_COLUMN)
    n_folds = min(n_folds, len(hours))

    if n_folds < 2:
        fold_ids = np.zeros(len(hours), dtype=int)
    else:
        fold_ids = np.floor(np.arange(len(hours)) * n_folds / len(hours)).astype(int)
        fold_ids = np.minimum(fold_ids, n_folds - 1)

    hour_folds = pd.DataFrame({TIME_COLUMN: hours, TIME_FOLD_COLUMN: fold_ids})
    return frame.merge(hour_folds, on=TIME_COLUMN, how="left")


def assign_space_time_folds(frame):
    frame = assign_space_folds(frame, N_SPACE_FOLDS)
    frame = assign_time_folds(frame, N_TIME_FOLDS)
    return frame


def feature_group(feature):
    if feature.startswith("era5_"):
        return "Atmospheric"
    if feature in TIME_FEATURES:
        return "Time"
    if feature in SPACE_FEATURES:
        return "Space"
    return "Other"


def make_rf_model():
    return Pipeline(
        [
            ("preprocessor", make_covariate_preprocessor()),
            (
                "random_forest",
                RandomForestRegressor(
                    n_estimators=N_ESTIMATORS,
                    min_samples_leaf=MIN_SAMPLES_LEAF,
                    max_features=MAX_FEATURES,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def score_predictions(y_true, y_pred):
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }

    if len(y_true) >= 2 and pd.Series(y_true).nunique(dropna=True) > 1:
        metrics["r2"] = float(r2_score(y_true, y_pred))
    else:
        metrics["r2"] = np.nan

    return metrics


def space_time_splits(frame):
    space_folds = sorted(frame[SPACE_FOLD_COLUMN].dropna().unique())
    time_folds = sorted(frame[TIME_FOLD_COLUMN].dropna().unique())

    for space_fold in space_folds:
        for time_fold in time_folds:
            # Score on the space/time intersection, while removing the whole
            # held-out space fold and the whole held-out time fold from training.
            test_mask = (
                (frame[SPACE_FOLD_COLUMN] == space_fold)
                & (frame[TIME_FOLD_COLUMN] == time_fold)
            )
            blocked_mask = (
                (frame[SPACE_FOLD_COLUMN] == space_fold)
                | (frame[TIME_FOLD_COLUMN] == time_fold)
            )
            train_mask = ~blocked_mask

            train_idx = frame.index[train_mask].to_numpy()
            test_idx = frame.index[test_mask].to_numpy()

            yield int(space_fold), int(time_fold), train_idx, test_idx


def fold_prediction_frame(frame, test_idx, y_test, prediction, baseline_prediction, fold_id):
    prediction_frame = frame.loc[
        test_idx,
        [TIME_COLUMN, LAT_COLUMN, LON_COLUMN, SPACE_FOLD_COLUMN, TIME_FOLD_COLUMN],
    ].copy()
    prediction_frame[TIME_COLUMN] = pd.to_datetime(prediction_frame[TIME_COLUMN])
    prediction_frame.insert(0, FOLD_ID_COLUMN, fold_id)
    prediction_frame["observed"] = y_test.to_numpy(dtype=float)
    prediction_frame["predicted"] = prediction
    prediction_frame["residual"] = prediction_frame["observed"] - prediction_frame["predicted"]
    prediction_frame["abs_error"] = prediction_frame["residual"].abs()
    prediction_frame["squared_error"] = prediction_frame["residual"] ** 2
    prediction_frame["baseline_predicted"] = baseline_prediction
    prediction_frame["baseline_residual"] = (
        prediction_frame["observed"] - prediction_frame["baseline_predicted"]
    )
    prediction_frame["baseline_abs_error"] = prediction_frame["baseline_residual"].abs()
    prediction_frame["baseline_squared_error"] = prediction_frame["baseline_residual"] ** 2

    return prediction_frame


def fold_importance_frame(model, feature_columns, fold_id, space_fold, time_fold, n_train, n_test):
    importances = pd.DataFrame(
        {
            FOLD_ID_COLUMN: fold_id,
            SPACE_FOLD_COLUMN: space_fold,
            TIME_FOLD_COLUMN: time_fold,
            "n_train": n_train,
            "n_test": n_test,
            "feature": model.named_steps["preprocessor"].get_feature_names_out(),
            IMPORTANCE_COLUMN: model.named_steps["random_forest"].feature_importances_,
        }
    )
    importances["feature_group"] = importances["feature"].map(feature_group)

    return importances


def run_space_time_benchmark(frame, feature_columns):
    prediction_frames = []
    importance_frames = []
    metrics_records = []

    X = frame[feature_columns]
    y = frame[Y_COLUMN].astype(float)

    fitted_fold_id = 0
    for space_fold, time_fold, train_idx, test_idx in space_time_splits(frame):
        if len(train_idx) < MIN_TRAIN_ROWS or len(test_idx) < MIN_TEST_ROWS:
            print(
                f"Skipped space fold {space_fold}, time fold {time_fold}: "
                f"n_train={len(train_idx):,} (min {MIN_TRAIN_ROWS}), "
                f"n_test={len(test_idx):,} (min {MIN_TEST_ROWS})"
            )
            continue

        X_train = X.loc[train_idx]
        X_test = X.loc[test_idx]
        y_train = frame.loc[train_idx, LOG_RADIANCE_COLUMN]
        y_test = y.loc[test_idx]

        if y_train.nunique(dropna=True) < 2:
            continue

        fold_id = f"space{space_fold}_time{time_fold}"
        model = make_rf_model()
        model.fit(X_train, y_train)

        prediction = np.exp(model.predict(X_test))
        baseline_prediction = float(np.exp(y_train.mean()))

        prediction_frames.append(
            fold_prediction_frame(
                frame=frame,
                test_idx=test_idx,
                y_test=y_test,
                prediction=prediction,
                baseline_prediction=baseline_prediction,
                fold_id=fold_id,
            )
        )
        importance_frames.append(
            fold_importance_frame(
                model=model,
                feature_columns=feature_columns,
                fold_id=fold_id,
                space_fold=space_fold,
                time_fold=time_fold,
                n_train=len(train_idx),
                n_test=len(test_idx),
            )
        )

        rf_metrics = score_predictions(y_test, prediction)
        baseline_metrics = score_predictions(
            y_test,
            np.full(len(y_test), baseline_prediction, dtype=float),
        )
        metrics_records.append(
            {
                FOLD_ID_COLUMN: fold_id,
                SPACE_FOLD_COLUMN: space_fold,
                TIME_FOLD_COLUMN: time_fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "rmse": rf_metrics["rmse"],
                "mae": rf_metrics["mae"],
                "r2": rf_metrics["r2"],
                "baseline_rmse": baseline_metrics["rmse"],
                "baseline_mae": baseline_metrics["mae"],
                "baseline_r2": baseline_metrics["r2"],
            }
        )

        fitted_fold_id += 1
        print(
            f"Fit {fitted_fold_id}: held out space fold {space_fold}, "
            f"time fold {time_fold}; n_train={len(train_idx):,}, n_test={len(test_idx):,}"
        )

    if not prediction_frames:
        raise ValueError("No valid space-time benchmark folds were fitted.")

    predictions = pd.concat(prediction_frames, ignore_index=True)
    importances = pd.concat(importance_frames, ignore_index=True)
    metrics = pd.DataFrame(metrics_records)

    return predictions, importances, metrics


def summarize_importances(importances):
    summary = (
        importances.groupby(["feature", "feature_group"])
        .agg(
            mean_importance=(IMPORTANCE_COLUMN, "mean"),
            median_importance=(IMPORTANCE_COLUMN, "median"),
            std_importance=(IMPORTANCE_COLUMN, "std"),
            max_importance=(IMPORTANCE_COLUMN, "max"),
            n_folds=(FOLD_ID_COLUMN, "nunique"),
        )
        .reset_index()
    )

    return summary.sort_values("mean_importance", ascending=False)


def overall_metrics(predictions):
    rf_metrics = score_predictions(predictions["observed"], predictions["predicted"])
    baseline_metrics = score_predictions(
        predictions["observed"],
        predictions["baseline_predicted"],
    )

    return pd.DataFrame(
        [
            {
                "validation_strategy": VALIDATION_STRATEGY,
                "target": LOG_RADIANCE_COLUMN,
                "prediction_scale": "radiance",
                "n_predictions": len(predictions),
                "n_estimators": N_ESTIMATORS,
                "min_samples_leaf": MIN_SAMPLES_LEAF,
                "max_features": MAX_FEATURES,
                "rmse": rf_metrics["rmse"],
                "mae": rf_metrics["mae"],
                "r2": rf_metrics["r2"],
                "baseline_rmse": baseline_metrics["rmse"],
                "baseline_mae": baseline_metrics["mae"],
                "baseline_r2": baseline_metrics["r2"],
            }
        ]
    )


def residual_norm(values):
    vmax = float(np.nanmax(np.abs(values)))
    if not np.isfinite(vmax) or vmax == 0:
        return None
    return TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)


def plot_space_time_predictions(predictions, fold_metrics, output_dir, prefix="rf_space_time"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)

    ax = axes[0, 0]
    ax.scatter(
        predictions["observed"],
        predictions["predicted"],
        s=24,
        alpha=0.7,
        edgecolor="none",
    )
    lo = min(predictions["observed"].min(), predictions["predicted"].min())
    hi = max(predictions["observed"].max(), predictions["predicted"].max())
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1, linestyle="--")
    ax.set_title("Observed vs predicted")
    ax.set_xlabel("Observed radiance")
    ax.set_ylabel("Out-of-fold predicted radiance")
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    ax.scatter(
        predictions[TIME_COLUMN],
        predictions["residual"],
        c=predictions[TIME_FOLD_COLUMN],
        cmap="viridis",
        s=24,
        alpha=0.75,
        edgecolor="none",
    )
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_title("Residuals over held-out time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Observed - predicted")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    points = ax.scatter(
        predictions[LON_COLUMN],
        predictions[LAT_COLUMN],
        c=predictions["residual"],
        cmap="coolwarm",
        norm=residual_norm(predictions["residual"]),
        s=30,
        alpha=0.85,
        edgecolor="none",
    )
    fig.colorbar(points, ax=ax, label="Observed - predicted")
    ax.set_title("Spatial residuals")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    ax = axes[1, 1]
    rmse_grid = fold_metrics.pivot(
        index=TIME_FOLD_COLUMN,
        columns=SPACE_FOLD_COLUMN,
        values="rmse",
    ).sort_index(ascending=True)
    image = ax.imshow(rmse_grid.to_numpy(), aspect="auto", cmap="magma")
    fig.colorbar(image, ax=ax, label="RMSE")
    ax.set_title("Fold RMSE")
    ax.set_xlabel("Held-out space fold")
    ax.set_ylabel("Held-out time fold")
    ax.set_xticks(np.arange(len(rmse_grid.columns)))
    ax.set_xticklabels(rmse_grid.columns)
    ax.set_yticks(np.arange(len(rmse_grid.index)))
    ax.set_yticklabels(rmse_grid.index)

    fig.suptitle("Random forest space-time blocked validation")
    output_path = output_dir / f"{prefix}_predictions.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Saved {output_path}")


def plot_space_time_importances(
    importance_summary,
    output_dir,
    max_features=30,
    prefix="rf_space_time",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_frame = importance_summary.head(max_features).copy()
    if plot_frame.empty:
        print("No random forest feature importances to plot.")
        return

    plot_frame = plot_frame.sort_values("mean_importance")
    colors = {
        "Atmospheric": "#4C78A8",
        "Time": "#F58518",
        "Space": "#54A24B",
        "Other": "#B279A2",
    }
    bar_colors = plot_frame["feature_group"].map(colors)

    fig_height = max(5, 0.34 * len(plot_frame))
    fig, ax = plt.subplots(figsize=(10, fig_height), constrained_layout=True)
    ax.barh(
        plot_frame["feature"],
        plot_frame["mean_importance"],
        xerr=plot_frame["std_importance"].fillna(0),
        color=bar_colors,
        alpha=0.9,
    )
    ax.set_title("Mean random forest feature importances across space-time folds")
    ax.set_xlabel("Mean impurity-based feature importance")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)

    handles = [
        plt.Line2D([0], [0], color=color, lw=6, label=label)
        for label, color in colors.items()
        if label in set(plot_frame["feature_group"])
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right")

    output_path = output_dir / f"{prefix}_feature_importances.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Saved {output_path}")


def save_outputs(frame, predictions, importances, importance_summary, metrics, overall):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fold_assignments = frame[
        [TIME_COLUMN, LAT_COLUMN, LON_COLUMN, SPACE_FOLD_COLUMN, TIME_FOLD_COLUMN]
    ].copy()
    fold_assignments[TIME_COLUMN] = pd.to_datetime(fold_assignments[TIME_COLUMN])

    fold_assignments.to_csv(OUT_DIR / "rf_space_time_fold_assignments.csv", index=False)
    predictions.to_csv(OUT_DIR / "rf_space_time_predictions.csv", index=False)
    importances.to_csv(OUT_DIR / "rf_space_time_feature_importances_by_fold.csv", index=False)
    importance_summary.to_csv(OUT_DIR / "rf_space_time_feature_importance_summary.csv", index=False)
    metrics.to_csv(OUT_DIR / "rf_space_time_metrics_by_fold.csv", index=False)
    overall.to_csv(OUT_DIR / "rf_space_time_overall_metrics.csv", index=False)

    print(f"Saved outputs in {OUT_DIR}")


def main():
    frame, feature_columns = load_model_data(DATA_PATH)
    frame = assign_space_time_folds(frame)

    predictions, importances, metrics = run_space_time_benchmark(frame, feature_columns)
    importance_summary = summarize_importances(importances)
    overall = overall_metrics(predictions)

    save_outputs(frame, predictions, importances, importance_summary, metrics, overall)
    plot_space_time_predictions(predictions, metrics, OUT_DIR)
    plot_space_time_importances(importance_summary, OUT_DIR)

    print(f"Validation strategy: {VALIDATION_STRATEGY}")
    print(f"Rows used: {len(frame):,}")
    print(f"Features used: {len(feature_columns):,}")
    print(f"Space folds: {frame[SPACE_FOLD_COLUMN].nunique():,}")
    print(f"Time folds: {frame[TIME_FOLD_COLUMN].nunique():,}")
    print("\nOverall out-of-fold metrics:")
    print(overall.to_string(index=False))
    print("\nTop feature importance summary:")
    print(importance_summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
