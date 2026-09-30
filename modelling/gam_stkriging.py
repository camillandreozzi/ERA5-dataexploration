"""Additive spline mean model plus PyKrige 3D residual kriging.

The same two-stage structure as rf_stkriging.py -- mean function, then space-time
kriging of its residuals -- with a GAM in place of the random forest.

Why a GAM. The RF residual kriging on the midnight subset fitted variograms with
a median range of about 7,000 km (min 3,672, max 23,350) and a structured sill
ratio of 0.90. A range of that size is not local spatial correlation; it is a
planetary-scale gradient. In other words the kriging stage was spending its
capacity re-learning the large-scale meridional pattern that the mean model had
failed to represent, even though `latitude` was the RF's single most important
feature. An axis-aligned tree ensemble approximates a smooth gradient with a
staircase, and with space-time blocked folds it cannot extrapolate that staircase
to a held-out latitude band at all: `era5_skt` has marginal correlation +0.43 with
the target yet scores -0.101 alone under latitude-blocked validation.

A GAM represents that gradient as an explicit smooth function instead, which both
extrapolates more gracefully and leaves the kriging stage free to model whatever
genuinely local structure remains.

What "GAM" means here. Penalised regression splines: each continuous covariate is
expanded into a B-spline basis and the coefficients are fitted by ridge
regression, with the penalty chosen by cross-validation on the training fold.
That is the same construction as mgcv's default, minus per-term smoothing
parameters -- one global penalty is shared by all terms. It is deliberately built
from scikit-learn rather than pygam or statsmodels, neither of which is installed
here or listed in euler/requirements.txt.

Longitude and local time are expanded with periodic splines so that -180 meets
180 and hour 24 meets hour 0, which an ordinary basis would tear apart.

Run with:
    python3 modelling/gam_stkriging.py --mode validate
"""

import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "paths.py").exists():
        sys.path.insert(0, str(_p))
        break
from paths import subset_data_path, subset_results_path

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib

matplotlib.use("Agg")

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

try:
    from modelling import benchmark_rf as rf_benchmark
    from modelling import rf_stkriging as kriging
    from modelling.covariate_preprocessing import (
        CATEGORICAL_COVARIATES,
        LOCAL_TIME_COLUMN,
        LOG_RADIANCE_COLUMN,
    )
except ModuleNotFoundError:
    import benchmark_rf as rf_benchmark
    import rf_stkriging as kriging
    from covariate_preprocessing import (
        CATEGORICAL_COVARIATES,
        LOCAL_TIME_COLUMN,
        LOG_RADIANCE_COLUMN,
    )


OUT_DIR = subset_results_path("modelling/gam_stkriging")
OUTPUT_PREFIX = "gam_stkriging_space_time"
DATA_PATH = subset_data_path("CLARA_ERA5_merged.parquet")

# Ocean-only covariates would drop every land cell; excluded for the same reason
# as in rf_stkriging.py.
EXCLUDED_COVARIATES = kriging.EXCLUDED_COVARIATES

# Spline basis. n_knots is kept small because the midnight subset has about a
# thousand rows spread over 16 blocked folds, so each fit sees a few hundred:
# a rich basis per covariate would interpolate noise. degree=3 gives the usual
# cubic splines.
N_KNOTS = 6
SPLINE_DEGREE = 3
KNOT_STRATEGY = "quantile"

# One global ridge penalty over a wide grid. RidgeCV picks it per training fold
# by efficient leave-one-out, so no extra CV loop is needed. The bases are
# standardised first (see make_gam_model), which is what lets a single penalty
# fall evenly across terms of very different natural scales.
RIDGE_ALPHAS = np.logspace(-3, 6, 60)

# Mean-model variance. The RF version estimates it from the spread across trees.
# A ridge-fitted GAM has no ensemble, so the analogue is the variance of the
# fitted linear predictor: sigma^2 * x' (X'X + alpha I)^-1 x, with sigma^2 the
# training residual variance. That is a genuine sampling variance of the mean
# under the fitted model, and unlike the RF proxy it is not an ensemble spread.
GAM_VARIANCE_METHOD = "ridge_linear_predictor_variance"


# Full cycle of each wrapping covariate. SplineTransformer has no `period`
# argument: with extrapolation="periodic" the period is the distance between the
# first and last knot, so the cycle has to be imposed through explicit knots.
# Inferring them from the data would be wrong here -- midnight-subset local times
# sit near 23.5-24 and 0-0.5, so a data-derived span would wrap over a fraction
# of an hour rather than over the day.
PERIODIC_BOUNDS = {
    "longitude": (-180.0, 180.0),
    LOCAL_TIME_COLUMN: (0.0, 24.0),
}


def make_spline_pipeline(column):
    """Impute, then expand into a B-spline basis, periodic where the axis wraps."""
    bounds = PERIODIC_BOUNDS.get(column)
    if bounds is None:
        spline = SplineTransformer(
            n_knots=N_KNOTS,
            degree=SPLINE_DEGREE,
            knots=KNOT_STRATEGY,
            extrapolation="constant",
            include_bias=False,
        )
    else:
        low, high = bounds
        spline = SplineTransformer(
            degree=SPLINE_DEGREE,
            knots=np.linspace(low, high, N_KNOTS).reshape(-1, 1),
            extrapolation="periodic",
            include_bias=False,
        )

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("spline", spline),
        ]
    )


def make_gam_preprocessor(frame, feature_columns):
    """One smooth term per continuous covariate, one-hot for the categoricals.

    Built per fold from the training frame, because the periodic terms need the
    observed range and the quantile knots need the training distribution.
    """
    categorical = [c for c in feature_columns if c in CATEGORICAL_COVARIATES]
    continuous = [c for c in feature_columns if c not in CATEGORICAL_COVARIATES]

    transformers = []
    for column in continuous:
        transformers.append((f"s_{column}", make_spline_pipeline(column), [column]))

    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value=-1,
                                                  keep_empty_features=True)),
                        ("encoder", OneHotEncoder(handle_unknown="ignore",
                                                  sparse_output=False)),
                    ]
                ),
                categorical,
            )
        )

    return ColumnTransformer(transformers, verbose_feature_names_out=False)


def make_gam_model(frame, feature_columns):
    """Additive spline bases -> standardise -> ridge with CV-chosen penalty."""
    return Pipeline(
        [
            ("preprocessor", make_gam_preprocessor(frame, feature_columns)),
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=RIDGE_ALPHAS)),
        ]
    )


def design_matrix(model, features):
    """The basis expansion the ridge step actually sees."""
    design = model.named_steps["preprocessor"].transform(features)
    return model.named_steps["scaler"].transform(design)


def fit_gam_log_moments(model, x_train, y_train, x_test):
    """Test-set mean prediction and the variance of the fitted linear predictor.

    Var(x'b) = sigma^2 * x' (X'X + alpha I)^-1 x for ridge with penalty alpha,
    taking sigma^2 from the training residuals. The intercept is unpenalised and
    excluded from the quadratic form, which slightly understates the variance; it
    is a diagnostic, not a calibrated predictive interval, the same caveat the RF
    tree-spread proxy carries.
    """
    prediction = model.predict(x_test)

    train_design = design_matrix(model, x_train)
    test_design = design_matrix(model, x_test)
    alpha = float(model.named_steps["ridge"].alpha_)

    residuals = np.asarray(y_train, dtype=float) - model.predict(x_train)
    dof = max(len(residuals) - np.linalg.matrix_rank(train_design), 1)
    sigma_squared = float(residuals @ residuals) / dof

    gram = train_design.T @ train_design + alpha * np.eye(train_design.shape[1])
    try:
        solved = np.linalg.solve(gram, test_design.T)
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(gram) @ test_design.T

    leverage = np.einsum("ij,ji->i", test_design, solved)
    variance = sigma_squared * np.clip(leverage, 0.0, None)
    return prediction, variance.astype(float)


def fold_term_frame(model, fold_id, space_fold, time_fold, n_train, n_test):
    """Per-term contribution, the GAM's analogue of RF feature importance.

    A single spline term spans many basis columns, so the coefficients are
    summarised per covariate as the root sum of squares of its standardised
    coefficients: how much that smooth moves the linear predictor overall.
    """
    names = list(model.named_steps["preprocessor"].get_feature_names_out())
    coefficients = np.asarray(model.named_steps["ridge"].coef_, dtype=float).ravel()

    records = []
    for name, coefficient in zip(names, coefficients):
        # rsplit, not split: SplineTransformer suffixes each basis column with
        # "_sp_<k>", and era5_sp (surface pressure) then yields "era5_sp_sp_0",
        # which a left split would truncate to "era5".
        term = name.rsplit("_sp_", 1)[0] if "_sp_" in name else name
        records.append({"basis": name, "term": term, "coefficient": coefficient})

    frame = pd.DataFrame(records)
    summary = (
        frame.groupby("term", as_index=False)
        .agg(
            term_strength=("coefficient", lambda c: float(np.sqrt(np.sum(np.square(c))))),
            n_basis=("coefficient", "size"),
        )
    )
    summary["feature"] = summary["term"]
    summary["feature_group"] = summary["feature"].map(rf_benchmark.feature_group)
    summary[rf_benchmark.FOLD_ID_COLUMN] = fold_id
    summary[rf_benchmark.SPACE_FOLD_COLUMN] = space_fold
    summary[rf_benchmark.TIME_FOLD_COLUMN] = time_fold
    summary["n_train"] = n_train
    summary["n_test"] = n_test
    summary["ridge_alpha"] = float(model.named_steps["ridge"].alpha_)
    return summary


def summarize_terms(terms):
    return (
        terms.groupby(["feature", "feature_group"], as_index=False)
        .agg(
            mean_strength=("term_strength", "mean"),
            median_strength=("term_strength", "median"),
            std_strength=("term_strength", "std"),
            max_strength=("term_strength", "max"),
            n_folds=("term_strength", "size"),
        )
        .sort_values("mean_strength", ascending=False)
        .reset_index(drop=True)
    )


def gam_fit_predict_factory(train_frame, feature_columns):
    """fit_predict closure for out_of_fold_residuals.

    The preprocessor is rebuilt per inner fit so the quantile knots come from
    that inner training set only -- otherwise the knot placement would carry
    information from the inner test rows.
    """
    def fit_predict(x_train, y_train, x_test):
        model = make_gam_model(train_frame.loc[x_train.index], feature_columns)
        model.fit(x_train, y_train)
        return model.predict(x_test)

    return fit_predict


def run_space_time_benchmark(frame, feature_columns, variogram_residuals="in-sample"):
    """Mirrors rf_stkriging.run_space_time_benchmark with the GAM mean model."""
    prediction_frames = []
    term_frames = []
    metrics_records = []

    feature_data = frame[feature_columns]
    target = frame[LOG_RADIANCE_COLUMN].astype(float)

    fitted_fold_id = 0
    for space_fold, time_fold, train_idx, test_idx in rf_benchmark.space_time_splits(frame):
        if len(train_idx) < rf_benchmark.MIN_TRAIN_ROWS or len(test_idx) < rf_benchmark.MIN_TEST_ROWS:
            print(
                f"Skipped space fold {space_fold}, time fold {time_fold}: "
                f"n_train={len(train_idx):,} (min {rf_benchmark.MIN_TRAIN_ROWS}), "
                f"n_test={len(test_idx):,} (min {rf_benchmark.MIN_TEST_ROWS})"
            )
            continue

        x_train = feature_data.loc[train_idx]
        x_test = feature_data.loc[test_idx]
        y_train = target.loc[train_idx]
        y_test = frame.loc[test_idx, rf_benchmark.Y_COLUMN].astype(float)

        if y_train.nunique(dropna=True) < 2:
            continue

        fold_id = f"space{space_fold}_time{time_fold}"
        model = make_gam_model(frame.loc[train_idx], feature_columns)
        model.fit(x_train, y_train)

        gam_train_prediction = model.predict(x_train)
        gam_test_prediction, gam_log_variance = fit_gam_log_moments(
            model, x_train, y_train, x_test
        )
        train_residuals = y_train.to_numpy(dtype=float) - gam_train_prediction

        coordinate_scaler = kriging.SpaceTimeCoordinateScaler.fit(frame.loc[train_idx])
        st_x_train, st_y_train, st_z_train = coordinate_scaler.transform(frame.loc[train_idx])
        st_x_test, st_y_test, st_z_test = coordinate_scaler.transform(frame.loc[test_idx])

        variogram_sample = None
        if variogram_residuals == "nested":
            oof_residuals, oof_frame = kriging.out_of_fold_residuals(
                train_frame=frame.loc[train_idx],
                feature_columns=feature_columns,
                target_column=LOG_RADIANCE_COLUMN,
                fit_predict=gam_fit_predict_factory(
                    frame.loc[train_idx].reset_index(drop=True), feature_columns
                ),
            )
            if oof_residuals is not None:
                # Same scaler as the kriging data, so both sets of residuals
                # live in one coordinate system.
                ox, oy, oz = coordinate_scaler.transform(oof_frame)
                variogram_sample = (ox, oy, oz, oof_residuals)

        residual_kriging, kriging_info = kriging.fit_residual_kriging(
            x_train=st_x_train,
            y_train=st_y_train,
            z_train=st_z_train,
            residuals=train_residuals,
            variogram_sample=variogram_sample,
        )
        kriged_residual_prediction, kriging_variance, n_closest = kriging.predict_residuals(
            kriging=residual_kriging,
            x_test=st_x_test,
            y_test=st_y_test,
            z_test=st_z_test,
            n_train=kriging_info["kriging_n_train"],
        )

        combined_log_prediction = gam_test_prediction + kriged_residual_prediction
        variance_components = kriging.radiance_variance_components(
            combined_log_prediction, gam_log_variance, kriging_variance
        )
        prediction = np.exp(combined_log_prediction)
        gam_mean_prediction = np.exp(gam_test_prediction)
        baseline_prediction = float(np.exp(y_train.mean()))

        # fold_prediction_frame names the mean-model column rf_mean_predicted.
        # Reused as-is so the two result sets stay column-compatible and can be
        # compared without renaming; the values here are the GAM's.
        prediction_frames.append(
            kriging.fold_prediction_frame(
                frame=frame,
                test_idx=test_idx,
                y_test=y_test,
                prediction=prediction,
                rf_mean_prediction=gam_mean_prediction,
                kriged_residual_prediction=kriged_residual_prediction,
                baseline_prediction=baseline_prediction,
                fold_id=fold_id,
                variance_components=variance_components,
            )
        )
        term_frames.append(
            fold_term_frame(
                model=model,
                fold_id=fold_id,
                space_fold=space_fold,
                time_fold=time_fold,
                n_train=len(train_idx),
                n_test=len(test_idx),
            )
        )

        final_metrics = rf_benchmark.score_predictions(y_test, prediction)
        gam_mean_metrics = rf_benchmark.score_predictions(y_test, gam_mean_prediction)
        baseline_metrics = rf_benchmark.score_predictions(
            y_test,
            np.full(len(y_test), baseline_prediction, dtype=float),
        )
        metrics_records.append(
            {
                rf_benchmark.FOLD_ID_COLUMN: fold_id,
                rf_benchmark.SPACE_FOLD_COLUMN: space_fold,
                rf_benchmark.TIME_FOLD_COLUMN: time_fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "rmse": final_metrics["rmse"],
                "mae": final_metrics["mae"],
                "r2": final_metrics["r2"],
                "gam_mean_rmse": gam_mean_metrics["rmse"],
                "gam_mean_mae": gam_mean_metrics["mae"],
                "gam_mean_r2": gam_mean_metrics["r2"],
                "baseline_rmse": baseline_metrics["rmse"],
                "baseline_mae": baseline_metrics["mae"],
                "baseline_r2": baseline_metrics["r2"],
                "ridge_alpha": float(model.named_steps["ridge"].alpha_),
                "n_basis_columns": int(design_matrix(model, x_train).shape[1]),
                "variogram_model": kriging.VARIOGRAM_MODEL,
                "variogram_nlags": kriging.VARIOGRAM_NLAGS,
                "kriging_backend": kriging.KRIGING_BACKEND,
                "kriging_n_closest_points": n_closest,
                "kriging_status": kriging_info["kriging_status"],
                "kriging_n_train": kriging_info["kriging_n_train"],
                "variogram_residuals": variogram_residuals,
                "variogram_source": kriging_info["variogram_source"],
                "variogram_psill": kriging_info["variogram_psill"],
                "variogram_range": kriging_info["variogram_range"],
                "variogram_nugget": kriging_info["variogram_nugget"],
                "variogram_structured_sill_ratio": kriging_info[
                    "variogram_structured_sill_ratio"
                ],
                "kriging_time_std_hours": coordinate_scaler.time_std_hours,
                "mean_kriging_log_variance": float(np.nanmean(kriging_variance)),
                "gam_variance_method": GAM_VARIANCE_METHOD,
                "mean_gam_log_variance": float(np.nanmean(gam_log_variance)),
            }
        )

        fitted_fold_id += 1
        print(
            f"Fit {fitted_fold_id}: held out space fold {space_fold}, "
            f"time fold {time_fold}; n_train={len(train_idx):,}, "
            f"n_test={len(test_idx):,}, alpha={model.named_steps['ridge'].alpha_:.4g}"
        )

    if not prediction_frames:
        raise ValueError("No valid GAM + residual kriging folds were fitted.")

    return (
        pd.concat(prediction_frames, ignore_index=True),
        pd.concat(term_frames, ignore_index=True),
        pd.DataFrame(metrics_records),
    )


def overall_metrics(predictions, metrics, variogram_residuals):
    final_metrics = rf_benchmark.score_predictions(
        predictions["observed"], predictions["predicted"]
    )
    gam_mean_metrics = rf_benchmark.score_predictions(
        predictions["observed"], predictions["rf_mean_predicted"]
    )
    baseline_metrics = rf_benchmark.score_predictions(
        predictions["observed"], predictions["baseline_predicted"]
    )

    return pd.DataFrame(
        [
            {
                "validation_strategy": rf_benchmark.VALIDATION_STRATEGY,
                "mean_model": "additive_penalised_splines",
                "target": LOG_RADIANCE_COLUMN,
                "prediction_scale": "radiance",
                "gam_variance_method": GAM_VARIANCE_METHOD,
                "variance_assumption": "independent_gaussian_log_errors",
                "n_predictions": len(predictions),
                "n_knots": N_KNOTS,
                "spline_degree": SPLINE_DEGREE,
                "knot_strategy": KNOT_STRATEGY,
                "median_ridge_alpha": float(metrics["ridge_alpha"].median()),
                "variogram_model": kriging.VARIOGRAM_MODEL,
                "variogram_residuals": variogram_residuals,
                "median_variogram_range": float(metrics["variogram_range"].median()),
                "rmse": final_metrics["rmse"],
                "mae": final_metrics["mae"],
                "r2": final_metrics["r2"],
                "gam_mean_rmse": gam_mean_metrics["rmse"],
                "gam_mean_mae": gam_mean_metrics["mae"],
                "gam_mean_r2": gam_mean_metrics["r2"],
                "baseline_rmse": baseline_metrics["rmse"],
                "baseline_mae": baseline_metrics["mae"],
                "baseline_r2": baseline_metrics["r2"],
            }
        ]
    )


def plot_term_strengths(summary, output_path, top_n=25):
    top = summary.head(top_n).iloc[::-1]
    colors = {
        "Atmospheric": "#2563eb",
        "Space": "#dc2626",
        "Time": "#f59e0b",
        "Other": "#6b7280",
    }
    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(top))))
    ax.barh(
        top["feature"],
        top["mean_strength"],
        color=[colors.get(g, "#6b7280") for g in top["feature_group"]],
    )
    ax.set_xlabel("mean term strength (root sum of squared standardised coefficients)")
    ax.set_title("GAM smooth-term strength, mean over folds")
    ax.grid(True, axis="x", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles, colors.keys(), loc="lower right", fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"figure -> {output_path}")


def save_outputs(frame, predictions, terms, term_summary, metrics, overall):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fold_assignments = frame[
        [
            rf_benchmark.TIME_COLUMN,
            rf_benchmark.LAT_COLUMN,
            rf_benchmark.LON_COLUMN,
            rf_benchmark.SPACE_FOLD_COLUMN,
            rf_benchmark.TIME_FOLD_COLUMN,
        ]
    ].copy()
    fold_assignments[rf_benchmark.TIME_COLUMN] = pd.to_datetime(
        fold_assignments[rf_benchmark.TIME_COLUMN]
    )

    fold_assignments.to_csv(OUT_DIR / f"{OUTPUT_PREFIX}_fold_assignments.csv", index=False)
    predictions.to_csv(OUT_DIR / f"{OUTPUT_PREFIX}_predictions.csv", index=False)
    terms.to_csv(OUT_DIR / f"{OUTPUT_PREFIX}_term_strengths_by_fold.csv", index=False)
    term_summary.to_csv(OUT_DIR / f"{OUTPUT_PREFIX}_term_strength_summary.csv", index=False)
    metrics.to_csv(OUT_DIR / f"{OUTPUT_PREFIX}_metrics_by_fold.csv", index=False)
    overall.to_csv(OUT_DIR / f"{OUTPUT_PREFIX}_overall_metrics.csv", index=False)

    print(f"Saved outputs in {OUT_DIR}")


def load_model_data(data_path):
    frame, feature_columns = rf_benchmark.load_model_data(data_path)
    feature_columns = [name for name in feature_columns if name not in EXCLUDED_COVARIATES]
    return frame.drop(columns=list(EXCLUDED_COVARIATES), errors="ignore"), feature_columns


def parse_args():
    parser = argparse.ArgumentParser(
        description="Additive spline mean model plus PyKrige 3D residual kriging."
    )
    parser.add_argument(
        "--mode",
        choices=["validate"],
        default="validate",
        help=(
            "Only blocked validation is implemented. Full-grid prediction would "
            "need the grid-streaming path from rf_stkriging.py."
        ),
    )
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument(
        "--variogram-residuals",
        choices=kriging.VARIOGRAM_RESIDUAL_CHOICES,
        default="in-sample",
        help=(
            "Which residuals the variogram is estimated from. 'in-sample' is the "
            "original behaviour and uses the residuals the mean model just "
            "minimised. 'nested' uses out-of-fold residuals from inner blocked "
            "folds of the training rows, which reflect genuine predictive error "
            "and generally raise the nugget, damping the kriged correction. "
            "Kriging interpolates from the in-sample residuals either way."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    kriging.require_pykrige()

    frame, feature_columns = load_model_data(args.data_path)
    print(f"Loaded {len(frame):,} rows with {len(feature_columns)} covariates.")

    frame = rf_benchmark.assign_space_time_folds(frame.copy())
    predictions, terms, metrics = run_space_time_benchmark(
        frame, feature_columns, variogram_residuals=args.variogram_residuals
    )
    term_summary = summarize_terms(terms)
    overall = overall_metrics(predictions, metrics, args.variogram_residuals)

    save_outputs(frame, predictions, terms, term_summary, metrics, overall)
    plot_term_strengths(term_summary, OUT_DIR / f"{OUTPUT_PREFIX}_term_strengths.png")
    rf_benchmark.plot_space_time_predictions(
        predictions, metrics, OUT_DIR, prefix=OUTPUT_PREFIX
    )

    row = overall.iloc[0]
    print(
        f"\nGAM + kriging: RMSE {row['rmse']:.3f}, R2 {row['r2']:.4f}\n"
        f"GAM mean only: RMSE {row['gam_mean_rmse']:.3f}, R2 {row['gam_mean_r2']:.4f}\n"
        f"Baseline:      RMSE {row['baseline_rmse']:.3f}, R2 {row['baseline_r2']:.4f}\n"
        f"Median variogram range: {row['median_variogram_range']:,.0f} "
        "(scaled km, space and time mixed)"
    )
    print("\nTop terms:")
    print(term_summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
