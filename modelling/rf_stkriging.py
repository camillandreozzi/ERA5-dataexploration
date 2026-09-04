import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

try:
    from pykrige.ok3d import OrdinaryKriging3D
except ModuleNotFoundError as error:
    OrdinaryKriging3D = None
    PYKRIGE_IMPORT_ERROR = error
else:
    PYKRIGE_IMPORT_ERROR = None

try:
    import benchmark_rf as rf_benchmark
except ModuleNotFoundError:
    from modelling import benchmark_rf as rf_benchmark


OUT_DIR = Path("results/modelling/rf_stkriging")
OUTPUT_PREFIX = "rf_stkriging_space_time"
GRID_OUTPUT_DIR = OUT_DIR / "grid_predictions"

VARIOGRAM_MODEL = "spherical"
VARIOGRAM_NLAGS = 6
VARIOGRAM_WEIGHT = True
KRIGING_BACKEND = "loop"
N_CLOSEST_POINTS = 64
GRID_KRIGING_BACKEND = "vectorized"
GRID_N_CLOSEST_POINTS = None
GRID_BATCH_SIZE = 50_000
MIN_KRIGING_ROWS = 20
MIN_STRUCTURED_SILL_RATIO = 0.05
FALLBACK_VARIANCE_PARTIAL_SILL_SHARE = 0.9
FALLBACK_RANGE_QUANTILE = 0.35
EARTH_RADIUS_KM = 6_371.0088
TIME_AXIS_SCALE_MULTIPLIER = 1.0
WRAP_LONGITUDE_FOR_KRIGING = False
GRID_HOUR_ALL = "all"
GRID_HOUR_LATEST = "latest"


@dataclass
class SpaceTimeCoordinateScaler:
    center_latitude: float
    center_longitude: float
    time_mean_hours: float
    time_std_hours: float
    spatial_scale_km: float

    @classmethod
    def fit(cls, frame):
        longitude = frame[rf_benchmark.LON_COLUMN].to_numpy(dtype=float)
        latitude = frame[rf_benchmark.LAT_COLUMN].to_numpy(dtype=float)
        center_longitude = center_longitude_degrees(longitude)
        center_latitude = float(np.nanmean(latitude))

        time_hours = timestamp_hours(frame[rf_benchmark.TIME_COLUMN])
        time_mean_hours = float(np.nanmean(time_hours))
        time_std_hours = float(np.nanstd(time_hours))
        if not np.isfinite(time_std_hours) or time_std_hours == 0:
            time_std_hours = 1.0

        x_km, y_km = project_lon_lat_to_km(
            longitude=longitude,
            latitude=latitude,
            center_longitude=center_longitude,
            center_latitude=center_latitude,
        )
        spatial_scales = np.nanstd(np.column_stack([x_km, y_km]), axis=0)
        spatial_scales = spatial_scales[np.isfinite(spatial_scales) & (spatial_scales > 0)]
        if len(spatial_scales):
            spatial_scale_km = float(np.median(spatial_scales))
        else:
            spatial_scale_km = 1.0

        return cls(
            center_latitude=center_latitude,
            center_longitude=center_longitude,
            time_mean_hours=time_mean_hours,
            time_std_hours=time_std_hours,
            spatial_scale_km=spatial_scale_km,
        )

    def transform(self, frame):
        x_km, y_km = project_lon_lat_to_km(
            longitude=frame[rf_benchmark.LON_COLUMN].to_numpy(dtype=float),
            latitude=frame[rf_benchmark.LAT_COLUMN].to_numpy(dtype=float),
            center_longitude=self.center_longitude,
            center_latitude=self.center_latitude,
        )
        time_hours = timestamp_hours(frame[rf_benchmark.TIME_COLUMN])
        standardized_time = (time_hours - self.time_mean_hours) / self.time_std_hours
        z_km = standardized_time * self.spatial_scale_km * TIME_AXIS_SCALE_MULTIPLIER

        return x_km, y_km, z_km


def circular_mean_degrees(values):
    radians = np.deg2rad(values)
    mean_sin = np.nanmean(np.sin(radians))
    mean_cos = np.nanmean(np.cos(radians))

    if not np.isfinite(mean_sin) or not np.isfinite(mean_cos):
        return 0.0
    if np.hypot(mean_sin, mean_cos) < 1e-12:
        return float(np.nanmedian(values))

    return float(np.rad2deg(np.arctan2(mean_sin, mean_cos)))


def center_longitude_degrees(values):
    if WRAP_LONGITUDE_FOR_KRIGING:
        return circular_mean_degrees(values)

    return 0.0


def longitude_delta_degrees(longitude, center_longitude):
    if WRAP_LONGITUDE_FOR_KRIGING:
        return ((longitude - center_longitude + 180.0) % 360.0) - 180.0

    return longitude - center_longitude


def project_lon_lat_to_km(longitude, latitude, center_longitude, center_latitude):
    delta_lon = longitude_delta_degrees(longitude, center_longitude)
    delta_lat = latitude - center_latitude
    center_latitude_rad = np.deg2rad(center_latitude)

    x_km = EARTH_RADIUS_KM * np.deg2rad(delta_lon) * np.cos(center_latitude_rad)
    y_km = EARTH_RADIUS_KM * np.deg2rad(delta_lat)

    return x_km, y_km


def timestamp_hours(values):
    timestamps = pd.to_datetime(values)
    return timestamps.astype("int64").to_numpy(dtype=float) / (1_000_000_000.0 * 3600.0)


def require_pykrige():
    if OrdinaryKriging3D is None:
        raise ImportError(
            "PyKrige is required for rf_stkriging.py. Install it with "
            "`python3 -m pip install pykrige` and rerun this script."
        ) from PYKRIGE_IMPORT_ERROR


def finite_residual_training_data(x, y, z, residuals):
    residuals = np.asarray(residuals, dtype=float)
    finite_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(residuals)

    return x[finite_mask], y[finite_mask], z[finite_mask], residuals[finite_mask]


def distance_quantile(x, y, z, quantile):
    coordinates = np.column_stack([x, y, z])
    n_coordinates = len(coordinates)

    if n_coordinates < 2:
        return 1.0

    if n_coordinates <= 2_500:
        from scipy.spatial.distance import pdist

        distances = pdist(coordinates)
        distances = distances[np.isfinite(distances) & (distances > 0)]
    else:
        rng = np.random.default_rng(rf_benchmark.RANDOM_SEED)
        first = rng.integers(0, n_coordinates, size=250_000)
        second = rng.integers(0, n_coordinates, size=250_000)
        pair_mask = first != second
        differences = coordinates[first[pair_mask]] - coordinates[second[pair_mask]]
        distances = np.linalg.norm(differences, axis=1)
        distances = distances[np.isfinite(distances) & (distances > 0)]

    if len(distances) == 0:
        return 1.0

    return float(np.nanquantile(distances, quantile))


def empirical_variogram_parameters(x, y, z, residuals):
    residual_variance = float(np.nanvar(residuals))
    if not np.isfinite(residual_variance) or residual_variance <= 0:
        residual_variance = 1.0

    partial_sill = residual_variance * FALLBACK_VARIANCE_PARTIAL_SILL_SHARE
    nugget = residual_variance - partial_sill
    variogram_range = distance_quantile(x, y, z, FALLBACK_RANGE_QUANTILE)

    return {
        "psill": partial_sill,
        "range": variogram_range,
        "nugget": nugget,
    }


def variogram_parameters_from_model(kriging):
    psill, variogram_range, nugget = kriging.variogram_model_parameters
    return {
        "psill": float(psill),
        "range": float(variogram_range),
        "nugget": float(nugget),
    }


def structured_sill_ratio(parameters):
    total_sill = parameters["psill"] + parameters["nugget"]
    if not np.isfinite(total_sill) or total_sill <= 0:
        return 0.0

    return parameters["psill"] / total_sill


def make_residual_kriging(x_train, y_train, z_train, residuals, variogram_parameters=None):
    return OrdinaryKriging3D(
        x_train,
        y_train,
        z_train,
        residuals,
        variogram_model=VARIOGRAM_MODEL,
        variogram_parameters=variogram_parameters,
        nlags=VARIOGRAM_NLAGS,
        weight=VARIOGRAM_WEIGHT,
        verbose=False,
        enable_plotting=False,
        exact_values=True,
        pseudo_inv=True,
        pseudo_inv_type="pinv",
    )


@dataclass
class FittedRFResidualKriging:
    model: object
    coordinate_scaler: SpaceTimeCoordinateScaler
    residual_kriging: object
    kriging_info: dict
    observation_tree: object


def kriging_window_size(n_train, n_closest_points):
    if n_closest_points is None or n_train <= n_closest_points:
        return None
    return max(2, min(n_closest_points, n_train))


def fit_residual_kriging(x_train, y_train, z_train, residuals):
    require_pykrige()

    x_train, y_train, z_train, residuals = finite_residual_training_data(
        x=x_train,
        y=y_train,
        z=z_train,
        residuals=residuals,
    )

    if len(residuals) < MIN_KRIGING_ROWS or pd.Series(residuals).nunique(dropna=True) < 2:
        return None, {
            "kriging_status": "skipped_constant_or_small_residual_set",
            "kriging_n_train": len(residuals),
            "variogram_source": "none",
            "variogram_psill": np.nan,
            "variogram_range": np.nan,
            "variogram_nugget": np.nan,
            "variogram_structured_sill_ratio": np.nan,
        }

    kriging = make_residual_kriging(x_train, y_train, z_train, residuals)
    variogram_parameters = variogram_parameters_from_model(kriging)
    variogram_source = "auto"

    if structured_sill_ratio(variogram_parameters) < MIN_STRUCTURED_SILL_RATIO:
        variogram_parameters = empirical_variogram_parameters(
            x=x_train,
            y=y_train,
            z=z_train,
            residuals=residuals,
        )
        kriging = make_residual_kriging(
            x_train,
            y_train,
            z_train,
            residuals,
            variogram_parameters=variogram_parameters,
        )
        variogram_source = "empirical_fallback"

    return kriging, {
        "kriging_status": "fit",
        "kriging_n_train": len(residuals),
        "variogram_source": variogram_source,
        "variogram_psill": variogram_parameters["psill"],
        "variogram_range": variogram_parameters["range"],
        "variogram_nugget": variogram_parameters["nugget"],
        "variogram_structured_sill_ratio": structured_sill_ratio(variogram_parameters),
    }


def predict_residuals(
    kriging,
    x_test,
    y_test,
    z_test,
    n_train,
    backend=KRIGING_BACKEND,
    n_closest_points=N_CLOSEST_POINTS,
):
    if kriging is None:
        return (
            np.zeros(len(x_test), dtype=float),
            np.full(len(x_test), np.nan, dtype=float),
            None,
        )

    n_closest = kriging_window_size(n_train, n_closest_points)
    residual_prediction, residual_variance = kriging.execute(
        "points",
        x_test,
        y_test,
        z_test,
        backend=backend,
        n_closest_points=n_closest,
    )

    residual_prediction = np.ma.filled(residual_prediction, np.nan)
    residual_variance = np.ma.filled(residual_variance, np.nan)
    residual_prediction = np.asarray(residual_prediction, dtype=float).reshape(-1)
    residual_variance = np.asarray(residual_variance, dtype=float).reshape(-1)

    nonfinite_mask = ~np.isfinite(residual_prediction)
    if nonfinite_mask.any():
        residual_prediction[nonfinite_mask] = 0.0
        residual_variance[nonfinite_mask] = np.nan

    return residual_prediction, residual_variance, n_closest


def fold_prediction_frame(
    frame,
    test_idx,
    y_test,
    prediction,
    rf_mean_prediction,
    kriged_residual_prediction,
    kriging_variance,
    baseline_prediction,
    fold_id,
):
    prediction_frame = frame.loc[
        test_idx,
        [
            rf_benchmark.TIME_COLUMN,
            rf_benchmark.LAT_COLUMN,
            rf_benchmark.LON_COLUMN,
            rf_benchmark.SPACE_FOLD_COLUMN,
            rf_benchmark.TIME_FOLD_COLUMN,
        ],
    ].copy()
    prediction_frame[rf_benchmark.TIME_COLUMN] = pd.to_datetime(
        prediction_frame[rf_benchmark.TIME_COLUMN]
    )
    prediction_frame.insert(0, rf_benchmark.FOLD_ID_COLUMN, fold_id)
    prediction_frame["observed"] = y_test.to_numpy(dtype=float)
    prediction_frame["predicted"] = prediction
    prediction_frame["rf_mean_predicted"] = rf_mean_prediction
    prediction_frame["kriged_residual_predicted"] = kriged_residual_prediction
    prediction_frame["kriging_variance"] = kriging_variance
    prediction_frame["residual"] = prediction_frame["observed"] - prediction_frame["predicted"]
    prediction_frame["abs_error"] = prediction_frame["residual"].abs()
    prediction_frame["squared_error"] = prediction_frame["residual"] ** 2
    prediction_frame["rf_mean_residual"] = (
        prediction_frame["observed"] - prediction_frame["rf_mean_predicted"]
    )
    prediction_frame["baseline_predicted"] = baseline_prediction
    prediction_frame["baseline_residual"] = (
        prediction_frame["observed"] - prediction_frame["baseline_predicted"]
    )
    prediction_frame["baseline_abs_error"] = prediction_frame["baseline_residual"].abs()
    prediction_frame["baseline_squared_error"] = prediction_frame["baseline_residual"] ** 2

    return prediction_frame


def run_space_time_benchmark(frame, feature_columns):
    prediction_frames = []
    importance_frames = []
    metrics_records = []

    feature_data = frame[feature_columns]
    target = frame[rf_benchmark.Y_COLUMN].astype(float)

    fitted_fold_id = 0
    for space_fold, time_fold, train_idx, test_idx in rf_benchmark.space_time_splits(frame):
        if len(train_idx) < rf_benchmark.MIN_TRAIN_ROWS or len(test_idx) < rf_benchmark.MIN_TEST_ROWS:
            continue

        x_train = feature_data.loc[train_idx]
        x_test = feature_data.loc[test_idx]
        y_train = target.loc[train_idx]
        y_test = target.loc[test_idx]

        if y_train.nunique(dropna=True) < 2:
            continue

        fold_id = f"space{space_fold}_time{time_fold}"
        model = rf_benchmark.make_rf_model()
        model.fit(x_train, y_train)

        rf_train_prediction = model.predict(x_train)
        rf_test_prediction = model.predict(x_test)
        train_residuals = y_train.to_numpy(dtype=float) - rf_train_prediction

        coordinate_scaler = SpaceTimeCoordinateScaler.fit(frame.loc[train_idx])
        st_x_train, st_y_train, st_z_train = coordinate_scaler.transform(frame.loc[train_idx])
        st_x_test, st_y_test, st_z_test = coordinate_scaler.transform(frame.loc[test_idx])

        residual_kriging, kriging_info = fit_residual_kriging(
            x_train=st_x_train,
            y_train=st_y_train,
            z_train=st_z_train,
            residuals=train_residuals,
        )
        kriged_residual_prediction, kriging_variance, n_closest = predict_residuals(
            kriging=residual_kriging,
            x_test=st_x_test,
            y_test=st_y_test,
            z_test=st_z_test,
            n_train=kriging_info["kriging_n_train"],
        )
        prediction = rf_test_prediction + kriged_residual_prediction
        baseline_prediction = float(y_train.mean())

        prediction_frames.append(
            fold_prediction_frame(
                frame=frame,
                test_idx=test_idx,
                y_test=y_test,
                prediction=prediction,
                rf_mean_prediction=rf_test_prediction,
                kriged_residual_prediction=kriged_residual_prediction,
                kriging_variance=kriging_variance,
                baseline_prediction=baseline_prediction,
                fold_id=fold_id,
            )
        )
        importance_frames.append(
            rf_benchmark.fold_importance_frame(
                model=model,
                feature_columns=feature_columns,
                fold_id=fold_id,
                space_fold=space_fold,
                time_fold=time_fold,
                n_train=len(train_idx),
                n_test=len(test_idx),
            )
        )

        final_metrics = rf_benchmark.score_predictions(y_test, prediction)
        rf_mean_metrics = rf_benchmark.score_predictions(y_test, rf_test_prediction)
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
                "rf_mean_rmse": rf_mean_metrics["rmse"],
                "rf_mean_mae": rf_mean_metrics["mae"],
                "rf_mean_r2": rf_mean_metrics["r2"],
                "baseline_rmse": baseline_metrics["rmse"],
                "baseline_mae": baseline_metrics["mae"],
                "baseline_r2": baseline_metrics["r2"],
                "variogram_model": VARIOGRAM_MODEL,
                "variogram_nlags": VARIOGRAM_NLAGS,
                "variogram_weight": VARIOGRAM_WEIGHT,
                "kriging_backend": KRIGING_BACKEND,
                "kriging_n_closest_points": n_closest,
                "kriging_status": kriging_info["kriging_status"],
                "kriging_n_train": kriging_info["kriging_n_train"],
                "variogram_source": kriging_info["variogram_source"],
                "variogram_psill": kriging_info["variogram_psill"],
                "variogram_range": kriging_info["variogram_range"],
                "variogram_nugget": kriging_info["variogram_nugget"],
                "variogram_structured_sill_ratio": kriging_info[
                    "variogram_structured_sill_ratio"
                ],
                "kriging_time_std_hours": coordinate_scaler.time_std_hours,
                "kriging_time_axis_scale_km": coordinate_scaler.spatial_scale_km
                * TIME_AXIS_SCALE_MULTIPLIER,
                "wrap_longitude_for_kriging": WRAP_LONGITUDE_FOR_KRIGING,
                "mean_kriging_variance": float(np.nanmean(kriging_variance)),
            }
        )

        fitted_fold_id += 1
        print(
            f"Fit {fitted_fold_id}: held out space fold {space_fold}, "
            f"time fold {time_fold}; n_train={len(train_idx):,}, n_test={len(test_idx):,}"
        )

    if not prediction_frames:
        raise ValueError("No valid RF + residual kriging folds were fitted.")

    predictions = pd.concat(prediction_frames, ignore_index=True)
    importances = pd.concat(importance_frames, ignore_index=True)
    metrics = pd.DataFrame(metrics_records)

    return predictions, importances, metrics


def overall_metrics(predictions):
    final_metrics = rf_benchmark.score_predictions(
        predictions["observed"],
        predictions["predicted"],
    )
    rf_mean_metrics = rf_benchmark.score_predictions(
        predictions["observed"],
        predictions["rf_mean_predicted"],
    )
    baseline_metrics = rf_benchmark.score_predictions(
        predictions["observed"],
        predictions["baseline_predicted"],
    )

    return pd.DataFrame(
        [
            {
                "validation_strategy": rf_benchmark.VALIDATION_STRATEGY,
                "n_predictions": len(predictions),
                "n_estimators": rf_benchmark.N_ESTIMATORS,
                "min_samples_leaf": rf_benchmark.MIN_SAMPLES_LEAF,
                "max_features": rf_benchmark.MAX_FEATURES,
                "variogram_model": VARIOGRAM_MODEL,
                "variogram_nlags": VARIOGRAM_NLAGS,
                "variogram_weight": VARIOGRAM_WEIGHT,
                "kriging_backend": KRIGING_BACKEND,
                "max_n_closest_points": N_CLOSEST_POINTS,
                "time_axis_scale_multiplier": TIME_AXIS_SCALE_MULTIPLIER,
                "rmse": final_metrics["rmse"],
                "mae": final_metrics["mae"],
                "r2": final_metrics["r2"],
                "rf_mean_rmse": rf_mean_metrics["rmse"],
                "rf_mean_mae": rf_mean_metrics["mae"],
                "rf_mean_r2": rf_mean_metrics["r2"],
                "baseline_rmse": baseline_metrics["rmse"],
                "baseline_mae": baseline_metrics["mae"],
                "baseline_r2": baseline_metrics["r2"],
            }
        ]
    )


def schema_names(data_path):
    return pq.ParquetFile(data_path).schema_arrow.names


def availability_reference_column(data_path, feature_columns):
    names = set(schema_names(data_path))
    for feature in feature_columns:
        if rf_benchmark.feature_group(feature) != "Atmospheric":
            continue
        if feature in names:
            return feature
        for dependency in rf_benchmark.DERIVED_DEPENDENCIES.get(feature, []):
            if dependency in names:
                return dependency

    return None


def latest_grid_hour(data_path, feature_columns=None):
    filter_expression = None
    if feature_columns is not None:
        reference_column = availability_reference_column(data_path, feature_columns)
        if reference_column is not None:
            filter_expression = ds.field(reference_column).is_valid()

    table = ds.dataset(data_path, format="parquet").to_table(
        columns=[rf_benchmark.TIME_COLUMN],
        filter=filter_expression,
    )
    hours = pd.to_datetime(table[rf_benchmark.TIME_COLUMN].to_pandas())
    if hours.empty:
        raise ValueError("No grid hours with available ERA5 covariates were found.")

    return hours.max()


def parse_grid_hour(value, data_path, feature_columns=None):
    normalized = str(value).strip().lower()
    if normalized == GRID_HOUR_ALL:
        return None, GRID_HOUR_ALL
    if normalized == GRID_HOUR_LATEST:
        hour = latest_grid_hour(data_path, feature_columns=feature_columns)
    else:
        hour = pd.Timestamp(value)

    if pd.isna(hour):
        raise ValueError(f"Could not parse grid prediction hour: {value}")

    return pd.Timestamp(hour).to_pydatetime(), pd.Timestamp(hour).strftime("%Y%m%dT%H%M%S")


def prediction_columns_for_features(data_path, feature_columns):
    names = schema_names(data_path)
    available = set(names)
    needed = {rf_benchmark.TIME_COLUMN, rf_benchmark.LAT_COLUMN, rf_benchmark.LON_COLUMN}

    for feature in feature_columns:
        if feature in available:
            needed.add(feature)
        elif feature in rf_benchmark.DERIVED_DEPENDENCIES:
            needed.update(rf_benchmark.DERIVED_DEPENDENCIES[feature])

    missing = sorted(
        {rf_benchmark.TIME_COLUMN, rf_benchmark.LAT_COLUMN, rf_benchmark.LON_COLUMN}
        - available
    )
    if missing:
        raise ValueError(f"Merged parquet is missing required prediction columns: {missing}")

    return [column for column in names if column in needed]


def grid_filter_expression(data_path, grid_hour, feature_columns):
    names = set(schema_names(data_path))
    filter_expression = None

    if "era5_n_datapoints" in names:
        filter_expression = ds.field("era5_n_datapoints") > 0

    reference_column = availability_reference_column(data_path, feature_columns)
    if reference_column is not None:
        availability_filter = ds.field(reference_column).is_valid()
        if filter_expression is None:
            filter_expression = availability_filter
        else:
            filter_expression = filter_expression & availability_filter

    if grid_hour is not None:
        hour_filter = ds.field(rf_benchmark.TIME_COLUMN) == grid_hour
        if filter_expression is None:
            filter_expression = hour_filter
        else:
            filter_expression = filter_expression & hour_filter

    return filter_expression


def prepare_grid_prediction_frame(frame, feature_columns):
    frame = frame.copy()
    frame = rf_benchmark.add_derived_covariates(frame)
    frame = rf_benchmark.add_time_space_features(frame)
    frame = frame.replace([np.inf, -np.inf], np.nan)

    for column in feature_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    atmospheric_features = [
        feature
        for feature in feature_columns
        if rf_benchmark.feature_group(feature) == "Atmospheric"
    ]
    required_columns = [
        rf_benchmark.TIME_COLUMN,
        rf_benchmark.LAT_COLUMN,
        rf_benchmark.LON_COLUMN,
        *atmospheric_features,
    ]

    return frame.dropna(subset=required_columns).copy()


def fit_final_rf_residual_kriging(frame, feature_columns):
    feature_data = frame[feature_columns]
    target = frame[rf_benchmark.Y_COLUMN].astype(float)

    model = rf_benchmark.make_rf_model()
    model.fit(feature_data, target)

    rf_train_prediction = model.predict(feature_data)
    train_residuals = target.to_numpy(dtype=float) - rf_train_prediction

    coordinate_scaler = SpaceTimeCoordinateScaler.fit(frame)
    st_x_train, st_y_train, st_z_train = coordinate_scaler.transform(frame)
    observation_tree = cKDTree(np.column_stack([st_x_train, st_y_train, st_z_train]))
    residual_kriging, kriging_info = fit_residual_kriging(
        x_train=st_x_train,
        y_train=st_y_train,
        z_train=st_z_train,
        residuals=train_residuals,
    )

    print(
        "Fitted final model: "
        f"n_train={len(frame):,}, kriging_status={kriging_info['kriging_status']}, "
        f"kriging_n_train={kriging_info['kriging_n_train']:,}, "
        f"variogram_source={kriging_info['variogram_source']}, "
        f"variogram_psill={kriging_info['variogram_psill']:.3f}, "
        f"variogram_range={kriging_info['variogram_range']:.3f}, "
        f"variogram_nugget={kriging_info['variogram_nugget']:.3f}"
    )
    kriging_info["training_frame"] = frame[
        [rf_benchmark.TIME_COLUMN, rf_benchmark.LAT_COLUMN, rf_benchmark.LON_COLUMN]
    ].copy()

    return FittedRFResidualKriging(
        model=model,
        coordinate_scaler=coordinate_scaler,
        residual_kriging=residual_kriging,
        kriging_info=kriging_info,
        observation_tree=observation_tree,
    )


def predict_grid_frame(
    fitted,
    frame,
    feature_columns,
    kriging_backend=GRID_KRIGING_BACKEND,
    n_closest_points=GRID_N_CLOSEST_POINTS,
):
    if frame.empty:
        return pd.DataFrame()

    rf_mean_prediction = fitted.model.predict(frame[feature_columns])
    st_x, st_y, st_z = fitted.coordinate_scaler.transform(frame)
    nearest_observation_distance, _ = fitted.observation_tree.query(
        np.column_stack([st_x, st_y, st_z]),
        k=1,
    )
    kriged_residual_prediction, kriging_variance, _ = predict_residuals(
        kriging=fitted.residual_kriging,
        x_test=st_x,
        y_test=st_y,
        z_test=st_z,
        n_train=fitted.kriging_info["kriging_n_train"],
        backend=kriging_backend,
        n_closest_points=n_closest_points,
    )
    prediction = rf_mean_prediction + kriged_residual_prediction

    return pd.DataFrame(
        {
            rf_benchmark.TIME_COLUMN: pd.to_datetime(frame[rf_benchmark.TIME_COLUMN]),
            rf_benchmark.LAT_COLUMN: frame[rf_benchmark.LAT_COLUMN].to_numpy(dtype=float),
            rf_benchmark.LON_COLUMN: frame[rf_benchmark.LON_COLUMN].to_numpy(dtype=float),
            "predicted_radiance": prediction,
            "predicted_radiance_variance": kriging_variance,
            "rf_mean_predicted_radiance": rf_mean_prediction,
            "kriged_residual_radiance": kriged_residual_prediction,
            "kriging_variance": kriging_variance,
            "nearest_observation_distance_km": nearest_observation_distance,
        }
    )


def write_grid_predictions(
    fitted,
    feature_columns,
    data_path,
    output_dir,
    grid_hour=GRID_HOUR_LATEST,
    batch_size=GRID_BATCH_SIZE,
    max_grid_rows=None,
    kriging_backend=GRID_KRIGING_BACKEND,
    n_closest_points=GRID_N_CLOSEST_POINTS,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parsed_hour, hour_label = parse_grid_hour(
        grid_hour,
        data_path,
        feature_columns=feature_columns,
    )
    output_path = output_dir / f"{OUTPUT_PREFIX}_grid_predictions_{hour_label}.parquet"
    summary_path = output_dir / f"{OUTPUT_PREFIX}_grid_predictions_{hour_label}_summary.csv"
    plot_path = output_dir / f"{OUTPUT_PREFIX}_grid_predictions_{hour_label}.png"

    scanner = ds.dataset(data_path, format="parquet").scanner(
        columns=prediction_columns_for_features(data_path, feature_columns),
        filter=grid_filter_expression(data_path, parsed_hour, feature_columns),
        batch_size=batch_size,
    )

    writer = None
    n_seen = 0
    n_written = 0
    n_batches = 0

    try:
        for batch in scanner.to_batches():
            n_batches += 1
            n_seen += batch.num_rows

            prediction_input = prepare_grid_prediction_frame(
                batch.to_pandas(),
                feature_columns=feature_columns,
            )
            if max_grid_rows is not None:
                remaining = max_grid_rows - n_written
                if remaining <= 0:
                    break
                prediction_input = prediction_input.head(remaining)

            grid_predictions = predict_grid_frame(
                fitted=fitted,
                frame=prediction_input,
                feature_columns=feature_columns,
                kriging_backend=kriging_backend,
                n_closest_points=n_closest_points,
            )

            if grid_predictions.empty:
                if max_grid_rows is not None and n_written >= max_grid_rows:
                    break
                continue

            table = pa.Table.from_pandas(grid_predictions, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)

            n_written += len(grid_predictions)
            if n_batches == 1 or n_batches % 10 == 0:
                print(
                    f"Grid prediction batches={n_batches:,}; "
                    f"source_rows={n_seen:,}; written_rows={n_written:,}"
                )

            if max_grid_rows is not None and n_written >= max_grid_rows:
                break
    finally:
        if writer is not None:
            writer.close()

    if n_written == 0:
        raise ValueError("No grid rows with usable ERA5 covariates were predicted.")

    summary = pd.DataFrame(
        [
            {
                "grid_hour": GRID_HOUR_ALL if parsed_hour is None else pd.Timestamp(parsed_hour),
                "n_source_rows_scanned": n_seen,
                "n_predictions": n_written,
                "batch_size": batch_size,
                "output_path": str(output_path),
                "variogram_model": VARIOGRAM_MODEL,
                "variogram_nlags": VARIOGRAM_NLAGS,
                "variogram_weight": VARIOGRAM_WEIGHT,
                "kriging_backend": kriging_backend,
                "kriging_n_closest_points": n_closest_points,
                "kriging_n_train": fitted.kriging_info["kriging_n_train"],
                "variogram_source": fitted.kriging_info["variogram_source"],
                "variogram_psill": fitted.kriging_info["variogram_psill"],
                "variogram_range": fitted.kriging_info["variogram_range"],
                "variogram_nugget": fitted.kriging_info["variogram_nugget"],
                "variogram_structured_sill_ratio": fitted.kriging_info[
                    "variogram_structured_sill_ratio"
                ],
                "time_axis_scale_multiplier": TIME_AXIS_SCALE_MULTIPLIER,
                "wrap_longitude_for_kriging": WRAP_LONGITUDE_FOR_KRIGING,
                "time_axis_scale_km": fitted.coordinate_scaler.spatial_scale_km
                * TIME_AXIS_SCALE_MULTIPLIER,
            }
        ]
    )
    summary.to_csv(summary_path, index=False)

    if parsed_hour is not None:
        plot_grid_predictions(output_path, plot_path, training_frame=fitted.kriging_info.get("training_frame"))

    print(f"Saved grid predictions to {output_path}")
    print(f"Saved grid prediction summary to {summary_path}")
    if parsed_hour is not None:
        print(f"Saved grid prediction map to {plot_path}")

    return output_path, summary_path, plot_path if parsed_hour is not None else None


def plot_grid_predictions(prediction_path, output_path, training_frame=None):
    predictions = pd.read_parquet(
        prediction_path,
        columns=[
            rf_benchmark.TIME_COLUMN,
            rf_benchmark.LAT_COLUMN,
            rf_benchmark.LON_COLUMN,
            "predicted_radiance",
            "predicted_radiance_variance",
        ],
    )

    if predictions.empty:
        return

    prediction_grid = (
        predictions.pivot_table(
            index=rf_benchmark.LAT_COLUMN,
            columns=rf_benchmark.LON_COLUMN,
            values="predicted_radiance",
            aggfunc="mean",
        )
        .sort_index()
        .sort_index(axis=1)
    )
    variance_grid = (
        predictions.pivot_table(
            index=rf_benchmark.LAT_COLUMN,
            columns=rf_benchmark.LON_COLUMN,
            values="predicted_radiance_variance",
            aggfunc="mean",
        )
        .sort_index()
        .sort_index(axis=1)
    )

    extent = [
        float(prediction_grid.columns.min()),
        float(prediction_grid.columns.max()),
        float(prediction_grid.index.min()),
        float(prediction_grid.index.max()),
    ]
    if extent[0] == extent[1]:
        extent[0] -= 0.5
        extent[1] += 0.5
    if extent[2] == extent[3]:
        extent[2] -= 0.5
        extent[3] += 0.5

    fig, axes = rf_benchmark.plt.subplots(2, 1, figsize=(20, 10), constrained_layout=True)

    image = axes[0].imshow(
        prediction_grid.to_numpy(),
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="viridis",
    )
    fig.colorbar(image, ax=axes[0], label="Predicted radiance")
    axes[0].set_title("Grid predicted radiance")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")

    image = axes[1].imshow(
        variance_grid.to_numpy(),
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="magma",
    )
    fig.colorbar(image, ax=axes[1], label="Residual kriging variance")
    axes[1].set_title("Grid residual kriging variance")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")

    if training_frame is not None:
        prediction_hour = pd.to_datetime(predictions[rf_benchmark.TIME_COLUMN]).iloc[0]
        observed_at_hour = training_frame[
            pd.to_datetime(training_frame[rf_benchmark.TIME_COLUMN]) == prediction_hour
        ]
        if not observed_at_hour.empty:
            for ax in axes:
                ax.scatter(
                    observed_at_hour[rf_benchmark.LON_COLUMN],
                    observed_at_hour[rf_benchmark.LAT_COLUMN],
                    s=12,
                    c="white",
                    edgecolor="black",
                    linewidth=0.4,
                    alpha=0.9,
                )

    fig.savefig(output_path, dpi=200)
    rf_benchmark.plt.close(fig)


def plot_space_time_predictions(predictions, fold_metrics, output_dir, prefix=OUTPUT_PREFIX):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = rf_benchmark.plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)

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
        predictions[rf_benchmark.TIME_COLUMN],
        predictions["residual"],
        c=predictions[rf_benchmark.TIME_FOLD_COLUMN],
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
        predictions[rf_benchmark.LON_COLUMN],
        predictions[rf_benchmark.LAT_COLUMN],
        c=predictions["residual"],
        cmap="coolwarm",
        norm=rf_benchmark.residual_norm(predictions["residual"]),
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
        index=rf_benchmark.TIME_FOLD_COLUMN,
        columns=rf_benchmark.SPACE_FOLD_COLUMN,
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

    fig.suptitle("Random forest + residual 3D kriging space-time blocked validation")
    output_path = output_dir / f"{prefix}_predictions.png"
    fig.savefig(output_path, dpi=200)
    rf_benchmark.plt.close(fig)

    print(f"Saved {output_path}")


def save_outputs(frame, predictions, importances, importance_summary, metrics, overall):
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
    importances.to_csv(
        OUT_DIR / f"{OUTPUT_PREFIX}_feature_importances_by_fold.csv",
        index=False,
    )
    importance_summary.to_csv(
        OUT_DIR / f"{OUTPUT_PREFIX}_feature_importance_summary.csv",
        index=False,
    )
    metrics.to_csv(OUT_DIR / f"{OUTPUT_PREFIX}_metrics_by_fold.csv", index=False)
    overall.to_csv(OUT_DIR / f"{OUTPUT_PREFIX}_overall_metrics.csv", index=False)

    print(f"Saved outputs in {OUT_DIR}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Random forest mean model plus PyKrige 3D residual kriging."
    )
    parser.add_argument(
        "--mode",
        choices=["validate", "predict-grid", "both"],
        default="validate",
        help="Run blocked validation, grid prediction, or both.",
    )
    parser.add_argument(
        "--grid-hour",
        default=GRID_HOUR_LATEST,
        help=(
            "Grid prediction hour. Use 'latest' for the latest hour in the parquet, "
            "'all' for all hours, or an explicit timestamp."
        ),
    )
    parser.add_argument(
        "--grid-batch-size",
        type=int,
        default=GRID_BATCH_SIZE,
        help="Rows per streamed grid prediction batch.",
    )
    parser.add_argument(
        "--grid-kriging-backend",
        choices=["vectorized", "loop"],
        default=GRID_KRIGING_BACKEND,
        help="PyKrige backend for grid predictions.",
    )
    parser.add_argument(
        "--grid-n-closest-points",
        type=int,
        default=GRID_N_CLOSEST_POINTS,
        help=(
            "Optional moving-window size for grid kriging. PyKrige requires "
            "the loop backend when this is set."
        ),
    )
    parser.add_argument(
        "--max-grid-rows",
        type=int,
        default=None,
        help="Optional cap for smoke-testing grid prediction.",
    )
    parser.add_argument(
        "--grid-output-dir",
        type=Path,
        default=GRID_OUTPUT_DIR,
        help="Directory for grid prediction parquet, summary, and map outputs.",
    )

    args = parser.parse_args()

    if args.grid_n_closest_points is not None:
        if args.grid_n_closest_points < 2:
            parser.error("--grid-n-closest-points must be at least 2 when set.")
        if args.grid_kriging_backend != "loop":
            parser.error("--grid-n-closest-points requires --grid-kriging-backend loop.")

    return args


def main():
    args = parse_args()
    require_pykrige()

    frame, feature_columns = rf_benchmark.load_model_data(rf_benchmark.DATA_PATH)

    if args.mode in {"validate", "both"}:
        validation_frame = rf_benchmark.assign_space_time_folds(frame.copy())

        predictions, importances, metrics = run_space_time_benchmark(
            validation_frame,
            feature_columns,
        )
        importance_summary = rf_benchmark.summarize_importances(importances)
        overall = overall_metrics(predictions)

        save_outputs(validation_frame, predictions, importances, importance_summary, metrics, overall)
        plot_space_time_predictions(predictions, metrics, OUT_DIR)
        rf_benchmark.plot_space_time_importances(
            importance_summary,
            OUT_DIR,
            prefix=OUTPUT_PREFIX,
        )

        print(f"Validation strategy: {rf_benchmark.VALIDATION_STRATEGY}")
        print(f"Rows used: {len(validation_frame):,}")
        print(f"Features used: {len(feature_columns):,}")
        print(
            "Space folds: "
            f"{validation_frame[rf_benchmark.SPACE_FOLD_COLUMN].nunique():,}"
        )
        print(
            "Time folds: "
            f"{validation_frame[rf_benchmark.TIME_FOLD_COLUMN].nunique():,}"
        )
        print("\nOverall out-of-fold metrics:")
        print(overall.to_string(index=False))
        print("\nTop RF mean-function feature importance summary:")
        print(importance_summary.head(20).to_string(index=False))

    if args.mode in {"predict-grid", "both"}:
        fitted = fit_final_rf_residual_kriging(frame, feature_columns)
        write_grid_predictions(
            fitted=fitted,
            feature_columns=feature_columns,
            data_path=rf_benchmark.DATA_PATH,
            output_dir=args.grid_output_dir,
            grid_hour=args.grid_hour,
            batch_size=args.grid_batch_size,
            max_grid_rows=args.max_grid_rows,
            kriging_backend=args.grid_kriging_backend,
            n_closest_points=args.grid_n_closest_points,
        )


if __name__ == "__main__":
    main()
