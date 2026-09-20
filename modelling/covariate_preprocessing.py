"""Shared feature typing for the mean models, including RF residual kriging."""

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Nominal ECMWF codes. Cover fractions, LAI, and land-sea fraction are continuous.
CATEGORICAL_COVARIATES = frozenset({"era5_tvl", "era5_tvh", "era5_slt"})

LOCAL_TIME_COLUMN = "local_time"
LOG_RADIANCE_COLUMN = "log_radiance"


def add_local_time(frame):
    """Use the merged local clock time, or derive CET for older merged files."""
    if LOCAL_TIME_COLUMN not in frame:
        local = pd.to_datetime(frame["hour"], utc=True).dt.tz_convert(ZoneInfo("CET"))
        frame[LOCAL_TIME_COLUMN] = (
            local.dt.hour + local.dt.minute / 60 + local.dt.second / 3600
        ).astype(np.float32)
    values = pd.to_numeric(frame[LOCAL_TIME_COLUMN], errors="coerce")
    if (~np.isfinite(values) | (values < 0) | (values >= 24)).any():
        raise ValueError("local_time must be finite and in [0, 24) for every row.")
    frame[LOCAL_TIME_COLUMN] = values
    return frame


def add_log_radiance(frame, radiance_column):
    """Keep finite, strictly positive observations and preserve raw radiance."""
    radiance = pd.to_numeric(frame[radiance_column], errors="coerce")
    keep = np.isfinite(radiance) & (radiance > 0)
    print(f"Excluded {(~keep).sum():,} nonpositive/nonfinite radiance rows before log transform.")
    frame = frame.loc[keep].copy()
    if frame.empty:
        raise ValueError("No positive radiance observations remain for log modelling.")
    frame[LOG_RADIANCE_COLUMN] = np.log(radiance.loc[keep].astype(float))
    return frame


def categorical_columns(frame):
    return [name for name in frame.columns if name in CATEGORICAL_COVARIATES]


def continuous_columns(frame):
    return [name for name in frame.columns if name not in CATEGORICAL_COVARIATES]


def make_covariate_preprocessor():
    """Learn imputations and categories from training rows only.

    Missing categories get a dedicated sentinel; unseen prediction categories
    encode as all zeros. Neither case introduces an ordering of category codes.
    """
    return ColumnTransformer(
        [
            ("continuous", SimpleImputer(strategy="median", keep_empty_features=True),
             continuous_columns),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value=-1,
                                          keep_empty_features=True)),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical_columns),
        ],
        verbose_feature_names_out=False,
    )
