"""Space-time blocked cross-validation settings shared by the mean models.

The defaults reproduce the temporal-subset runs. The spatial subset (Italy,
2020) has only ~200 hourly CLARA cells, so euler/env.sh lowers them through
these environment variables instead of editing the scripts:

    ERA5_N_SPACE_FOLDS   number of k-means location clusters     (default 4)
    ERA5_N_TIME_FOLDS    number of contiguous time blocks        (default 4)
    ERA5_MIN_TRAIN_ROWS  skip a fold with fewer training rows    (default 50)
    ERA5_MIN_TEST_ROWS   skip a fold with fewer test rows        (default 5)
"""

import os


def _int_env(name, default):
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


N_SPACE_FOLDS = _int_env("ERA5_N_SPACE_FOLDS", 4)
N_TIME_FOLDS = _int_env("ERA5_N_TIME_FOLDS", 4)
MIN_TRAIN_ROWS = _int_env("ERA5_MIN_TRAIN_ROWS", 50)
MIN_TEST_ROWS = _int_env("ERA5_MIN_TEST_ROWS", 5)
