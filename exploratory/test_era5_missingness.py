"""Run with python3 -B -m unittest exploratory.test_era5_missingness."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from exploratory.explore_era5 import (
    SpaceTimeMissingness, plot_space_time_missingness, save_space_time_missingness,
)


class MissingnessTests(unittest.TestCase):
    def test_counts_are_exact_across_batches_and_outputs(self):
        frame = pd.DataFrame({
            "hour": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02",
                                    "2020-01-02", "2020-01-01", "2020-01-02"]),
            "latitude": [0., 1., 0., 1., 2., 2.],
            "longitude": [10., 11., 10., 11., 12., 12.],
            "era5_a": [0., np.nan, 2., np.inf, -1., np.nan],
            "era5_b": [np.nan, 1., 0., 2., -np.inf, 3.],
        })
        variables = ["era5_a", "era5_b"]
        for batch_size in (1, 3, 6):
            counts = SpaceTimeMissingness(variables)
            for start in range(0, len(frame), batch_size):
                counts.update(frame.iloc[start:start + batch_size])
            for index, variable in enumerate(variables):
                expected = frame.assign(missing=~np.isfinite(frame[variable]))
                for keys, actual in ((["latitude", "longitude"], counts.spatial_frame(index)),
                                     (["hour"], counts.temporal_frame().query("variable == @variable"))):
                    grouped = expected.groupby(keys).missing.agg(["size", "sum", "mean"])
                    actual = actual.set_index(keys).sort_index()
                    np.testing.assert_array_equal(actual.n_rows, grouped["size"])
                    np.testing.assert_array_equal(actual.n_missing, grouped["sum"])
                    np.testing.assert_allclose(actual.missing_rate, grouped["mean"])
            self.assertEqual(counts.spatial_counts[:, 0].sum(), len(frame))

        with TemporaryDirectory() as tmp:
            output = Path(tmp)
            save_space_time_missingness(counts, output)
            saved = pd.read_parquet(output / "covariate_missingness_by_location.parquet")
            self.assertEqual(len(saved), 6)
            self.assertEqual(saved.n_missing.sum(), 5)
            plot_space_time_missingness(counts, output)
            self.assertGreater((output / "covariate_missingness_by_hour.png").stat().st_size, 1000)
            self.assertGreater((output / "covariate_missingness_spatial_01.png").stat().st_size, 1000)

    def test_invalid_coordinates_fail_instead_of_losing_rows(self):
        counts = SpaceTimeMissingness(["era5_a"])
        with self.assertRaises(ValueError):
            counts.update(pd.DataFrame({"hour": [pd.Timestamp("2020-01-01")],
                                        "latitude": [np.nan], "longitude": [0.],
                                        "era5_a": [1.]}))


if __name__ == "__main__":
    unittest.main()
