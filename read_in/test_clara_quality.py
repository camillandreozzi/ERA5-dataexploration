"""Run with python3 -B -m unittest read_in.test_clara_quality."""

import unittest

import numpy as np
import pandas as pd

from read_in.temporal_subset.data_preprocessing import (
    CLARA_LATITUDE_COLUMN,
    CLARA_LONGITUDE_COLUMN,
    CLARA_RADIANCE_COLUMN,
    CLARA_TIME_COLUMN,
    RADIANCE_MAX,
    RADIANCE_MIN,
    aggregate_clara_hourly,
    radiance_quality_mask,
    radiance_quality_report,
)


# One ERA5 cell, so every observation snaps to the same (hour, lon, lat) key
# and the tests are about the radiance filter rather than the grid snapping.
LATITUDE = np.array([10.0, 11.0])
LONGITUDE = np.array([20.0, 21.0])


def observations(radiances, hours=None):
    """CLARA rows that all land on the (10.0, 20.0) cell of the grid above."""
    hours = hours if hours is not None else ["2020-12-01 00:00"] * len(radiances)
    return pd.DataFrame({
        CLARA_TIME_COLUMN: pd.to_datetime(hours),
        CLARA_RADIANCE_COLUMN: np.asarray(radiances, dtype=float),
        CLARA_LATITUDE_COLUMN: [10.0] * len(radiances),
        CLARA_LONGITUDE_COLUMN: [20.0] * len(radiances),
    })


class RadianceQualityMaskTests(unittest.TestCase):
    def test_keeps_only_finite_values_inside_the_range(self):
        values = pd.Series([
            -37.2,          # negative, the temporal subset's failure mode
            RADIANCE_MIN,   # exactly zero is rejected: log is undefined
            1.0,
            RADIANCE_MAX,   # the upper bound itself is kept
            RADIANCE_MAX + 1e-6,
            91_912.7,       # the calibration cluster near 9e4
            np.inf,
            np.nan,
        ])
        np.testing.assert_array_equal(
            radiance_quality_mask(values),
            [False, False, True, True, False, False, False, False],
        )

    def test_report_attributes_each_rejection_to_one_reason(self):
        report = radiance_quality_report(pd.Series([-1.0, 0.0, 9e4, np.inf, np.nan]))
        counts = dict(zip(report["reason"], report["n"]))
        self.assertEqual(counts["nonfinite"], 2)
        self.assertEqual(counts[f"<= {RADIANCE_MIN:g}"], 2)
        self.assertEqual(counts[f"> {RADIANCE_MAX:g}"], 1)


class AggregateClaraHourlyTests(unittest.TestCase):
    def test_rejected_rows_leave_the_mean_and_the_count(self):
        # Without the filter the mean would be (100 + 91912.7) / 2 and the
        # count 2, which is how the calibration cluster used to reach the
        # models: the contaminated mean stays positive, so add_log_radiance
        # never notices it.
        hourly, _ = aggregate_clara_hourly(
            observations([100.0, 91_912.7]), LATITUDE, LONGITUDE
        )
        self.assertEqual(len(hourly), 1)
        self.assertAlmostEqual(float(hourly["clara_radiance_hourly_mean"].iloc[0]), 100.0)
        self.assertEqual(int(hourly["clara_n_datapoints"].iloc[0]), 1)

    def test_cell_with_no_valid_observation_is_not_emitted(self):
        hourly, _ = aggregate_clara_hourly(observations([-13.7, 9e4]), LATITUDE, LONGITUDE)
        self.assertTrue(hourly.empty)

    def test_valid_observations_still_average(self):
        hourly, _ = aggregate_clara_hourly(observations([100.0, 200.0]), LATITUDE, LONGITUDE)
        self.assertAlmostEqual(float(hourly["clara_radiance_hourly_mean"].iloc[0]), 150.0)
        self.assertEqual(int(hourly["clara_n_datapoints"].iloc[0]), 2)

    def test_rejection_in_one_hour_leaves_another_hour_untouched(self):
        hourly, _ = aggregate_clara_hourly(
            observations(
                [-13.7, 180.0],
                hours=["2020-12-01 00:00", "2020-12-01 01:00"],
            ),
            LATITUDE,
            LONGITUDE,
        )
        self.assertEqual(len(hourly), 1)
        self.assertEqual(hourly["hour"].iloc[0], pd.Timestamp("2020-12-01 01:00"))
        self.assertAlmostEqual(float(hourly["clara_radiance_hourly_mean"].iloc[0]), 180.0)


if __name__ == "__main__":
    unittest.main()
