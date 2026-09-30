"""Run with python3 -B -m unittest read_in.test_clara_source."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

for _p in Path(__file__).resolve().parents:
    if (_p / "paths.py").exists():
        sys.path.insert(0, str(_p))
        break
from paths import data_path

from read_in.clara_source import (
    GEO_GAP_COLUMN,
    LATITUDE_COLUMN,
    LOCAL_TIME_COLUMN,
    LONGITUDE_COLUMN,
    POSITIVE_LONGITUDE_COLUMN,
    RADIANCE_COLUMN,
    TIME_COLUMN,
    _deduplicate_timestamps,
    _is_dead_olr_frame,
    _vector_to_degrees,
    add_longitude_and_local_time,
    interpolate_geolocation,
    jday_to_datetime,
    parse_vector_column,
)


def olr_frame(times, radiances):
    return pd.DataFrame({
        TIME_COLUMN: pd.to_datetime(times),
        RADIANCE_COLUMN: np.asarray(radiances, dtype=float),
    })


def geo_frame(times, latitudes, longitudes, view_angles=None):
    n = len(latitudes)
    return pd.DataFrame({
        "time": pd.to_datetime(times),
        "view_angle": np.zeros(n) if view_angles is None else np.asarray(view_angles, float),
        "latitude": np.asarray(latitudes, dtype=float),
        "longitude": np.asarray(longitudes, dtype=float),
    })


class JulianDayTests(unittest.TestCase):
    def test_round_trips_a_known_pair(self):
        # 2020-05-01 00:00:51 UTC, taken from CLARA_OLR_2020_5.save.
        converted = jday_to_datetime([2458970.5005902778])[0]
        self.assertEqual(converted.floor("s"), pd.Timestamp("2020-05-01 00:00:51"))

    def test_unix_epoch(self):
        self.assertEqual(jday_to_datetime([2440587.5])[0], pd.Timestamp("1970-01-01"))


class BigEndianTests(unittest.TestCase):
    def test_big_endian_input_survives_a_groupby(self):
        """The IDL fields are >f8; left that way a later groupby raises."""
        jday = np.array([2458970.5, 2458970.5 + 1 / 2880], dtype=">f8")
        olr = np.array([100.0, 200.0], dtype=">f8")

        frame = pd.DataFrame({
            TIME_COLUMN: jday_to_datetime(jday),
            RADIANCE_COLUMN: np.asarray(olr, dtype="<f8"),
        })
        grouped = frame.groupby(frame[TIME_COLUMN].dt.floor("h"))[RADIANCE_COLUMN].mean()
        self.assertAlmostEqual(float(grouped.iloc[0]), 150.0)


class DeadFileTests(unittest.TestCase):
    def test_small_all_zero_file_is_dead(self):
        frame = olr_frame(["2021-09-01"] * 30, [0.0] * 30)
        self.assertTrue(_is_dead_olr_frame(frame))

    def test_small_non_zero_file_is_kept(self):
        frame = olr_frame(["2021-09-01"] * 30, [0.0] * 29 + [120.0])
        self.assertFalse(_is_dead_olr_frame(frame))

    def test_large_all_zero_file_is_kept(self):
        frame = olr_frame(["2021-09-01"] * 1500, [0.0] * 1500)
        self.assertFalse(_is_dead_olr_frame(frame))


class DeduplicationTests(unittest.TestCase):
    def test_exact_duplicates_collapse(self):
        frame = olr_frame(["2020-05-01 00:00:30", "2020-05-01 00:00:30"], [100.0, 100.0])
        self.assertEqual(len(_deduplicate_timestamps(frame, verbose=False)), 1)

    def test_small_disagreement_is_tolerated(self):
        """The one real collision in the delivery, 381.562 vs 380.909."""
        frame = olr_frame(
            ["2024-09-20 00:01:28", "2024-09-20 00:01:28"], [381.562231, 380.908871]
        )
        kept = _deduplicate_timestamps(frame, verbose=False)
        self.assertEqual(len(kept), 1)
        self.assertAlmostEqual(float(kept[RADIANCE_COLUMN].iloc[0]), 381.562231)

    def test_wide_disagreement_raises(self):
        frame = olr_frame(["2020-05-01 00:00:30"] * 2, [100.0, 87000.0])
        with self.assertRaises(ValueError):
            _deduplicate_timestamps(frame, verbose=False)


class VectorParsingTests(unittest.TestCase):
    def test_parses_a_triple(self):
        parts = parse_vector_column(pd.Series(["(1.0,2.0,3.0)", "(-4.5,0,6)"]))
        self.assertEqual(list(parts.columns), ["x", "y", "z"])
        self.assertAlmostEqual(float(parts["y"].iloc[0]), 2.0)
        self.assertAlmostEqual(float(parts["x"].iloc[1]), -4.5)

    def test_malformed_row_raises(self):
        with self.assertRaises(ValueError):
            parse_vector_column(pd.Series(["(1.0,2.0,3.0)", "(1.0,2.0)"]))

    def test_scalar_in_parentheses_is_recognised(self):
        latitude = parse_vector_column(pd.Series(["(45.0,0,0)", "(-30.0,0,0)"]))
        longitude = parse_vector_column(pd.Series(["(10.0,0,0)", "(-170.0,0,0)"]))
        lat, lon = _vector_to_degrees(latitude, longitude)
        np.testing.assert_allclose(lat, [45.0, -30.0])
        np.testing.assert_allclose(lon, [10.0, -170.0])

    def test_unit_vectors_are_converted(self):
        # 0 N, 90 E and 90 N.
        latitude = parse_vector_column(pd.Series(["(0,1,0)", "(0,0,1)"]))
        longitude = latitude
        lat, lon = _vector_to_degrees(latitude, longitude)
        np.testing.assert_allclose(lat, [0.0, 90.0], atol=1e-9)
        np.testing.assert_allclose(lon[0], 90.0, atol=1e-9)

    def test_unrecognised_encoding_raises(self):
        latitude = parse_vector_column(pd.Series(["(7.0,3.0,9.0)"]))
        with self.assertRaises(ValueError):
            _vector_to_degrees(latitude, latitude)


class InterpolationTests(unittest.TestCase):
    def test_interpolates_across_the_antimeridian(self):
        """-179.9 and 179.9 must meet at +/-180, not at 0."""
        geo = geo_frame(
            ["2020-05-01 00:00:00", "2020-05-01 00:01:00"], [0.0, 0.0], [-179.9, 179.9]
        )
        olr = olr_frame(["2020-05-01 00:00:30"], [120.0])

        located = interpolate_geolocation(olr, geo, max_gap_seconds=90.0)
        longitude = float(located[LONGITUDE_COLUMN].iloc[0])
        self.assertAlmostEqual(abs(longitude), 180.0, places=6)

    def test_midpoint_along_a_meridian(self):
        """At constant longitude the great circle is the meridian, so the
        interpolated latitude is the plain mean."""
        geo = geo_frame(
            ["2020-05-01 00:00:00", "2020-05-01 00:01:00"], [10.0, 20.0], [30.0, 30.0]
        )
        olr = olr_frame(["2020-05-01 00:00:30"], [120.0])

        located = interpolate_geolocation(olr, geo, max_gap_seconds=90.0)
        self.assertAlmostEqual(float(located[LATITUDE_COLUMN].iloc[0]), 15.0, places=6)
        self.assertAlmostEqual(float(located[LONGITUDE_COLUMN].iloc[0]), 30.0, places=6)

    def test_midpoint_along_the_equator(self):
        geo = geo_frame(
            ["2020-05-01 00:00:00", "2020-05-01 00:01:00"], [0.0, 0.0], [30.0, 40.0]
        )
        olr = olr_frame(["2020-05-01 00:00:30"], [120.0])

        located = interpolate_geolocation(olr, geo, max_gap_seconds=90.0)
        self.assertAlmostEqual(float(located[LONGITUDE_COLUMN].iloc[0]), 35.0, places=6)

    def test_oblique_midpoint_follows_the_great_circle(self):
        """Off the meridian and off the equator the great-circle midpoint is
        NOT the arithmetic mean of the angles, and must not be."""
        geo = geo_frame(
            ["2020-05-01 00:00:00", "2020-05-01 00:01:00"], [10.0, 20.0], [30.0, 40.0]
        )
        olr = olr_frame(["2020-05-01 00:00:30"], [120.0])

        located = interpolate_geolocation(olr, geo, max_gap_seconds=90.0)
        latitude = float(located[LATITUDE_COLUMN].iloc[0])
        self.assertGreater(latitude, 15.0)
        self.assertAlmostEqual(latitude, 15.0547, places=3)

    def test_gap_beyond_tolerance_is_nan(self):
        geo = geo_frame(
            ["2020-05-01 00:00:00", "2020-05-01 00:10:00"], [10.0, 20.0], [30.0, 40.0]
        )
        olr = olr_frame(["2020-05-01 00:05:00"], [120.0])

        located = interpolate_geolocation(olr, geo, max_gap_seconds=90.0)
        self.assertTrue(np.isnan(located[LATITUDE_COLUMN].iloc[0]))
        self.assertAlmostEqual(float(located[GEO_GAP_COLUMN].iloc[0]), 300.0)

    def test_outside_the_track_is_nan(self):
        geo = geo_frame(
            ["2020-05-01 00:00:00", "2020-05-01 00:01:00"], [10.0, 20.0], [30.0, 40.0]
        )
        olr = olr_frame(["2020-05-01 01:00:00"], [120.0])

        located = interpolate_geolocation(olr, geo, max_gap_seconds=90.0)
        self.assertTrue(np.isnan(located[LATITUDE_COLUMN].iloc[0]))

    def test_tolerance_defaults_to_three_times_the_cadence(self):
        times = pd.date_range("2020-05-01", periods=5, freq="1s")
        geo = geo_frame(times, [10.0] * 5, [30.0] * 5)
        olr = olr_frame(["2020-05-01 00:00:02"], [120.0])

        located = interpolate_geolocation(olr, geo)
        self.assertFalse(np.isnan(located[LATITUDE_COLUMN].iloc[0]))


class LocalTimeTests(unittest.TestCase):
    def located(self, times, longitudes):
        frame = olr_frame(times, [120.0] * len(times))
        frame[LATITUDE_COLUMN] = 0.0
        frame[LONGITUDE_COLUMN] = np.asarray(longitudes, dtype=float)
        return add_longitude_and_local_time(frame)

    def test_greenwich_noon(self):
        result = self.located(["2020-05-01 12:00:00"], [0.0])
        self.assertAlmostEqual(float(result[LOCAL_TIME_COLUMN].iloc[0]), 12.0)

    def test_antimeridian_midnight_is_noon(self):
        result = self.located(["2020-05-01 00:00:00"], [180.0])
        self.assertAlmostEqual(float(result[LOCAL_TIME_COLUMN].iloc[0]), 12.0)

    def test_wraps_backwards_past_midnight(self):
        result = self.located(["2020-05-01 00:30:00"], [-15.0])
        self.assertAlmostEqual(float(result[LOCAL_TIME_COLUMN].iloc[0]), 23.5)

    def test_sub_second_precision_is_kept(self):
        result = self.located(["2020-05-01 00:00:00.500000"], [0.0])
        self.assertGreater(float(result[LOCAL_TIME_COLUMN].iloc[0]), 0.0)

    def test_positive_longitude_matches(self):
        result = self.located(["2020-05-01 00:00:00"] * 3, [-179.5, 0.0, 179.5])
        positive = result[POSITIVE_LONGITUDE_COLUMN]
        self.assertTrue(((positive >= 0.0) & (positive < 360.0)).all())
        np.testing.assert_allclose(positive.to_numpy(), [180.5, 0.0, 179.5])

    def test_longitude_is_normalised_into_range(self):
        result = self.located(["2020-05-01 00:00:00"] * 2, [190.0, -190.0])
        np.testing.assert_allclose(result[LONGITUDE_COLUMN].to_numpy(), [-170.0, 170.0])


OLD_PICKLE = data_path("CLARA.pkl")


@unittest.skipUnless(OLD_PICKLE.exists(), "the reference CLARA.pkl is not on disk")
class OldPickleRegressionTests(unittest.TestCase):
    """The new OLR must be the same physical quantity the old pickle carried.

    Matching May 2020 by nearest timestamp, the two correlate at r > 0.9999 and
    differ by a near-constant offset. A drop in r means the loader is reading the
    wrong field or mis-converting the Julian day. The offset itself is a question
    for the data producer, not something this pipeline resolves, so it is only
    bounded loosely here.
    """

    def test_may_2020_matches_the_old_pickle(self):
        from read_in.clara_source import load_olr

        old = pd.read_pickle(OLD_PICKLE)
        old[TIME_COLUMN] = pd.to_datetime(old[TIME_COLUMN])
        old = old.loc[
            old[TIME_COLUMN].between("2020-05-01", "2020-06-01"),
            [TIME_COLUMN, RADIANCE_COLUMN],
        ].sort_values(TIME_COLUMN)

        new = load_olr(start="2020-05-01", end="2020-06-01", verbose=False)

        matched = pd.merge_asof(
            old, new, on=TIME_COLUMN, direction="nearest",
            tolerance=pd.Timedelta("40s"), suffixes=("_old", "_new"),
        ).dropna()

        self.assertGreater(len(matched), 10_000)
        correlation = matched[f"{RADIANCE_COLUMN}_old"].corr(matched[f"{RADIANCE_COLUMN}_new"])
        self.assertGreater(correlation, 0.9999)

        offset = (matched[f"{RADIANCE_COLUMN}_old"] - matched[f"{RADIANCE_COLUMN}_new"]).median()
        self.assertAlmostEqual(offset, -6.53, places=1)


if __name__ == "__main__":
    unittest.main()
