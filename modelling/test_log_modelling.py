"""Run with python3 -B -m unittest modelling.test_log_modelling."""

from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from modelling import benchmark_lasso as lasso, benchmark_rf as rf, rf_stkriging as kriging
from modelling.covariate_preprocessing import add_local_time, add_log_radiance


class LogModellingTests(unittest.TestCase):
    def test_positive_target_preserves_raw_values(self):
        frame = pd.DataFrame({rf.Y_COLUMN: [-3, 0, 1, np.e, np.nan, np.inf]})
        result = add_log_radiance(frame, rf.Y_COLUMN)
        np.testing.assert_allclose(result[rf.Y_COLUMN], [1, np.e])
        np.testing.assert_allclose(result.log_radiance, [0, 1])
        with self.assertRaises(ValueError):
            add_log_radiance(frame.iloc[:2], rf.Y_COLUMN)

    def test_local_time_coverage_and_validation(self):
        frame = pd.DataFrame({"hour": ["2020-12-01 23:00", "2020-07-01 00:00"],
                              "longitude": [0, 270], "latitude": [45, -45]})
        np.testing.assert_allclose(add_local_time(frame.copy()).local_time, [0, 2])
        frame["local_time"] = [7.5, 18.25]
        np.testing.assert_allclose(add_local_time(frame.copy()).local_time, [7.5, 18.25])
        for invalid in (np.nan, np.inf, -1, 24):
            with self.assertRaises(ValueError):
                add_local_time(frame.assign(local_time=invalid))

    def test_training_and_unobserved_grid_use_same_local_time(self):
        source = pd.DataFrame({"hour": pd.date_range("2020-12-01", periods=30, freq="h"),
                               "latitude": 45., "longitude": 270.,
                               rf.Y_COLUMN: [0., -1.] + [2.] * 27 + [np.nan],
                               rf.COUNT_COLUMN: [1] * 29 + [0]})
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "merged.parquet"
            for has_local_time in (False, True):
                if has_local_time:
                    source["local_time"] = 7.5
                source.to_parquet(path, index=False)
                for module in (rf, lasso):
                    frame, features = module.load_model_data(path)
                    self.assertEqual(len(frame), 27)
                    self.assertIn("local_time", features)
                    self.assertNotIn("log_radiance", features)
                    self.assertNotIn(rf.Y_COLUMN, features)
                    columns = kriging.prediction_columns_for_features(path, features)
                    grid = kriging.prepare_grid_prediction_frame(pd.read_parquet(path, columns=columns), features)
                    self.assertEqual(len(grid), len(source))
                    self.assertTrue(grid.local_time.notna().all())
                    np.testing.assert_allclose(frame.local_time, grid.loc[frame.index, "local_time"])

    def test_benchmarks_fit_logs_and_score_exponentiated_predictions(self):
        frame = pd.DataFrame({"hour": pd.date_range("2020-12-01", periods=6, freq="h"),
                               "latitude": np.arange(6.), "longitude": np.arange(6.),
                               "local_time": np.arange(6.), "space_fold": 0, "time_fold": 0,
                               rf.Y_COLUMN: [1., 2., 4., 8., 3., 5.]})
        frame = add_log_radiance(frame, rf.Y_COLUMN)
        for module, factory, diagnostic in ((rf, "make_rf_model", "fold_importance_frame"),
                                             (lasso, "make_lasso_model", "fold_coefficient_frame"),
                                             (kriging, None, None)):
            fitted_targets = []
            model = SimpleNamespace(
                fit=lambda x, y: fitted_targets.append(y.to_numpy().copy()),
                predict=lambda x: np.full(len(x), np.log(2.)),
                named_steps={"lasso": SimpleNamespace(alpha_=0.1)},
            )
            owner = rf if module is kriging else module
            with ExitStack() as stack:
                for name, value in (("MIN_TRAIN_ROWS", 1), ("MIN_TEST_ROWS", 1)):
                    stack.enter_context(patch.object(owner, name, value))
                stack.enter_context(patch.object(owner, "space_time_splits", return_value=[(0, 0, np.arange(4), np.arange(4, 6))]))
                stack.enter_context(patch.object(owner, factory or "make_rf_model", return_value=model))
                stack.enter_context(patch.object(owner, diagnostic or "fold_importance_frame", return_value=pd.DataFrame({"feature": ["local_time"]})))
                if module is kriging:
                    stack.enter_context(patch.object(kriging, "require_pykrige"))
                    stack.enter_context(patch.object(kriging, "predict_rf_log_moments", return_value=(np.full(2, np.log(2.)), np.full(2, 0.2))))
                    stack.enter_context(patch.object(kriging, "predict_residuals", return_value=(np.full(2, np.log(3.)), np.full(2, 0.5), None)))
                    residual_fit = stack.enter_context(patch.object(kriging, "fit_residual_kriging", wraps=kriging.fit_residual_kriging))
                predictions, _, metrics = module.run_space_time_benchmark(frame, ["local_time"])
                np.testing.assert_allclose(fitted_targets[0], np.log([1, 2, 4, 8]))
                expected = 6. if module is kriging else 2.
                np.testing.assert_allclose(predictions.predicted, expected)
                np.testing.assert_allclose(predictions.observed, [3, 5])
                np.testing.assert_allclose(predictions.baseline_predicted, np.exp(np.log([1, 2, 4, 8]).mean()))
                self.assertAlmostEqual(metrics.rmse.iloc[0], np.sqrt(np.mean((np.array([3, 5]) - expected) ** 2)))
                if module is kriging:
                    np.testing.assert_allclose(residual_fit.call_args.kwargs["residuals"], np.log([1, 2, 4, 8]) - np.log(2))
                    fitted = kriging.fit_final_rf_residual_kriging(frame, ["local_time"])
                    np.testing.assert_allclose(fitted_targets[-1], frame.log_radiance)
                    grid = kriging.predict_grid_frame(fitted, frame.iloc[-2:], ["local_time"])
                    np.testing.assert_allclose(grid.predicted_radiance, 6.)
                    np.testing.assert_allclose(grid.rf_mean_predicted_radiance * grid.kriged_residual_factor, grid.predicted_radiance)
                    np.testing.assert_allclose(grid.kriging_log_variance, 0.5)
                    expected_variance = 36 * np.exp(0.7) * np.expm1(0.7)
                    np.testing.assert_allclose(grid.predicted_radiance_variance, expected_variance)
                    np.testing.assert_allclose(predictions.predicted_radiance_variance, expected_variance)

    def test_rf_moments_match_tree_predictions_after_preprocessing(self):
        features = pd.DataFrame({"local_time": [0., 1., 3., np.nan, 5., 7., 10., 13.],
                                 "era5_slt": [1., 2., 1., 2., np.nan, 1., 2., 1.]})
        model = rf.make_rf_model().set_params(random_forest__n_estimators=7,
                                              random_forest__min_samples_leaf=1,
                                              random_forest__n_jobs=1)
        model.fit(features, np.log(np.arange(1., 9.)))
        mean, variance = kriging.predict_rf_log_moments(model, features)
        transformed = model.named_steps["preprocessor"].transform(features)
        tree_values = np.array([tree.predict(transformed) for tree in model.named_steps["random_forest"].estimators_])
        np.testing.assert_allclose(mean, model.predict(features))
        np.testing.assert_allclose(variance, tree_values.var(axis=0, ddof=1))
        self.assertTrue((variance > 0).any())

    def test_variance_conversion_and_additive_contributions(self):
        mu = np.log([2., 10., 100., 4.])
        rf_var = np.array([0., 0.2, 0., 0.2])
        kriging_var = np.array([0., 0., 0.3, 0.3])
        result = kriging.radiance_variance_components(mu, rf_var, kriging_var)
        total_log = rf_var + kriging_var
        np.testing.assert_allclose(result["predicted_radiance_variance"],
                                   np.exp(2 * mu + total_log) * np.expm1(total_log))
        np.testing.assert_allclose(result["predicted_radiance_variance"],
                                   result["rf_radiance_variance"] + result["kriging_radiance_variance"])
        self.assertEqual(result["rf_radiance_variance"][2], 0)
        self.assertEqual(result["kriging_radiance_variance"][1], 0)
        unknown = kriging.radiance_variance_components(1., 0.2, np.nan)
        self.assertTrue(np.isnan(unknown["predicted_radiance_variance"]))
        roundoff = kriging.radiance_variance_components(1., 0., -1e-14)
        self.assertEqual(roundoff["predicted_radiance_variance"], 0)
        with self.assertRaises(ValueError):
            kriging.radiance_variance_components(1., 0.2, -0.1)


if __name__ == "__main__":
    unittest.main()
