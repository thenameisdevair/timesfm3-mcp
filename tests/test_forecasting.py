"""Unit tests for TimesFM-3 MCP forecast helpers.

Uses a fake forecaster so we can verify shapes and validation without
downloading google/timesfm-3.0-pytorch.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from forecasting import run_forecast, serialize_quantiles


class FakeForecaster:
    def __init__(self, forecast, quantiles):
        self._forecast = forecast
        self._quantiles = quantiles
        self.calls: list[dict] = []

    def predict_batch(self, contexts, horizon, **kwargs):
        self.calls.append(
            {
                "contexts": contexts,
                "horizon": horizon,
                **kwargs,
            }
        )
        yield SimpleNamespace(forecast=self._forecast, quantiles=self._quantiles)


def _univariate_out(horizon: int = 5, n_q: int = 9):
    forecast = np.linspace(10, 14, horizon, dtype=np.float32)
    quantiles = np.stack(
        [forecast - 1 + 0.25 * i for i in range(n_q)],
        axis=1,
    )
    return forecast, quantiles


def _multivariate_out(n_series: int, horizon: int = 4, n_q: int = 9):
    forecast = np.arange(n_series * horizon, dtype=np.float32).reshape(n_series, horizon)
    quantiles = np.stack([forecast + 0.1 * i for i in range(n_q)], axis=-1)
    return forecast, quantiles


class SerializeQuantilesTests(unittest.TestCase):
    def test_univariate_nine_heads(self):
        q = np.arange(5 * 9, dtype=np.float32).reshape(5, 9)
        packed = serialize_quantiles(q)
        self.assertEqual(list(packed.keys()), [f"q{i}0" for i in range(1, 10)])
        self.assertEqual(packed["q10"], q[:, 0].tolist())
        self.assertEqual(packed["q90"], q[:, 8].tolist())

    def test_none(self):
        self.assertIsNone(serialize_quantiles(None))


class ValidationTests(unittest.TestCase):
    def test_empty_history(self):
        fake = FakeForecaster(*_univariate_out())
        out = run_forecast(fake, history=[], horizon=3)
        self.assertEqual(out["status"], "error")
        self.assertIn("history", out["error"])

    def test_empty_series(self):
        fake = FakeForecaster(*_univariate_out())
        out = run_forecast(fake, series=[], horizon=3)
        self.assertEqual(out["status"], "error")

    def test_uneven_series_lengths(self):
        fake = FakeForecaster(*_multivariate_out(2))
        out = run_forecast(fake, series=[[1, 2, 3], [1, 2]], horizon=2)
        self.assertEqual(out["status"], "error")
        self.assertIn("same length", out["error"])

    def test_horizon_must_be_positive(self):
        fake = FakeForecaster(*_univariate_out())
        out = run_forecast(fake, history=[1.0, 2.0], horizon=0)
        self.assertEqual(out["status"], "error")
        self.assertIn("horizon", out["error"])

    def test_history_and_series_together_rejected(self):
        fake = FakeForecaster(*_univariate_out())
        out = run_forecast(
            fake,
            history=[1.0, 2.0],
            series=[[1.0, 2.0]],
            horizon=2,
        )
        self.assertEqual(out["status"], "error")
        self.assertIn("not both", out["error"])

    def test_series_ids_length(self):
        fake = FakeForecaster(*_multivariate_out(2))
        out = run_forecast(
            fake,
            series=[[1, 2, 3], [4, 5, 6]],
            series_ids=["only_one"],
            horizon=4,
        )
        self.assertEqual(out["status"], "error")
        self.assertIn("series_ids", out["error"])

    def test_past_covariates_must_match_context(self):
        fake = FakeForecaster(*_univariate_out())
        out = run_forecast(
            fake,
            series=[[1, 2, 3, 4]],
            past_covariates=[[1, 2]],
            horizon=2,
        )
        self.assertEqual(out["status"], "error")
        self.assertIn("past_covariates", out["error"])

    def test_future_covariates_must_be_context_plus_horizon(self):
        fake = FakeForecaster(*_univariate_out())
        out = run_forecast(
            fake,
            series=[[1, 2, 3, 4]],
            future_covariates=[[0, 1, 0, 1]],
            horizon=3,
        )
        self.assertEqual(out["status"], "error")
        self.assertIn("expected 7", out["error"])


class UnivariatePathTests(unittest.TestCase):
    def test_history_uses_1d_context(self):
        forecast, quantiles = _univariate_out(horizon=5)
        fake = FakeForecaster(forecast, quantiles)
        history = [10.5, 12.1, 14.8, 15.2, 18.0]
        out = run_forecast(fake, history=history, horizon=5)
        self.assertEqual(out["status"], "success")
        self.assertEqual(out["mode"], "univariate")
        self.assertEqual(out["n_series"], 1)
        self.assertEqual(out["context_length"], 5)
        self.assertEqual(out["forecast"], forecast.astype(float).tolist())
        self.assertEqual(out["quantiles"]["q50"], quantiles[:, 4].astype(float).tolist())
        self.assertEqual(out["series"][0]["id"], "series_0")
        self.assertNotIn("timestamps", out)
        self.assertNotIn("timestamps", out["series"][0])
        self.assertNotIn("freq", out)
        self.assertNotIn("history_end", out)

        call = fake.calls[0]
        self.assertEqual(len(call["contexts"]), 1)
        self.assertEqual(call["contexts"][0].ndim, 1)
        self.assertEqual(call["contexts"][0].shape, (5,))
        self.assertIsNone(call["past_only_covariates"])
        self.assertIsNone(call["past_future_covariates"])
        self.assertTrue(call["return_quantiles"])
        self.assertFalse(call["use_symmetric_averaging"])

    def test_single_row_series_matches_history(self):
        forecast, quantiles = _univariate_out(horizon=3)
        fake = FakeForecaster(forecast, quantiles)
        out = run_forecast(fake, series=[[1.0, 2.0, 3.0, 4.0]], horizon=3)
        self.assertEqual(out["mode"], "univariate")
        self.assertEqual(fake.calls[0]["contexts"][0].ndim, 1)


class MultivariatePathTests(unittest.TestCase):
    def test_two_targets_use_2d_context(self):
        forecast, quantiles = _multivariate_out(n_series=2, horizon=4)
        fake = FakeForecaster(forecast, quantiles)
        sku_a = [10, 11, 12, 13, 14, 15]
        sku_b = [20, 21, 22, 23, 24, 25]
        out = run_forecast(
            fake,
            series=[sku_a, sku_b],
            series_ids=["sku_a", "sku_b"],
            horizon=4,
        )
        self.assertEqual(out["status"], "success")
        self.assertEqual(out["mode"], "multivariate")
        self.assertEqual(out["n_series"], 2)
        self.assertNotIn("forecast", out)
        self.assertNotIn("timestamps", out)
        self.assertNotIn("timestamps", out["series"][0])
        self.assertEqual(out["series"][0]["id"], "sku_a")
        self.assertEqual(out["series"][1]["id"], "sku_b")
        self.assertEqual(len(out["series"][0]["forecast"]), 4)
        self.assertEqual(len(out["series"][0]["quantiles"]["q10"]), 4)

        ctx = fake.calls[0]["contexts"][0]
        self.assertEqual(ctx.shape, (2, 6))
        np.testing.assert_array_equal(ctx[0], np.asarray(sku_a, dtype=np.float32))

    def test_univariate_with_future_covariate_uses_2d_context(self):
        forecast = np.linspace(1, 3, 3, dtype=np.float32)
        quantiles = np.stack([forecast] * 9, axis=1)
        fake = FakeForecaster(forecast, quantiles)
        series = [[1, 2, 3, 4]]
        future = [[0, 0, 0, 0, 1, 1, 1]]  # T=4, H=3
        out = run_forecast(fake, series=series, future_covariates=[future[0]], horizon=3)
        self.assertEqual(out["status"], "success")
        self.assertEqual(out["mode"], "univariate")
        ctx = fake.calls[0]["contexts"][0]
        self.assertEqual(ctx.shape, (1, 4))
        pf = fake.calls[0]["past_future_covariates"][0]
        self.assertEqual(pf.shape, (1, 7))

    def test_past_and_future_covariates_forwarded(self):
        forecast, quantiles = _multivariate_out(n_series=2, horizon=2)
        fake = FakeForecaster(forecast, quantiles)
        series = [[1, 2, 3, 4], [5, 6, 7, 8]]
        past = [[9, 9, 9, 9]]
        future = [[0, 1, 0, 1, 1, 1]]
        out = run_forecast(
            fake,
            series=series,
            past_covariates=past,
            future_covariates=future,
            horizon=2,
        )
        self.assertEqual(out["status"], "success")
        call = fake.calls[0]
        self.assertEqual(call["contexts"][0].shape, (2, 4))
        self.assertEqual(call["past_only_covariates"][0].shape, (1, 4))
        self.assertEqual(call["past_future_covariates"][0].shape, (1, 6))

    def test_model_series_count_mismatch(self):
        forecast, quantiles = _multivariate_out(n_series=1, horizon=2)
        fake = FakeForecaster(forecast, quantiles)
        out = run_forecast(fake, series=[[1, 2], [3, 4]], horizon=2)
        self.assertEqual(out["status"], "error")
        self.assertIn("expected 2", out["error"])


class CalendarTests(unittest.TestCase):
    def test_daily_start_freq_labels_forecast(self):
        forecast, quantiles = _univariate_out(horizon=3)
        fake = FakeForecaster(forecast, quantiles)
        history = [10, 11, 13, 12, 14, 16, 15, 17]  # T=8
        out = run_forecast(
            fake,
            history=history,
            horizon=3,
            start="2026-08-25",
            freq="D",
        )
        self.assertEqual(out["status"], "success")
        self.assertEqual(out["freq"], "D")
        self.assertEqual(out["history_end"], "2026-09-01")
        self.assertEqual(out["timestamps"], ["2026-09-02", "2026-09-03", "2026-09-04"])
        self.assertEqual(out["series"][0]["timestamps"], out["timestamps"])
        self.assertEqual(out["forecast"], forecast.astype(float).tolist())
        self.assertEqual(len(fake.calls), 1)
        self.assertNotIn("start", fake.calls[0])

    def test_start_without_freq_errors_before_model(self):
        fake = FakeForecaster(*_univariate_out())
        out = run_forecast(fake, history=[1.0, 2.0, 3.0], horizon=2, start="2026-08-25")
        self.assertEqual(out["status"], "error")
        self.assertIn("together", out["error"])
        self.assertEqual(fake.calls, [])

    def test_freq_without_start_errors_before_model(self):
        fake = FakeForecaster(*_univariate_out())
        out = run_forecast(fake, history=[1.0, 2.0, 3.0], horizon=2, freq="D")
        self.assertEqual(out["status"], "error")
        self.assertEqual(fake.calls, [])

    def test_unknown_freq(self):
        fake = FakeForecaster(*_univariate_out())
        out = run_forecast(
            fake, history=[1.0, 2.0, 3.0], horizon=2, start="2026-08-25", freq="Q"
        )
        self.assertEqual(out["status"], "error")
        self.assertIn("freq", out["error"])
        self.assertEqual(fake.calls, [])

    def test_hourly(self):
        forecast, quantiles = _univariate_out(horizon=2)
        fake = FakeForecaster(forecast, quantiles)
        out = run_forecast(
            fake,
            history=[1.0, 2.0, 3.0],
            horizon=2,
            start="2026-09-01T10:00:00",
            freq="H",
        )
        self.assertEqual(out["status"], "success")
        self.assertEqual(out["freq"], "H")
        self.assertEqual(out["history_end"], "2026-09-01T12:00:00")
        self.assertEqual(
            out["timestamps"],
            ["2026-09-01T13:00:00", "2026-09-01T14:00:00"],
        )

    def test_weekly(self):
        forecast, quantiles = _univariate_out(horizon=2)
        fake = FakeForecaster(forecast, quantiles)
        out = run_forecast(
            fake,
            history=[1.0, 2.0, 3.0, 4.0],
            horizon=2,
            start="2026-08-24",
            freq="W",
        )
        self.assertEqual(out["history_end"], "2026-09-14")
        self.assertEqual(out["timestamps"], ["2026-09-21", "2026-09-28"])

    def test_monthly_clamps_end_of_month(self):
        forecast, quantiles = _univariate_out(horizon=2)
        fake = FakeForecaster(forecast, quantiles)
        out = run_forecast(
            fake,
            history=[1.0, 2.0],
            horizon=2,
            start="2026-01-31",
            freq="M",
        )
        self.assertEqual(out["history_end"], "2026-02-28")
        self.assertEqual(out["timestamps"], ["2026-03-31", "2026-04-30"])

    def test_regular_timestamp_list(self):
        forecast, quantiles = _univariate_out(horizon=2)
        fake = FakeForecaster(forecast, quantiles)
        out = run_forecast(
            fake,
            history=[1.0, 2.0, 3.0],
            horizon=2,
            timestamps=["2026-08-01", "2026-08-02", "2026-08-03"],
        )
        self.assertEqual(out["status"], "success")
        self.assertEqual(out["freq"], "D")
        self.assertEqual(out["history_end"], "2026-08-03")
        self.assertEqual(out["timestamps"], ["2026-08-04", "2026-08-05"])

    def test_gapped_timestamps_rejected(self):
        fake = FakeForecaster(*_univariate_out())
        out = run_forecast(
            fake,
            history=[1.0, 2.0, 3.0],
            horizon=2,
            timestamps=["2026-08-01", "2026-08-02", "2026-08-04"],
        )
        self.assertEqual(out["status"], "error")
        self.assertIn("not strictly regular", out["error"])
        self.assertEqual(fake.calls, [])

    def test_timestamps_wrong_length(self):
        fake = FakeForecaster(*_univariate_out())
        out = run_forecast(
            fake,
            history=[1.0, 2.0, 3.0],
            horizon=2,
            timestamps=["2026-08-01", "2026-08-02"],
        )
        self.assertEqual(out["status"], "error")
        self.assertIn("expected 3", out["error"])
        self.assertEqual(fake.calls, [])

    def test_cannot_mix_timestamps_with_start(self):
        fake = FakeForecaster(*_univariate_out())
        out = run_forecast(
            fake,
            history=[1.0, 2.0, 3.0],
            horizon=2,
            start="2026-08-01",
            freq="D",
            timestamps=["2026-08-01", "2026-08-02", "2026-08-03"],
        )
        self.assertEqual(out["status"], "error")
        self.assertIn("not both", out["error"])
        self.assertEqual(fake.calls, [])

    def test_multivariate_shares_one_calendar(self):
        forecast, quantiles = _multivariate_out(n_series=2, horizon=3)
        fake = FakeForecaster(forecast, quantiles)
        out = run_forecast(
            fake,
            series=[[1, 2, 3, 4], [5, 6, 7, 8]],
            series_ids=["sku_a", "sku_b"],
            horizon=3,
            start="2026-08-01",
            freq="D",
        )
        self.assertEqual(out["status"], "success")
        self.assertEqual(out["mode"], "multivariate")
        self.assertNotIn("timestamps", out)
        self.assertEqual(
            out["series"][0]["timestamps"],
            ["2026-08-05", "2026-08-06", "2026-08-07"],
        )
        self.assertEqual(
            out["series"][1]["timestamps"],
            out["series"][0]["timestamps"],
        )


if __name__ == "__main__":
    unittest.main()
