import numpy as np
import pytest
from ml.arima.model import ARIMAForecaster


class TestARIMAForecaster:

    def setup_method(self):
        np.random.seed(42)
        # Simulate stationary fluctuation series
        self.flucts = np.random.randn(300).astype(np.float32)
        self.forecaster = ARIMAForecaster()

    def test_fit_sets_order(self):
        self.forecaster.fit(self.flucts)
        assert self.forecaster.order is not None
        assert len(self.forecaster.order) == 3

    def test_order_values_are_nonnegative(self):
        self.forecaster.fit(self.flucts)
        p, d, q = self.forecaster.order
        assert p >= 0 and d >= 0 and q >= 0

    def test_forecast_one_step_returns_scalar(self):
        self.forecaster.fit(self.flucts)
        pred = self.forecaster.forecast_one_step(self.flucts[:200])
        assert isinstance(pred, float)
        assert np.isfinite(pred)

    def test_forecast_series_length(self):
        self.forecaster.fit(self.flucts)
        preds = self.forecaster.forecast_series(
            initial_train=self.flucts[:200],
            n_steps=30
        )
        assert len(preds) == 30

    def test_forecast_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            self.forecaster.forecast_one_step(self.flucts)

    def test_predictions_are_finite(self):
        self.forecaster.fit(self.flucts)
        preds = self.forecaster.forecast_series(self.flucts[:200], 10)
        assert np.all(np.isfinite(preds))
