import numpy as np
import pytest
from ml.aggregator.static_weights import StaticWeightOptimizer
from ml.aggregator.adaptive_weights import AdaptiveWeightAggregator


class TestStaticWeightOptimizer:

    def setup_method(self):
        np.random.seed(0)
        self.n = 100
        self.actual = np.random.randn(self.n).astype(np.float32) + 150.0
        # Simulate model forecasts with different error profiles
        self.bilstm = self.actual + np.random.randn(self.n) * 2.0
        self.arima = self.actual + np.random.randn(self.n) * 3.0
        self.optimizer = StaticWeightOptimizer()

    def test_optimize_returns_valid_weights(self):
        alpha, beta = self.optimizer.optimize(
            self.bilstm, self.arima, self.actual
        )
        assert 0.0 <= alpha <= 1.0
        assert 0.0 <= beta <= 1.0
        assert abs(alpha + beta - 1.0) < 1e-6

    def test_combine_before_optimize_raises(self):
        with pytest.raises(RuntimeError):
            self.optimizer.combine(self.bilstm, self.arima)

    def test_combine_output_length(self):
        self.optimizer.optimize(self.bilstm, self.arima, self.actual)
        combined = self.optimizer.combine(self.bilstm, self.arima)
        assert len(combined) == self.n

    def test_combined_rmse_le_worst_model(self):
        from ml.evaluation.metrics import rmse
        self.optimizer.optimize(self.bilstm, self.arima, self.actual)
        combined = self.optimizer.combine(self.bilstm, self.arima)
        rmse_combined = rmse(self.actual, combined)
        rmse_arima = rmse(self.actual, self.arima)
        # Combined should not be worse than pure ARIMA
        assert rmse_combined <= rmse_arima + 0.1


class TestAdaptiveWeightAggregator:

    def setup_method(self):
        np.random.seed(1)
        self.n = 80
        self.actual = np.random.randn(self.n).astype(np.float32) + 200.0
        self.bilstm = self.actual + np.random.randn(self.n) * 1.5
        self.arima = self.actual + np.random.randn(self.n) * 4.0
        self.aggregator = AdaptiveWeightAggregator(window=10)

    def test_output_length(self):
        combined = self.aggregator.combine(
            self.bilstm, self.arima, self.actual
        )
        assert len(combined) == self.n

    def test_values_are_finite(self):
        combined = self.aggregator.combine(
            self.bilstm, self.arima, self.actual
        )
        assert np.all(np.isfinite(combined))

    def test_output_bounded_between_inputs(self):
        combined = self.aggregator.combine(
            self.bilstm, self.arima, self.actual
        )
        low = np.minimum(self.bilstm, self.arima)
        high = np.maximum(self.bilstm, self.arima)
        assert np.all(combined >= low - 1e-5)
        assert np.all(combined <= high + 1e-5)
