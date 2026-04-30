import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from ml.pipeline import ForecastPipeline


class TestForecastPipeline:

    def test_pipeline_instantiates(self):
        p = ForecastPipeline("AAPL", use_adaptive=False)
        assert p.ticker == "AAPL"
        assert p.use_adaptive is False

    def test_pipeline_ticker_uppercased(self):
        p = ForecastPipeline("msft")
        assert p.ticker == "MSFT"

    @patch("ml.pipeline.DataService.fetch")
    def test_pipeline_raises_on_insufficient_data(self, mock_fetch):
        # Simulate too little data
        mock_fetch.return_value = np.random.randn(100).astype(np.float32) + 100
        p = ForecastPipeline("TEST")
        with pytest.raises(Exception):
            p.run()

    def test_pipeline_result_keys(self):
        """
        Integration smoke test using synthetic data.
        Skipped in CI unless RUN_INTEGRATION=1 env var is set.
        """
        import os
        if not os.getenv("RUN_INTEGRATION"):
            pytest.skip("Integration test skipped (set RUN_INTEGRATION=1)")

        p = ForecastPipeline("AAPL", period="3y")
        result = p.run()
        required_keys = [
            "ticker", "metrics", "alpha", "beta",
            "actual", "predicted", "bilstm_only",
            "arima_only", "n_test_days"
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

        assert result["metrics"]["RMSE"] > 0
        assert 0.0 <= result["alpha"] <= 1.0
        assert len(result["actual"]) == len(result["predicted"])
