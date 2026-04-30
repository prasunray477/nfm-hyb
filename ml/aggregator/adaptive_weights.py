import numpy as np
from app.core.config import config
from app.core.logger import logger


class AdaptiveWeightAggregator:
    """
    Dynamically adjusts model weights based on recent prediction
    accuracy. Models that performed better recently receive more weight.
    """

    def __init__(self, window: int = config.ADAPTIVE_WINDOW):
        self.window = window

    def combine(
        self,
        bilstm_forecasts: np.ndarray,
        arima_forecasts: np.ndarray,
        actual_prices: np.ndarray
    ) -> np.ndarray:
        """
        Combine forecasts with rolling inverse-error weighting.

        Parameters
        ----------
        bilstm_forecasts : shape (N,)  Bi-LSTM price predictions
        arima_forecasts  : shape (N,)  ARIMA price predictions
        actual_prices    : shape (N,)  actual prices (for error tracking)

        Returns
        -------
        np.ndarray  shape (N,)  combined predictions
        """
        n = len(bilstm_forecasts)
        combined = np.zeros(n, dtype=np.float32)

        for t in range(n):
            if t < self.window:
                # Insufficient history → equal weights
                alpha_t = 0.5
            else:
                start = t - self.window
                e_bilstm = float(np.mean(
                    np.abs(bilstm_forecasts[start:t]
                           - actual_prices[start:t])
                ))
                e_arima = float(np.mean(
                    np.abs(arima_forecasts[start:t]
                           - actual_prices[start:t])
                ))
                # Guard against zero errors
                e_bilstm = max(e_bilstm, 1e-8)
                e_arima = max(e_arima, 1e-8)

                alpha_t = (1.0 / e_bilstm) / (
                    1.0 / e_bilstm + 1.0 / e_arima
                )

            combined[t] = (
                alpha_t * bilstm_forecasts[t]
                + (1.0 - alpha_t) * arima_forecasts[t]
            )

        logger.info(
            f"Adaptive aggregation complete. "
            f"Mean α across test: "
            f"{sum([(1.0/max(abs(b-a),1e-8))/(1.0/max(abs(b-a),1e-8)+1.0/max(abs(r-a),1e-8)) for b,r,a in zip(bilstm_forecasts[self.window:], arima_forecasts[self.window:], actual_prices[self.window:])])/max(n-self.window,1):.3f}"
        )
        return combined
