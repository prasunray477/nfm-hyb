import numpy as np
from app.core.config import config
from app.core.logger import logger


class StaticWeightOptimizer:
    """
    Grid search over α ∈ [0,1] to minimize validation RMSE.
    Finds the optimal static blend of Bi-LSTM and ARIMA outputs.
    """

    def __init__(self):
        self.alpha_star: float = None
        self.beta_star: float = None
        self._val_rmse_curve: list = None

    def optimize(
        self,
        bilstm_val: np.ndarray,
        arima_val: np.ndarray,
        actual_val: np.ndarray
    ) -> tuple:
        """
        Find optimal α* on validation set.

        Parameters
        ----------
        bilstm_val : Bi-LSTM price forecasts on validation set
        arima_val  : ARIMA price forecasts on validation set
        actual_val : actual prices on validation set

        Returns
        -------
        (alpha_star, beta_star)
        """
        step = config.ALPHA_GRID_STEP
        alphas = np.arange(0.0, 1.0 + step / 2, step)
        rmse_values = []
        best_rmse = np.inf
        best_alpha = 0.5

        for alpha in alphas:
            combined = alpha * bilstm_val + (1 - alpha) * arima_val
            rmse = float(
                np.sqrt(np.mean((combined - actual_val) ** 2))
            )
            rmse_values.append(rmse)
            if rmse < best_rmse:
                best_rmse = rmse
                best_alpha = alpha

        self.alpha_star = float(best_alpha)
        self.beta_star = float(1.0 - best_alpha)
        self._val_rmse_curve = rmse_values

        logger.info(
            f"Weight optimization complete. "
            f"α* (BiLSTM) = {self.alpha_star:.2f}, "
            f"β* (ARIMA)  = {self.beta_star:.2f}, "
            f"Best val RMSE = {best_rmse:.4f}"
        )
        return self.alpha_star, self.beta_star

    def combine(
        self,
        bilstm_forecasts: np.ndarray,
        arima_forecasts: np.ndarray
    ) -> np.ndarray:
        if self.alpha_star is None:
            raise RuntimeError("Call optimize() before combine().")
        return (
            self.alpha_star * bilstm_forecasts
            + self.beta_star * arima_forecasts
        ).astype(np.float32)
