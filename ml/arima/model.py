import numpy as np
import os
import joblib
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from pmdarima import auto_arima
from app.core.config import config
from app.core.logger import logger


class ARIMAForecaster:
    """
    Fits ARIMA on the stationary fluctuation series and generates
    one-step-ahead forecasts using an expanding window strategy.
    """

    def __init__(self):
        self.order: tuple = None
        self._train_series: np.ndarray = None
        self._fitted: bool = False

    # ── Public API ────────────────────────────────────────────────

    def fit(self, train_series: np.ndarray) -> "ARIMAForecaster":
        """
        Select optimal ARIMA order via AIC and fit on training data.

        Parameters
        ----------
        train_series : np.ndarray  stationary fluctuation series
        """
        self._train_series = train_series.copy().astype(np.float64)

        logger.info(
            "Running auto_arima for optimal (p,d,q) selection..."
        )

        auto_model = auto_arima(
            self._train_series,
            start_p=0, max_p=config.MAX_P,
            start_q=0, max_q=config.MAX_Q,
            d=0,                            # Already stationary
            seasonal=False,
            information_criterion='aic',
            stepwise=True,                  # Faster than full grid
            suppress_warnings=True,
            error_action='ignore',
            trace=False
        )

        self.order = auto_model.order
        logger.info(f"Selected ARIMA order: {self.order}")

        # Verify residual quality
        fitted = ARIMA(self._train_series, order=self.order).fit()
        lb_result = acorr_ljungbox(
            fitted.resid, lags=[10], return_df=True
        )
        lb_p = lb_result['lb_pvalue'].iloc[0]

        if lb_p < 0.05:
            logger.warning(
                f"Ljung-Box p={lb_p:.4f} < 0.05. "
                "Residual autocorrelation present. "
                "Consider increasing max_p or max_q."
            )
        else:
            logger.info(
                f"Residual check passed. "
                f"Ljung-Box p={lb_p:.4f}. AIC={fitted.aic:.2f}"
            )

        self._fitted = True
        return self

    def forecast_one_step(self, history: np.ndarray) -> float:
        """
        Fit ARIMA on all available history and forecast 1 step.
        Uses expanding window — called once per test day.
        """
        if self.order is None:
            raise RuntimeError("Call fit() before forecast_one_step().")

        model = ARIMA(
            history.astype(np.float64),
            order=self.order
        ).fit()
        forecast = model.forecast(steps=1)
        if hasattr(forecast, 'iloc'):
            return float(forecast.iloc[0])
        return float(forecast[0])

    def forecast_series(
        self,
        initial_train: np.ndarray,
        n_steps: int
    ) -> np.ndarray:
        """
        Generate n_steps one-step-ahead forecasts.
        Each step uses all data up to that point (expanding window).
        """
        history = list(initial_train.astype(np.float64))
        forecasts = []

        for step in range(n_steps):
            pred = self.forecast_one_step(np.array(history))
            forecasts.append(pred)
            # Append NaN placeholder; caller replaces with actual
            history.append(float('nan'))

            if (step + 1) % 20 == 0:
                logger.info(
                    f"ARIMA forecasting: {step + 1}/{n_steps} steps"
                )

        return np.array(forecasts, dtype=np.float32)

    def save(self, ticker: str) -> str:
        """Persist order and train series for later reuse."""
        os.makedirs(config.ARIMA_MODEL_DIR, exist_ok=True)
        path = os.path.join(
            config.ARIMA_MODEL_DIR, f"{ticker}_arima.joblib"
        )
        joblib.dump({
            'order': self.order,
            'train_series': self._train_series
        }, path)
        logger.info(f"ARIMA config saved → {path}")
        return path

    def load(self, ticker: str) -> "ARIMAForecaster":
        path = os.path.join(
            config.ARIMA_MODEL_DIR, f"{ticker}_arima.joblib"
        )
        data = joblib.load(path)
        self.order = data['order']
        self._train_series = data['train_series']
        self._fitted = True
        logger.info(
            f"ARIMA loaded from {path}. Order: {self.order}"
        )
        return self
