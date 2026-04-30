import numpy as np
from statsmodels.tsa.stattools import adfuller
from app.core.config import config
from app.core.logger import logger


def enforce_stationarity(
    prices: np.ndarray,
    significance: float = config.ADF_SIGNIFICANCE,
    max_diffs: int = 2
) -> np.ndarray:
    """
    Apply differencing until the ADF test confirms stationarity.

    Parameters
    ----------
    prices      : np.ndarray  raw closing price series
    significance: float       ADF p-value threshold (default 0.05)
    max_diffs   : int         maximum differencing iterations

    Returns
    -------
    np.ndarray  stationary fluctuation series U_t
    """
    series = prices.copy().astype(np.float64)
    n_diffs = 0

    for iteration in range(max_diffs + 1):
        adf_result = adfuller(series, autolag='AIC')
        p_value = adf_result[1]
        adf_stat = adf_result[0]

        logger.info(
            f"ADF test iteration {iteration}: "
            f"statistic={adf_stat:.4f}, p-value={p_value:.4f}"
        )

        if p_value <= significance:
            logger.info(
                f"Stationarity confirmed after {n_diffs} "
                f"differencing step(s). p={p_value:.4f}"
            )
            return series.astype(np.float32)

        if iteration < max_diffs:
            series = np.diff(series)
            n_diffs += 1
            logger.info(
                f"Series non-stationary (p={p_value:.4f}). "
                f"Applying differencing #{n_diffs}..."
            )

    logger.warning(
        f"Series still non-stationary after {max_diffs} "
        "differencing(s). Proceeding with caution."
    )
    return series.astype(np.float32)
