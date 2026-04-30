import numpy as np
import pandas as pd
from app.core.logger import logger


def clean_price_series(prices: np.ndarray) -> np.ndarray:
    """
    Remove NaN, Inf values and forward-fill gaps
    in a closing price array.

    Parameters
    ----------
    prices : np.ndarray  raw closing price array

    Returns
    -------
    np.ndarray  cleaned float32 price array
    """
    series = pd.Series(prices.astype(np.float64))

    # Replace inf with NaN then forward-fill
    series.replace([np.inf, -np.inf], np.nan, inplace=True)
    series.ffill(inplace=True)
    series.bfill(inplace=True)  # Handle leading NaNs

    n_dropped = int(prices.shape[0] - series.notna().sum())
    if n_dropped > 0:
        logger.warning(
            f"preprocessing: filled {n_dropped} missing/inf values"
        )

    result = series.dropna().values.astype(np.float32)
    logger.info(
        f"Preprocessing complete. "
        f"Input: {len(prices)}, Output: {len(result)} samples."
    )
    return result


def remove_outliers_iqr(
    series: np.ndarray,
    factor: float = 5.0
) -> np.ndarray:
    """
    Clip extreme outliers beyond factor × IQR from quartiles.
    Used on fluctuation series before ARIMA fitting.

    Parameters
    ----------
    series : np.ndarray   fluctuation series U_t
    factor : float        IQR multiplier (default 5.0 — conservative)

    Returns
    -------
    np.ndarray  clipped series same length as input
    """
    q1 = float(np.percentile(series, 25))
    q3 = float(np.percentile(series, 75))
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    clipped = np.clip(series, lower, upper)

    n_clipped = int(np.sum((series < lower) | (series > upper)))
    if n_clipped > 0:
        logger.info(
            f"IQR clipping: {n_clipped} values clipped "
            f"to [{lower:.4f}, {upper:.4f}]"
        )
    return clipped.astype(np.float32)
