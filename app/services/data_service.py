import numpy as np
import yfinance as yf
from app.core.config import config
from app.core.logger import logger


class DataService:
    """Fetches and validates historical OHLCV data."""

    @staticmethod
    def fetch(
        ticker: str,
        period: str = config.DEFAULT_PERIOD
    ) -> np.ndarray:
        """
        Download daily closing prices from Yahoo Finance.

        Parameters
        ----------
        ticker : str   e.g. "AAPL", "MSFT", "RELIANCE.NS"
        period : str   yfinance period string e.g. "5y", "3y"

        Returns
        -------
        np.ndarray     shape (T,) float32 closing prices

        Raises
        ------
        ValueError     if ticker is invalid or data is insufficient
        """
        logger.info(f"Fetching {period} of data for '{ticker}'...")

        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            timeout=30
        )

        if df is None or df.empty:
            raise ValueError(
                f"No data returned for ticker '{ticker}'. "
                "Verify the symbol is correct."
            )

        # Handle MultiIndex columns from yfinance
        if isinstance(df.columns, tuple) or hasattr(df.columns, 'levels'):
            if 'Close' in df.columns.get_level_values(0):
                prices_series = df['Close'].dropna()
                if hasattr(prices_series, 'iloc'):
                    prices = prices_series.values.flatten()
                else:
                    prices = prices_series.flatten()
            else:
                prices = df.iloc[:, 3].dropna().values.flatten()
        else:
            prices = df['Close'].dropna().values.flatten()

        prices = prices.astype(np.float32)

        if len(prices) < config.MIN_TRADING_DAYS:
            raise ValueError(
                f"Insufficient data for '{ticker}': "
                f"{len(prices)} days retrieved, "
                f"minimum required is {config.MIN_TRADING_DAYS}. "
                "Try a longer period e.g. '5y'."
            )

        logger.info(
            f"Successfully fetched {len(prices)} trading days "
            f"for '{ticker}'. "
            f"Price range: {prices.min():.2f} – {prices.max():.2f}"
        )
        return prices
