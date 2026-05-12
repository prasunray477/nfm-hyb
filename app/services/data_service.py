import numpy as np
import os
import yfinance as yf
import yfinance.cache as yf_cache
import yfinance.shared as yf_shared
from app.core.config import config
from app.core.logger import logger


class DataFetchError(RuntimeError):
    """Raised when the market data provider cannot be reached."""


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
        normalized_ticker = ticker.upper().strip()
        DataService._configure_yfinance_cache()
        DataService._configure_proxy()
        yf_shared._ERRORS.pop(normalized_ticker, None)

        df = yf.download(
            normalized_ticker,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            timeout=30,
        )

        if df is None or df.empty:
            download_error = yf_shared._ERRORS.get(normalized_ticker)
            if download_error and DataService._is_provider_failure(download_error):
                raise DataFetchError(
                    f"Could not download market data for '{normalized_ticker}' "
                    "from Yahoo Finance. The backend cannot reach "
                    "Yahoo Finance right now. Check your internet connection, "
                    "firewall/VPN/proxy settings, or set YFINANCE_PROXY in "
                    "your .env file, then retry. "
                    f"Provider error: {download_error}"
                )

            raise ValueError(
                f"No data returned for ticker '{normalized_ticker}'. "
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

    @staticmethod
    def _is_provider_failure(error: str) -> bool:
        network_markers = (
            "ConnectionError",
            "HTTPSConnectionPool",
            "Failed to establish a new connection",
            "Max retries exceeded",
            "ReadTimeout",
            "Timeout",
            "WinError",
            "ProxyError",
            "SSLError",
            "OperationalError",
            "unable to open database file",
            "JSONDecodeError",
            "Expecting value",
        )
        return any(marker in error for marker in network_markers)

    @staticmethod
    def _configure_yfinance_cache() -> None:
        cache_dir = f"{config.DATA_DIR}/yfinance_cache"
        try:
            os.makedirs(cache_dir, exist_ok=True)
            yf_cache.set_cache_location(cache_dir)
        except Exception as e:
            logger.warning(f"Could not configure yfinance cache: {e}")

    @staticmethod
    def _configure_proxy() -> None:
        proxy = os.getenv("YFINANCE_PROXY")
        if not proxy:
            return

        os.environ.setdefault("HTTP_PROXY", proxy)
        os.environ.setdefault("HTTPS_PROXY", proxy)
