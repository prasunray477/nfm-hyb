from pydantic import BaseModel, Field
from typing import List, Optional


class PredictRequest(BaseModel):
    ticker: str = Field(
        ...,
        min_length=1,
        max_length=20,
        example="AAPL",
        description="Stock Symbol (e.g. AAPL, MSFT, RELIANCE.NS)"
    )
    use_adaptive: bool = Field(
        False,
        description="Use adaptive inverse-error weights instead of static"
    )
    period: str = Field(
        "5y",
        description="Data period (yfinance format: 1y, 3y, 5y)"
    )


class MetricsSchema(BaseModel):
    RMSE: float
    MAE: float
    MAPE: float
    Theils_U: float
    Directional_Accuracy: float


class PredictResponse(BaseModel):
    # ── Original fields (unchanged) ────────────────────────────────
    ticker: str
    metrics: MetricsSchema
    alpha: float = Field(description="Deep Learning Model Weight")
    beta: float = Field(description="Statistical Model Weight")
    actual: List[float]
    predicted: List[float]
    bilstm_only: List[float]
    arima_only: List[float]
    n_test_days: int
    arima_order: Optional[List[int]] = None
    historical_dates: List[str]
    historical_prices: List[float]
    daily_movements: List[float]
    movement_dates: List[str]
    train_end_idx: int
    triples: List[List[float]]
    triple_dates: List[str]

    # ── New fields: 30-Day Future Projection ───────────────────────
    future_dates: List[str]
    bilstm_future: List[float]
    arima_future: List[float]
    combined_future: List[float]
    upper_band: List[float]
    lower_band: List[float]
    last_known_price: float
    trend_label: str           # "Bullish" | "Bearish" | "Neutral"
    trend_strength: float      # 0–100
    recommendation: str        # "BUY" | "HOLD" | "SELL"
    recommendation_reason: str
    projected_change_pct: float
    confidence_margin: float
