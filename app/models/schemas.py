from pydantic import BaseModel, Field
from typing import List, Optional


class PredictRequest(BaseModel):
    ticker: str = Field(
        ...,
        min_length=1,
        max_length=20,
        example="AAPL",
        description="Stock ticker symbol"
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
    ticker: str
    metrics: MetricsSchema
    alpha: float = Field(description="Bi-LSTM weight")
    beta: float = Field(description="ARIMA weight")
    actual: List[float]
    predicted: List[float]
    bilstm_only: List[float]
    arima_only: List[float]
    n_test_days: int
    arima_order: Optional[List[int]] = None
