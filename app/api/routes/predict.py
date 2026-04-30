from fastapi import APIRouter, HTTPException
from app.models.schemas import PredictRequest, PredictResponse
from ml.pipeline import ForecastPipeline
from app.core.logger import logger

router = APIRouter(tags=["Prediction"])


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Run full forecasting pipeline for a stock ticker"
)
def predict(request: PredictRequest):
    """
    Execute the complete Neutrosophic Bi-LSTM + ARIMA
    forecasting pipeline for the given stock ticker.

    - Downloads historical data via yfinance
    - Applies neutrosophic normalization
    - Trains Bi-LSTM and fits ARIMA
    - Optimizes ensemble weights
    - Returns predictions and evaluation metrics
    """
    logger.info(
        f"Prediction request received: ticker={request.ticker}, "
        f"adaptive={request.use_adaptive}"
    )

    try:
        pipeline = ForecastPipeline(
            ticker=request.ticker.upper(),
            use_adaptive=request.use_adaptive,
            period=request.period
        )
        results = pipeline.run()
        return PredictResponse(**results)

    except ValueError as e:
        logger.warning(f"Validation error for {request.ticker}: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Pipeline error for {request.ticker}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal pipeline error: {str(e)}"
        )
