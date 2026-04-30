from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from app.core.logger import logger
from ml.pipeline import ForecastPipeline
import json
import os
from app.core.config import config

router = APIRouter(tags=["Training"])

# Simple in-memory job tracker (resets on restart — acceptable for free tier)
_job_status: dict = {}


class TrainRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=20, example="MSFT")
    period: str = Field("5y", description="yfinance period string")
    use_adaptive: bool = False


class TrainResponse(BaseModel):
    job_id: str
    status: str
    message: str


class TrainStatusResponse(BaseModel):
    job_id: str
    status: str          # "pending" | "running" | "complete" | "failed"
    result: dict = None
    error: str = None


def _run_training_job(job_id: str, ticker: str, period: str, adaptive: bool):
    """Background task: runs pipeline and stores result."""
    _job_status[job_id] = {"status": "running", "result": None, "error": None}
    try:
        pipeline = ForecastPipeline(
            ticker=ticker, use_adaptive=adaptive, period=period
        )
        result = pipeline.run()

        # Persist result to disk for retrieval
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        result_path = os.path.join(
            config.RESULTS_DIR, f"{job_id}.json"
        )
        with open(result_path, "w") as f:
            json.dump(result, f)

        _job_status[job_id] = {
            "status": "complete",
            "result": result,
            "error": None
        }
        logger.info(f"Training job {job_id} completed successfully.")

    except Exception as e:
        logger.error(f"Training job {job_id} failed: {e}")
        _job_status[job_id] = {
            "status": "failed",
            "result": None,
            "error": str(e)
        }


@router.post(
    "/train",
    response_model=TrainResponse,
    summary="Trigger async training pipeline"
)
def trigger_training(
    request: TrainRequest,
    background_tasks: BackgroundTasks
):
    """
    Start the full pipeline as a background job.
    Returns a job_id immediately. Poll /api/train/status/{job_id}
    to check completion.
    """
    import uuid
    job_id = f"{request.ticker.upper()}_{uuid.uuid4().hex[:8]}"
    _job_status[job_id] = {"status": "pending", "result": None, "error": None}

    background_tasks.add_task(
        _run_training_job,
        job_id=job_id,
        ticker=request.ticker.upper(),
        period=request.period,
        adaptive=request.use_adaptive
    )

    logger.info(f"Training job queued: {job_id}")
    return TrainResponse(
        job_id=job_id,
        status="pending",
        message=f"Training started for {request.ticker.upper()}. "
                f"Poll /api/train/status/{job_id} for updates."
    )


@router.get(
    "/train/status/{job_id}",
    response_model=TrainStatusResponse,
    summary="Check training job status"
)
def get_training_status(job_id: str):
    if job_id not in _job_status:
        # Try loading from disk (survives in-memory reset)
        result_path = os.path.join(config.RESULTS_DIR, f"{job_id}.json")
        if os.path.exists(result_path):
            with open(result_path) as f:
                result = json.load(f)
            return TrainStatusResponse(
                job_id=job_id, status="complete", result=result
            )
        raise HTTPException(
            status_code=404, detail=f"Job '{job_id}' not found."
        )

    job = _job_status[job_id]
    return TrainStatusResponse(
        job_id=job_id,
        status=job["status"],
        result=job.get("result"),
        error=job.get("error")
    )
