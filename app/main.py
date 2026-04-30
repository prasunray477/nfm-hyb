from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import config
from app.core.logger import logger
from app.api.routes import predict, train, health

app = FastAPI(
    title=config.APP_NAME,
    version=config.VERSION,
    description=(
        "Neutrosophic Logic + Bi-LSTM + ARIMA "
        "hybrid stock price forecasting system."
    ),
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(health.router, prefix="/api")
app.include_router(predict.router, prefix="/api")
app.include_router(train.router, prefix="/api")


@app.on_event("startup")
async def startup_event():
    import os
    os.makedirs(config.BILSTM_MODEL_DIR, exist_ok=True)
    os.makedirs(config.ARIMA_MODEL_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    logger.info(f"{config.APP_NAME} v{config.VERSION} started.")


@app.get("/")
def root():
    return {
        "app": config.APP_NAME,
        "version": config.VERSION,
        "docs": "/docs"
    }
