import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── Application ───────────────────────────────────────
    APP_NAME: str = "Neutrosophic Stock Forecaster"
    VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False") == "True"

    # ── Server ────────────────────────────────────────────
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    FRONTEND_URL: str = os.getenv(
        "FRONTEND_URL", "http://localhost:8501"
    )

    # ── Filesystem ────────────────────────────────────────
    DATA_DIR: str = "data"
    RAW_DIR: str = "data/raw"
    PROCESSED_DIR: str = "data/processed"
    BILSTM_MODEL_DIR: str = os.getenv(
        "BILSTM_MODEL_DIR", "data/models/bilstm"
    )
    ARIMA_MODEL_DIR: str = os.getenv(
        "ARIMA_MODEL_DIR", "data/models/arima"
    )
    RESULTS_DIR: str = os.getenv("RESULTS_DIR", "data/results")

    # ── Data ──────────────────────────────────────────────
    MIN_TRADING_DAYS: int = 756          # ~3 years minimum
    DEFAULT_PERIOD: str = "5y"
    ADF_SIGNIFICANCE: float = 0.05

    # ── Neutrosophic Parameters ───────────────────────────
    ENTROPY_WINDOW: int = 10             # m = 10 days (optimal: 9-11)
    NUM_LINGUISTIC_LABELS: int = 5

    # ── Bi-LSTM Parameters ────────────────────────────────
    SEQUENCE_WINDOW: int = 60            # W = 60 days
    BILSTM_UNITS_L1: int = 128           # 64 forward + 64 backward
    BILSTM_UNITS_L2: int = 64            # 32 forward + 32 backward
    DENSE_UNITS: int = 32
    DROPOUT_L1: float = 0.30
    DROPOUT_L2: float = 0.20
    LEARNING_RATE: float = 0.001
    BATCH_SIZE: int = 32
    MAX_EPOCHS: int = 100
    EARLY_STOPPING_PATIENCE: int = 10
    LR_REDUCE_PATIENCE: int = 5
    LR_REDUCE_FACTOR: float = 0.5
    HUBER_DELTA: float = 1.0

    # ── ARIMA Parameters ──────────────────────────────────
    MAX_P: int = 5
    MAX_Q: int = 5
    MAX_D: int = 1

    # ── Aggregation ───────────────────────────────────────
    ALPHA_GRID_STEP: float = 0.01        # 101-point grid search
    ADAPTIVE_WINDOW: int = 20            # Rolling error window

    # ── Train/Val/Test Split ──────────────────────────────
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    # TEST_RATIO = 1 - 0.70 - 0.15 = 0.15  (implicit)


config = Config()
