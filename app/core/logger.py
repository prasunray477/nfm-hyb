import os
import sys
from loguru import logger

# Remove default handler
logger.remove()

# Console handler — human-readable
logger.add(
    sys.stdout,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    ),
    level="INFO",
    colorize=True
)

# File handler — structured for debugging
os.makedirs("data/results", exist_ok=True)
logger.add(
    "data/results/app.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} | {message}",
    rotation="10 MB",
    retention="30 days",
    level="DEBUG",
    enqueue=True        # Thread-safe async logging
)
