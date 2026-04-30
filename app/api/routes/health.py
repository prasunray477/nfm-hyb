from fastapi import APIRouter
from app.core.config import config

router = APIRouter(tags=["System"])


@router.get("/health", summary="Service health check")
def health():
    return {
        "status": "healthy",
        "app": config.APP_NAME,
        "version": config.VERSION
    }
