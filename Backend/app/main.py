from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.logging import logger
from app.services.model_service import model_service
from app.api.endpoints import router as api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing application and loading ML model...")
    try:
        model_service.load()
    except Exception as e:
        logger.error(f"Failed to load model during startup: {e}")
        raise e
    yield
    logger.info("Shutting down application...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "model_loaded": model_service.model is not None
    }

app.include_router(api_router, prefix=settings.API_V1_STR)
