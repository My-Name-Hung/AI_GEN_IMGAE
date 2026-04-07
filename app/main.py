"""
AI Designer System - FastAPI Backend
Entry point for the application
"""
import logging

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from app.routers import generate, train, vectorize
from app.routers.generate import diffusion_service
from app.routers.train import get_training_overview

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Designer System",
    description="Logo/Poster Generation with Diffusion + CLIP Guidance",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(generate.router, prefix="/api")
app.include_router(train.router, prefix="/api")
app.include_router(vectorize.router, prefix="/api")


@app.get("/health")
async def health_check(warmup: bool = Query(default=False)):
    """Health check endpoint with model and training status.

    Query params:
    - warmup=true: force-load model before returning health
    """
    if warmup:
        try:
            model_status = diffusion_service.warmup()
        except Exception:
            model_status = diffusion_service.get_status()
    else:
        model_status = diffusion_service.get_status()

    training_status = get_training_overview()

    if model_status.get("loaded"):
        overall = "healthy"
    elif model_status.get("load_error"):
        overall = "degraded"
    else:
        overall = "starting"

    return {
        "status": overall,
        "service": "ai-designer",
        "model": model_status,
        "training": training_status,
    }


@app.get("/api/model/status")
async def model_status():
    """Return diffusion model + default LoRA status."""
    return diffusion_service.get_status()


@app.post("/api/model/warmup")
async def model_warmup():
    """Force-load diffusion model and return status."""
    try:
        return {"success": True, "model": diffusion_service.warmup()}
    except Exception:
        return {"success": False, "model": diffusion_service.get_status()}


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "AI Designer System",
        "version": "1.0.0",
        "endpoints": {
            "generate": "/api/generate",
            "train": "/api/train",
            "vectorize": "/api/vectorize",
            "health": "/health",
            "model_status": "/api/model/status",
            "model_warmup": "/api/model/warmup",
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
