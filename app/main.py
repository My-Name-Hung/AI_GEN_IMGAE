"""
AI Designer System - FastAPI Backend
Entry point for the application
"""
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from app.routers import generate, train, vectorize, chat, layout
from app.services.inference import get_inference_service, detect_backend
from app.services.clip import CLIPService
from app.routers.train import get_training_overview
from app.services.lora_manager import get_lora_manager

# Global service instances (lazy-loaded)
_inference_service = None
_warmup_done = False

def get_inference_svc():
    global _inference_service
    if _inference_service is None:
        _inference_service = get_inference_service()
    return _inference_service


def _background_warmup():
    """Load all models in background thread so backend stays responsive."""
    import time

    global _warmup_done
    logger.info("[Warmup] Starting full model warmup in background thread...")

    # ── 1. Diffusion model ──────────────────────────────────────────────────
    try:
        svc = get_inference_svc()
        status = svc.warmup()
        backend = svc.backend
        logger.info(
            "[Warmup] Diffusion model ready — backend=%s, model=%s",
            backend, status.get("model", "unknown"),
        )
    except Exception as e:
        logger.warning("[Warmup] Diffusion model load failed: %s", e)

    # ── 2. CLIP model ──────────────────────────────────────────────────────
    try:
        clip_svc = CLIPService()
        _ = clip_svc.model  # trigger lazy load
        logger.info("[Warmup] CLIP model ready.")
    except Exception as e:
        logger.warning("[Warmup] CLIP model load failed (scoring will be slower on first use): %s", e)

    # ── 3. OpenCV (vectorizer) — fast import check ──────────────────────
    try:
        import cv2
        logger.info("[Warmup] OpenCV ready (version=%s)", cv2.__version__)
    except Exception as e:
        logger.warning("[Warmup] OpenCV import failed: %s", e)

    logger.info("[Warmup] === All models loaded ===")
    _warmup_done = True


# Ensure .env at project root is loaded before services initialize
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=False)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Backend starts instantly. Model loads in background thread."""
    logger.info("=== Backend ready ===")
    t = threading.Thread(target=_background_warmup, daemon=True)
    t.start()
    logger.info("Model warmup started in background (backend stays responsive)")
    yield
    logger.info("=== Backend shutting down ===")


app = FastAPI(
    title="AI Designer System",
    description="Logo/Poster Generation with Diffusion + CLIP Guidance",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(generate.router, prefix="/api")
app.include_router(train.router, prefix="/api")
app.include_router(vectorize.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(layout.router, prefix="/api")


@app.get("/health")
async def health_check(warmup: bool = Query(default=False)):
    """Health check endpoint with model and training status.

    Query params:
    - warmup=true: force-load model before returning health
    """
    if warmup:
        try:
            model_status = get_inference_svc().warmup()
        except Exception:
            model_status = get_inference_svc().get_status()
    else:
        model_status = get_inference_svc().get_status()

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
    """Return inference model + LoRA status."""
    svc = get_inference_svc()
    status = svc.get_status()
    status["warmup_done"] = _warmup_done
    return status


@app.post("/api/model/warmup")
async def model_warmup():
    """Force-load inference model."""
    try:
        return {"success": True, **get_inference_svc().warmup()}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/lora")
async def lora_status():
    """Return discovered and loaded LoRA adapters."""
    return get_lora_manager().get_status()


@app.post("/api/lora/{lora_type}/warmup")
async def lora_warmup(lora_type: str):
    """
    Pre-load a LoRA adapter into the pipeline (no generation).
    """
    try:
        manager = get_lora_manager()
        available = manager.available_types()
        if lora_type not in available:
            return {
                "success": False,
                "error": f"Unknown LoRA type '{lora_type}'. Available: {available}",
                "available": available,
            }
        svc = get_inference_svc()
        _ = svc.warmup()  # ensure model loaded
        # Get underlying diffusion pipeline from backend
        impl = svc._impl
        svc_obj = getattr(impl, "_svc", None)
        pipeline = getattr(impl, "_pipeline", None)
        if pipeline is None and svc_obj is not None:
            pipeline = getattr(svc_obj, "pipeline", None)
        if pipeline is None:
            return {"success": False, "error": "No pipeline available in current backend"}
        ok = manager.load_adapter(lora_type, pipeline, scale=1.0, stack=False)
        return {
            "success": ok,
            "lora_type": lora_type,
            "status": manager.get_status(),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/lora/unload")
async def lora_unload():
    """Unload all LoRA adapters from the pipeline."""
    try:
        manager = get_lora_manager()
        svc = get_inference_svc()
        impl = svc._impl
        pipeline = getattr(impl, "_pipeline", None) or getattr(impl, "_svc", None)
        if pipeline:
            manager.unload_all(pipeline)
        return {"success": True, "status": manager.get_status()}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/export/formats")
async def export_formats():
    """Return all supported export formats."""
    from app.services.export_service import SUPPORTED_FORMATS
    return {"formats": SUPPORTED_FORMATS, "default": "png"}


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
            "analyze_layout": "/api/analyze-layout",
            "suggest_composition": "/api/suggest-composition",
            "chat": "/api/chat",
            "export": "/api/export",
            "export_batch": "/api/export/batch",
            "export_formats": "/api/export/formats",
            "health": "/health",
            "model_status": "/api/model/status",
            "model_warmup": "/api/model/warmup",
            "lora_status": "/api/lora",
            "lora_warmup": "/api/lora/{lora_type}/warmup",
            "lora_unload": "/api/lora/unload",
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
