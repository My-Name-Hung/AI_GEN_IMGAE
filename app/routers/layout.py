"""
Layout Router - POST /api/analyze-layout
Analyzes image layout, detects regions, and generates composition guides.
"""
import base64
import io
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from PIL import Image

from app.models.schemas import LayoutAnalyzeRequest, LayoutAnalyzeResponse
from app.services.layout import get_layout_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/analyze-layout", response_model=LayoutAnalyzeResponse)
async def analyze_layout(request: LayoutAnalyzeRequest):
    """
    Analyze image layout: detect text/icon/background regions,
    generate rule-of-thirds guides, find contrast zones for text overlay.

    Returns a structured JSON layout schema.
    """
    try:
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data))
    except Exception as e:
        logger.error("Failed to decode image: %s", e)
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

    svc = get_layout_service()

    result = svc.analyze(
        image=image,
        detect_text=request.detect_text,
        detect_icons=request.detect_icons,
        suggest_layout=request.suggest_layout,
        output_path=None,
    )

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    # Save schema to file if output_dir specified
    saved_schema = None
    if request.output_dir:
        try:
            out_dir = Path(request.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            schema_path = out_dir / "layout_schema.json"
            saved_schema = svc.export_schema(result, str(schema_path))
        except Exception as e:
            logger.warning("Could not save layout schema: %s", e)

    return {
        "success": True,
        "layout": result.get("layout", {}),
        "regions": result.get("regions", []),
        "texts": result.get("texts", []),
        "icons": result.get("icons", []),
        "background_zones": result.get("background_zones", []),
        "contrast_zones": result.get("contrast_zones", []),
        "image_size": result.get("image_size", {}),
        "summary": result.get("summary", ""),
        "saved_schema": saved_schema,
    }


@router.post("/suggest-composition")
async def suggest_composition(request: LayoutAnalyzeRequest):
    """
    Suggest text/logo placement positions for an empty canvas.

    Modes:
      - "centered"  : single centered zone
      - "thirds"    : rule-of-thirds grid placements
      - "diagonal"  : diagonal composition
      - "split"     : left/right split
    """
    try:
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

    svc = get_layout_service()
    result = svc.suggest_composition(
        image=image,
        mode=request.composition_mode,
    )
    return {"success": True, **result}
