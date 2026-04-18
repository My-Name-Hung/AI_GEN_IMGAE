"""
Vectorization Router - POST /api/vectorize
PNG/JPG to SVG conversion using OpenCV contour tracing + RDP simplification.
"""
import base64
import io
import logging
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from PIL import Image

from app.models.schemas import VectorizeRequest, VectorizeResponse
from app.services.vectorizer import get_vectorizer_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/vectorize", response_model=VectorizeResponse)
async def vectorize_image(request: VectorizeRequest):
    """
    Convert raster image (PNG/JPG) to vector SVG.

    Algorithm:
      1. Color quantization (K-means / PIL median-cut)
      2. Per-color layer extraction via OpenCV inRange contours
      3. Path simplification (Ramer-Douglas-Peucker)
      4. Bezier smoothing (Catmull-Rom splines)
      5. Grouped SVG output

    Output:
      - svg_content: full SVG string
      - png_base64: quantized reference PNG
      - layers_json: layer metadata
    """
    t0 = time.time()

    try:
        image_data = base64.b64decode(request.image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64: {e}")

    try:
        image = Image.open(io.BytesIO(image_data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    svc = get_vectorizer_service()

    # Output directory
    out_dir = None
    filename_base = "vectorized"
    if request.output_dir:
        out_dir = request.output_dir
        filename_base = request.filename_base or "vectorized"

    result = svc.vectorize(
        image=image,
        color_quantization=request.color_quantization,
        simplify_tolerance=request.simplify_tolerance,
        bezier_smoothness=request.bezier_smoothness,
        output_dir=out_dir,
        filename_base=filename_base,
    )

    elapsed = time.time() - t0
    logger.info(
        "Vectorized %s → %d layers %d paths in %.1fs",
        image.size, result["metadata"]["num_colors"],
        result["metadata"]["total_paths"], elapsed,
    )

    return {
        "success": True,
        "svg_content": result["svg_content"],
        "png_base64": result["png_base64"],
        "layers_json": result["layers_json"],
        "metadata": result["metadata"],
    }
