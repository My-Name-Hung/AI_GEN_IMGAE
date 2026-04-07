"""
Vectorization Router - POST /vectorize
PNG/JPG to SVG conversion with Potrace
"""
from fastapi import APIRouter, HTTPException
import logging
import base64
import io
from PIL import Image

from app.models.schemas import VectorizeRequest, VectorizeResponse
from app.services.vectorizer import VectorizerService

logger = logging.getLogger(__name__)
router = APIRouter()

vectorizer_service = VectorizerService()


@router.post("/vectorize", response_model=VectorizeResponse)
async def vectorize_image(request: VectorizeRequest):
    """
    Convert raster image (PNG/JPG) to vector SVG.
    
    Uses Potrace for tracing with optional color quantization
    and Bezier curve fitting.
    """
    try:
        logger.info("Starting vectorization")
        
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        result = vectorizer_service.vectorize(
            image=image,
            color_quantization=request.color_quantization,
            simplify_tolerance=request.simplify_tolerance
        )
        
        return {
            "success": True,
            "svg_content": result.get("svg_content"),
            "png_base64": result.get("png_base64"),
            "layers_json": result.get("layers_json"),
            "metadata": {
                "original_size": result.get("original_size"),
                "num_colors": result.get("num_colors"),
                "num_paths": result.get("num_paths")
            }
        }
        
    except Exception as e:
        logger.error(f"Vectorization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
