"""
Generation Router - POST /generate
Smart generation with auto LoRA detection, prompt enhancement, and CLIP quality filtering.
"""
from fastapi import APIRouter, HTTPException
import logging
import base64
import io
from pathlib import Path
from datetime import datetime
from PIL import Image

from app.models.schemas import GenerateRequest, GenerationResponse
from app.services.smart_generation import get_smart_service
from app.services.vectorizer import VectorizerService

logger = logging.getLogger(__name__)
router = APIRouter()

vectorizer_service = None


def _get_vectorizer():
    global vectorizer_service
    if vectorizer_service is None:
        vectorizer_service = VectorizerService()
    return vectorizer_service


def _save_generated_images(images, output_subdir: str | None = None) -> list[str]:
    """Persist generated images as PNG under outputs/generated and return saved paths."""
    base_dir = Path("outputs") / "generated"
    target_dir = base_dir / output_subdir if output_subdir else base_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    target_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    for idx, img in enumerate(images, start=1):
        file_path = target_dir / f"gen_{idx:02d}.png"
        img.save(file_path, format="PNG")
        saved_paths.append(str(file_path.resolve()))
    return saved_paths


def _base64_to_pil(image_b64: str):
    img_bytes = base64.b64decode(image_b64)
    return io.BytesIO(img_bytes)


@router.post("/generate", response_model=GenerationResponse)
async def generate_image(request: GenerateRequest):
    """
    Generate logo/poster images from natural language prompt.

    Pipeline:
      1. SmartPromptAnalyzer: auto-detect LoRA type, optimize params
      2. DiffusionService: generate with SDXL Turbo + LoRA
      3. CLIP filter: score images, auto-regenerate if quality too low
      4. Optional: SVG vectorization, file save

    Key features:
    - User just types natural language; system auto-selects model
    - CLIP quality filter ensures generated images match prompt
    - Auto-retry up to 2 times if CLIP score < threshold
    """
    try:
        smart_svc = get_smart_service()

        # Resolve lora_type override (None = auto-detect)
        lora_override = None
        if request.lora_type is not None:
            val = request.lora_type.value
            if val != "base":
                lora_override = val

        # Run smart generation
        result = smart_svc.generate(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt if request.negative_prompt else None,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
            num_images=request.num_images,
            mode=request.mode.value,
            lora_type=lora_override,
            lora_scale=request.lora_scale,
            lora_stack=request.lora_stack,
            enable_clip_filter=request.layout_aware,
            enable_auto_retry=True,
        )

        if not result.get("success") or not result.get("images"):
            raise HTTPException(
                status_code=500,
                detail="Generation failed: no images returned. Check backend logs."
            )

        images_base64 = result["images"]
        metadata = result["metadata"]

        # Optional SVG vectorization
        svg_paths = []
        if request.enable_vectorization and images_base64:
            try:
                pil_img = Image.open(_base64_to_pil(images_base64[0]))
                svg_result = _get_vectorizer().vectorize(pil_img)
                svg_paths = [svg_result.get("svg_content", "")]
                metadata["svg_num_paths"] = svg_result.get("num_paths", 0)
            except Exception as e:
                logger.warning("Vectorization failed: %s", e)
                metadata["svg_error"] = str(e)

        # Optional file save
        saved_paths = []
        if request.save_outputs and images_base64:
            try:
                pil_images = [
                    Image.open(_base64_to_pil(b64)) for b64 in images_base64
                ]
                saved_paths = _save_generated_images(pil_images, request.output_subdir)
                metadata["saved_outputs"] = saved_paths
            except Exception as save_error:
                logger.warning("Failed to save outputs: %s", save_error)
                metadata["save_error"] = str(save_error)

        return GenerationResponse(
            success=True,
            images=images_base64,
            svg_paths=svg_paths if svg_paths else None,
            metadata={
                **metadata,
                "lora_requested_path": request.lora_path,
                "use_default_lora": request.use_default_lora,
                "active_lora_adapters": result.get("analysis", {}).get("lora_type"),
                "auto_generated": True,
            },
            clip_scores=result.get("clip_scores"),
            analysis=result.get("analysis"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[Generate] Unhandled exception in generate_image")
        detail = str(e) if str(e) else f"{type(e).__name__} (empty message)"
        raise HTTPException(status_code=500, detail=detail)
