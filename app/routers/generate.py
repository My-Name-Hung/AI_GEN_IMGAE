"""
Generation Router - POST /generate
Gemini-first image generation with local diffusion fallback.
"""
from fastapi import APIRouter, HTTPException
import logging
import base64
import io
from pathlib import Path
from datetime import datetime

from app.models.schemas import GenerateRequest, GenerationResponse
from app.services.diffusion import DiffusionService
from app.services.clip import CLIPService
from app.services.vectorizer import VectorizerService
from app.services.neural_inference import get_neural_service, NeuralServiceError

logger = logging.getLogger(__name__)
router = APIRouter()

diffusion_service = DiffusionService()
clip_service = CLIPService()
vectorizer_service = VectorizerService()


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


def _decode_base64_to_pil(image_b64: str):
    img_bytes = base64.b64decode(image_b64)
    return io.BytesIO(img_bytes)


@router.post("/generate", response_model=GenerationResponse)
async def generate_image(request: GenerateRequest):
    """
    Generate logo/poster images from text prompt.

    Priority:
    1) Gemini image API
    2) Local Diffusion fallback only when Gemini fails
    """
    provider = "gemini"
    gemini_error = None

    try:
        logger.info("Generating image with Gemini-first strategy. Prompt: %s", request.prompt)

        images_base64 = []
        generated_images = []

        # 1) Gemini first
        try:
            neural_service = get_neural_service()
            gemini_result = neural_service.generate_images(
                prompt=request.prompt,
                num_images=request.num_images,
            )
            images_base64 = gemini_result.get("images", [])

            # Convert to PIL only if needed by optional features
            if request.layout_aware or request.enable_vectorization or request.save_outputs:
                from PIL import Image

                for img_b64 in images_base64:
                    buffered = _decode_base64_to_pil(img_b64)
                    generated_images.append(Image.open(buffered))

        except (NeuralServiceError, RuntimeError, Exception) as e:
            gemini_error = str(e)
            provider = "local_diffusion"
            logger.warning("Gemini image generation failed, fallback to local diffusion: %s", gemini_error)

            # 2) Fallback local diffusion
            generated_images = diffusion_service.generate(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                width=request.width,
                height=request.height,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                seed=request.seed,
                num_images=request.num_images,
                mode=request.mode.value,
                lora_path=request.lora_path,
                use_default_lora=request.use_default_lora,
                lora_scale=request.lora_scale,
            )

            images_base64 = []
            for img in generated_images:
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                images_base64.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))

        clip_scores = []
        if request.layout_aware and generated_images:
            for img in generated_images:
                score = clip_service.compute_score(img, request.prompt)
                clip_scores.append(float(score))

        model_status = diffusion_service.get_status()

        response_data = {
            "success": True,
            "images": images_base64,
            "metadata": {
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "requested_width": request.width,
                "requested_height": request.height,
                "effective_width": generated_images[0].width if generated_images else request.width,
                "effective_height": generated_images[0].height if generated_images else request.height,
                "mode": request.mode.value,
                "num_images": len(images_base64),
                "provider": provider,
                "gemini_error": gemini_error,
                "lora_requested_path": request.lora_path,
                "use_default_lora": request.use_default_lora,
                "active_lora_path": model_status.get("active_lora_path"),
                "active_lora_scale": model_status.get("active_lora_scale"),
                "fallback_used": provider != "gemini" or model_status.get("fallback_used"),
                "loaded_model": model_status.get("loaded_model"),
                "lora_compatibility_warning": model_status.get("lora_compatibility_warning"),
            },
            "clip_scores": clip_scores if clip_scores else None,
        }

        if request.enable_vectorization and generated_images:
            svg_result = vectorizer_service.vectorize(generated_images[0])
            response_data["svg_paths"] = [svg_result.get("svg_content", "")]

        if request.save_outputs and generated_images:
            try:
                saved_paths = _save_generated_images(generated_images, request.output_subdir)
                response_data["metadata"]["saved_outputs"] = saved_paths
            except Exception as save_error:
                logger.warning("Failed to save outputs: %s", save_error)
                response_data["metadata"]["save_error"] = str(save_error)

        return response_data

    except Exception as e:
        logger.exception("Generation failed")
        detail = str(e) if str(e) else f"{type(e).__name__} (empty message)"
        raise HTTPException(status_code=500, detail=detail)
