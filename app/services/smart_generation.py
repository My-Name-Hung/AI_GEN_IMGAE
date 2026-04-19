"""
Smart Generation Service - Full pipeline: analyze → generate → CLIP filter → auto-regenerate.
Uses unified InferenceService for automatic backend selection (CUDA → OpenVINO GPU → CPU).
"""
import logging
import random
import base64
import io
from typing import List, Optional

from PIL import Image

from app.services.inference import get_inference_service, InferenceService
from app.services.prompt_analyzer import get_prompt_analyzer, GenerationConfig
from app.services.clip import CLIPService

logger = logging.getLogger(__name__)

# Minimum CLIP score to accept the image (0.0–1.0)
MIN_CLIP_SCORE = 0.15
# Max auto-regenerations if CLIP score too low
MAX_AUTO_RETRIES = 2


class SmartGenerationService:
    """
    End-to-end intelligent generation pipeline.

    Flow:
      1. Analyze prompt → SmartPromptAnalyzer
      2. Generate image → InferenceService (auto-selects CUDA / OpenVINO GPU / CPU)
      3. Score with CLIP → CLIPService
      4. If score < threshold: auto-regenerate up to MAX_AUTO_RETRIES times
      5. Return best image + metadata
    """

    def __init__(self):
        self._inference = get_inference_service()
        self._analyzer = get_prompt_analyzer()
        self._clip = CLIPService()

        # Log which backend is being used
        status = self._inference.get_status()
        logger.info(
            "[SmartGen] Backend: %s (%s) — Model: %s",
            status.get("backend"), status.get("impl"), status.get("model"),
        )

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        seed: Optional[int] = None,
        num_images: int = 1,
        mode: str = "turbo",
        lora_type: Optional[str] = None,
        lora_scale: float = 1.0,
        lora_stack: bool = False,
        enable_clip_filter: bool = True,
        enable_auto_retry: bool = True,
    ) -> dict:
        """
        Generate images with smart analysis and CLIP quality filtering.

        Args:
            prompt: Natural language prompt (Vietnamese or English)
            negative_prompt: Override auto-generated negative prompt (optional)
            width, height: Image resolution
            num_inference_steps, guidance_scale: Inference params
            seed: Fixed seed for reproducibility (optional)
            num_images: Number of images to generate
            mode: "turbo" | "standard" | "quality"
            lora_type: Override auto-detected LoRA type (optional)
            lora_scale: LoRA weight scale
            lora_stack: Stack LoRAs instead of replacing
            enable_clip_filter: Run CLIP scoring on generated images
            enable_auto_retry: Auto-regenerate if CLIP score too low

        Returns:
            dict with: images (list of base64), metadata, clip_scores, analysis
        """
        # ── Step 1: Analyze prompt ──────────────────────────────────────────
        try:
            analysis = self._analyzer.analyze(prompt)
        except Exception as e:
            logger.warning("[SmartGen] Prompt analysis failed: %s — using defaults", e)
            analysis = self._analyzer.analyze("modern logo design")

        # Override with explicit params if provided
        if lora_type is not None:
            analysis.lora_type = lora_type
        if negative_prompt:
            analysis.negative_prompt = negative_prompt
        if mode is not None:
            analysis.mode = mode
        if width != 1024 or height != 1024:
            analysis.width = width
            analysis.height = height
        if num_inference_steps != 4:
            analysis.num_inference_steps = num_inference_steps
        if guidance_scale != 0.0:
            analysis.guidance_scale = guidance_scale

        # Force turbo + 2 steps when on CPU backend for speed
        status = self._inference.get_status()
        if status.get("backend") == "cpu":
            analysis.mode = "turbo"
            analysis.num_inference_steps = 2
            analysis.guidance_scale = 0.0
            logger.info("[SmartGen] CPU backend — forcing turbo mode, 2 steps")

        logger.info(
            "[SmartGen] prompt='%s' → lora=%s mode=%s steps=%d cfg=%.1f size=%dx%d",
            prompt[:80], analysis.lora_type, analysis.mode,
            analysis.num_inference_steps, analysis.guidance_scale,
            analysis.width, analysis.height,
        )

        # ── Step 2: Generate images ──────────────────────────────────────────
        if seed is not None:
            seeds = [seed] * num_images
        else:
            seeds = [random.randint(0, 2**31 - 1) for _ in range(num_images)]

        all_images: List[Image.Image] = []
        all_seeds: List[int] = []
        all_clip_scores: List[float] = []

        for i, cur_seed in enumerate(seeds):
            attempt = 0
            best_image = None
            best_score = -1.0

            while attempt <= MAX_AUTO_RETRIES:
                actual_seed = cur_seed if attempt == 0 else random.randint(0, 2**31 - 1)

                logger.info(
                    "[SmartGen] Generating image %d/%d (seed=%d, attempt=%d)",
                    i + 1, num_images, actual_seed, attempt + 1,
                )
                logger.info(
                    "[SmartGen] Full prompt sent to model:\n%s",
                    analysis.enhanced_prompt,
                )

                try:
                    images = self._inference.generate(
                        prompt=analysis.enhanced_prompt,
                        negative_prompt=analysis.negative_prompt,
                        width=analysis.width,
                        height=analysis.height,
                        num_inference_steps=analysis.num_inference_steps,
                        guidance_scale=analysis.guidance_scale,
                        seed=actual_seed,
                        num_images=1,
                        mode=analysis.mode,
                        lora_type=analysis.lora_type,
                        lora_scale=lora_scale,
                        lora_stack=lora_stack if attempt == 0 else True,
                    )
                except Exception as e:
                    logger.error("[SmartGen] Generation failed: %s", e)
                    break

                if not images:
                    logger.warning("[SmartGen] No images returned")
                    break

                img = images[0]

                # ── Step 3: CLIP scoring ────────────────────────────────────
                if enable_clip_filter:
                    try:
                        score = self._clip.compute_score(img, prompt)
                        score_val = float(score)
                        logger.info(
                            "[SmartGen] CLIP score=%.4f (threshold=%.4f)",
                            score_val, MIN_CLIP_SCORE,
                        )
                    except Exception as e:
                        logger.warning("[SmartGen] CLIP scoring failed: %s", e)
                        score_val = 1.0  # accept if scoring fails

                    if score_val > best_score:
                        best_score = score_val
                        best_image = img
                else:
                    best_image = img
                    best_score = 1.0

                # Accept if above threshold, or no more retries
                if best_score >= MIN_CLIP_SCORE or attempt >= MAX_AUTO_RETRIES:
                    break

                logger.info(
                    "[SmartGen] CLIP score too low (%.4f < %.4f), retrying...",
                    best_score, MIN_CLIP_SCORE,
                )
                attempt += 1

            if best_image is not None:
                all_images.append(best_image)
                all_seeds.append(actual_seed)
                all_clip_scores.append(best_score)

        # ── Step 4: Encode results ────────────────────────────────────────────
        images_base64 = []
        for img in all_images:
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            images_base64.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))

        # ── Step 5: Build response ───────────────────────────────────────────
        lora_labels = {
            "lora_logo_2d": "Logo 2D — Flat/Minimalist",
            "lora_logo_3d": "Logo 3D — 3D Render",
            "lora_poster": "Poster — Movie/Event",
            "base": "Base Model (SDXL Turbo)",
        }

        return {
            "success": len(all_images) > 0,
            "images": images_base64,
            "metadata": {
                "original_prompt": prompt,
                "enhanced_prompt": analysis.enhanced_prompt,
                "negative_prompt": analysis.negative_prompt,
                "lora_type": analysis.lora_type,
                "lora_label": lora_labels.get(analysis.lora_type, analysis.lora_type),
                "mode": analysis.mode,
                "num_inference_steps": analysis.num_inference_steps,
                "guidance_scale": analysis.guidance_scale,
                "width": analysis.width,
                "height": analysis.height,
                "effective_width": all_images[0].width if all_images else analysis.width,
                "effective_height": all_images[0].height if all_images else analysis.height,
                "num_images": len(all_images),
                "seeds": all_seeds,
                "provider": "local_diffusion",
                "subject_type": analysis.subject_type,
                "style_keywords": analysis.style_keywords,
                "auto_analysis_confidence": analysis.confidence,
            },
            "clip_scores": all_clip_scores if enable_clip_filter else None,
            "analysis": {
                "lora_type": analysis.lora_type,
                "lora_label": lora_labels.get(analysis.lora_type, ""),
                "mode": analysis.mode,
                "subject_type": analysis.subject_type,
                "styles": analysis.style_keywords,
                "confidence": analysis.confidence,
                "enhanced_prompt": analysis.enhanced_prompt,
            },
        }


_smart_service: Optional[SmartGenerationService] = None


def get_smart_service() -> SmartGenerationService:
    global _smart_service
    if _smart_service is None:
        _smart_service = SmartGenerationService()
    return _smart_service
