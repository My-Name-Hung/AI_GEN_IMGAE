"""
OpenVINO GPU Pipeline Service - Stable Diffusion on Intel GPU via OpenVINO.
Uses optimum-intel to load SDXL Turbo with OpenVINO on GPU.
"""
import logging
import os
from typing import List, Optional

from PIL import Image

logger = logging.getLogger(__name__)


class OVStableDiffusionService:
    """
    OpenVINO-powered Stable Diffusion pipeline for Intel GPU (Integrated + Discrete).

    Uses optimum-intel's OVStableDiffusionPipeline which:
    - Converts SDXL to OpenVINO IR (FP16)
    - Runs inference on Intel GPU via oneAPI/OpenVINO
    - Supports LoRA via load_lora_weights()
    - Supports OpenVINO's model compilation for faster subsequent runs
    """
    _instance = None
    _pipeline = None
    _load_error = None
    _loaded_model_name = None
    _requested_model_name = None
    _ov_compiled = False

    def __init__(
        self,
        model_name: str = "stabilityai/sdxl-turbo",
        compile_on_load: bool = False,
    ):
        self.model_name = model_name
        self.compile_on_load = compile_on_load
        self.device = "GPU"  # OpenVINO GPU device (Intel integrated/discrete)
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")

        # Check OpenVINO availability
        try:
            import openvino as ov
            core = ov.Core()
            devices = core.available_devices
            if "GPU" not in devices:
                logger.warning(
                    "OpenVINO GPU device not found. Available: %s. Falling back to CPU.",
                    devices,
                )
                self.device = "CPU"
            else:
                logger.info("OpenVINO devices available: %s", devices)
        except ImportError:
            logger.error("OpenVINO not installed. Run: pip install openvino optimum-intel[openvino]")
            self.device = "CPU"

        self._lora_manager = None
        try:
            from app.services.lora_manager import get_lora_manager
            self._lora_manager = get_lora_manager()
        except Exception:
            logger.warning("LoRAManager not available for OV pipeline")

    @property
    def pipeline(self):
        if OVStableDiffusionService._pipeline is None:
            if OVStableDiffusionService._load_error:
                raise RuntimeError(f"OpenVINO model unavailable: {OVStableDiffusionService._load_error}")
            self._load_model()
        return OVStableDiffusionService._pipeline

    def _load_model(self):
        """Load SDXL Turbo with OpenVINO on GPU."""
        OVStableDiffusionService._requested_model_name = self.model_name
        logger.info(
            "Loading OpenVINO pipeline: %s on device=%s",
            self.model_name,
            self.device,
        )

        try:
            from optimum.intel.openvino import OVStableDiffusionPipeline

            load_kwargs = {
                "device": self.device,
                "compile": self.compile_on_load,
            }
            if self.hf_token:
                load_kwargs["token"] = self.hf_token

            logger.info("Creating OVStableDiffusionPipeline (device=%s)...", self.device)
            OVStableDiffusionService._pipeline = OVStableDiffusionPipeline.from_pretrained(
                self.model_name,
                export=False,
                **load_kwargs,
            )

            # Compile model for faster inference after first run
            if self.compile_on_load and self.device == "GPU":
                try:
                    logger.info("Compiling OpenVINO model for GPU (one-time, may take 2-5 min)...")
                    OVStableDiffusionService._pipeline.compile()
                    OVStableDiffusionService._ov_compiled = True
                    logger.info("OpenVINO model compiled successfully.")
                except Exception as e:
                    logger.warning("OV compilation failed (non-fatal): %s", e)

            OVStableDiffusionService._loaded_model_name = self.model_name
            OVStableDiffusionService._load_error = None
            logger.info(
                "OpenVINO pipeline loaded: %s on %s",
                self.model_name,
                self.device,
            )
        except Exception as e:
            OVStableDiffusionService._pipeline = None
            OVStableDiffusionService._loaded_model_name = None
            OVStableDiffusionService._load_error = f"{type(e).__name__}: {str(e)}"
            logger.exception("OpenVINO pipeline load failed")
            raise

    def get_status(self) -> dict:
        lora_status = (
            self._lora_manager.get_status()
            if self._lora_manager else {"discovered": [], "loaded_adapters": [], "active_weights": {}}
        )
        return {
            "device": self.device,
            "backend": "openvino",
            "target_model": self.model_name,
            "requested_model": OVStableDiffusionService._requested_model_name,
            "loaded": OVStableDiffusionService._pipeline is not None,
            "loaded_model": OVStableDiffusionService._loaded_model_name,
            "compiled": OVStableDiffusionService._ov_compiled,
            "load_error": OVStableDiffusionService._load_error,
            "lora": {
                "discovered": lora_status["discovered"],
                "loaded_adapters": lora_status["loaded_adapters"],
                "active_weights": lora_status["active_weights"],
            },
        }

    def warmup(self) -> dict:
        _ = self.pipeline  # ensure loaded
        return self.get_status()

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        seed: Optional[int] = None,
        num_images: int = 1,
        mode: str = "turbo",
        lora_type: Optional[str] = None,
        lora_path: Optional[str] = None,
        lora_scale: float = 1.0,
        lora_stack: bool = False,
    ) -> List[Image.Image]:
        """
        Generate images using OpenVINO GPU pipeline with optional LoRA.

        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt
            width, height: Resolution (multiple of 8)
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale (0 for turbo mode)
            seed: Random seed
            num_images: Number of images
            mode: "turbo" | "standard" | "quality"
            lora_type: LoRA adapter type (lora_logo_2d, lora_logo_3d, lora_poster)
            lora_path: Custom LoRA path
            lora_scale: LoRA weight scale
            lora_stack: Stack LoRAs

        Returns:
            List of PIL Images
        """
        # Enforce 8-pixel alignment
        width = max(256, (width // 8) * 8)
        height = max(256, (height // 8) * 8)

        # Mode-specific overrides
        if mode == "turbo":
            num_inference_steps = min(num_inference_steps, 4)
            guidance_scale = 0.0
        elif mode == "standard":
            num_inference_steps = min(num_inference_steps, 8)
            guidance_scale = max(guidance_scale, 3.5)
        elif mode == "quality":
            num_inference_steps = max(num_inference_steps, 20)
            guidance_scale = max(guidance_scale, 7.5)

        # Seed
        if seed is None:
            import random
            seed = random.randint(0, 2**31 - 1)

        pipeline = self.pipeline

        # ── LoRA handling ───────────────────────────────────────────────────
        lora_loaded = False

        lora_key = None
        if lora_path:
            lora_key = "_custom_path"
        elif lora_type:
            lora_key = lora_type

        if lora_key and self._lora_manager:
            adapter_info = self._lora_manager._discovered.get(lora_key)
            if adapter_info:
                try:
                    lora_loaded = self._lora_manager.load_adapter(
                        lora_key, pipeline,
                        scale=lora_scale, stack=lora_stack,
                    )
                    if lora_loaded:
                        logger.info(
                            "OV: LoRA '%s' loaded (scale=%.2f, stack=%s)",
                            lora_key, lora_scale, lora_stack,
                        )
                except Exception as e:
                    logger.warning("OV: LoRA load failed: %s", e)
                    lora_loaded = False

        try:
            images = []
            for i in range(num_images):
                logger.info(
                    "[OV] Generating image %d/%d (seed=%d, steps=%d)",
                    i + 1, num_images, seed + i, num_inference_steps,
                )
                result = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=1,
                )
                images.extend(result.images)
        finally:
            if lora_loaded and self._lora_manager:
                try:
                    self._lora_manager.unload_all(pipeline)
                except Exception:
                    pass

        return images

    def unload_lora(self):
        if self._lora_manager:
            self._lora_manager.unload_all(self.pipeline)


# ─── Module-level singleton ────────────────────────────────────────────────────

_ov_service: Optional[OVStableDiffusionService] = None


def get_ov_service() -> OVStableDiffusionService:
    global _ov_service
    if _ov_service is None:
        _ov_service = OVStableDiffusionService()
    return _ov_service
