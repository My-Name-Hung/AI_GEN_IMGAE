"""
Diffusion Service - Stable Diffusion XL Turbo Inference
Uses diffusers pipeline with optional trained LoRA integration.
"""
import logging
import os
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class DiffusionService:
    _instance = None
    _pipeline = None
    _load_error = None
    _loaded_model_name = None
    _active_lora_path = None
    _active_lora_scale = None
    _fallback_used = False
    _requested_model_name = None
    _lora_compatibility_warning = None

    def __init__(self, model_name: str = "stabilityai/sdxl-turbo"):
        """
        Initialize Diffusion Service.

        - GPU: default SDXL Turbo
        - CPU: fallback to SD 1.5 for local testing
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.gpu_model_name = model_name
        self.cpu_model_name = os.getenv("CPU_DIFFUSION_MODEL", "runwayml/stable-diffusion-v1-5")
        self.cpu_fallback_model_name = os.getenv("CPU_FALLBACK_MODEL", "segmind/tiny-sd")
        self.profile = os.getenv("LOCAL_INFER_PROFILE", "balanced").lower()
        self.skip_primary_cpu_model = os.getenv("SKIP_PRIMARY_CPU_MODEL", "false").lower() in {"1", "true", "yes"}
        self.model_name = self.gpu_model_name if self.device == "cuda" else self.cpu_model_name
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")

        if self.device == "cpu" and self.profile == "fast":
            self.model_name = self.cpu_fallback_model_name

        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.default_lora_path = self._resolve_default_lora_path()

    @property
    def pipeline(self):
        """Lazy-load pipeline."""
        if DiffusionService._pipeline is None:
            if DiffusionService._load_error:
                raise RuntimeError(f"Model is unavailable: {DiffusionService._load_error}")
            self._load_model()
        return DiffusionService._pipeline

    def _resolve_default_lora_path(self) -> Optional[str]:
        """Find a default trained LoRA path from env/common project paths."""
        env_path = os.getenv("DEFAULT_LORA_PATH")
        candidates = []
        if env_path:
            candidates.append(Path(env_path))

        cwd = Path.cwd()
        candidates.extend([
            cwd / "lora_poster" / "final" / "unet_lora",
            cwd / "outputs" / "lora_poster" / "final" / "unet_lora",
        ])

        for c in candidates:
            if self._is_valid_lora_dir(c):
                return str(c)
        return None

    @staticmethod
    def _is_valid_lora_dir(path: Path) -> bool:
        """Check whether path looks like a valid PEFT LoRA export folder."""
        if not path or not path.exists() or not path.is_dir():
            return False
        config = path / "adapter_config.json"
        has_weights = (path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists()
        return config.exists() and has_weights

    def get_status(self) -> dict:
        """Return model + LoRA status."""
        return {
            "device": self.device,
            "profile": self.profile,
            "skip_primary_cpu_model": self.skip_primary_cpu_model,
            "target_model": self.model_name,
            "requested_model": DiffusionService._requested_model_name,
            "loaded": DiffusionService._pipeline is not None,
            "loaded_model": DiffusionService._loaded_model_name,
            "load_error": DiffusionService._load_error,
            "fallback_used": DiffusionService._fallback_used,
            "cpu_fallback_model": self.cpu_fallback_model_name,
            "default_lora_path": self.default_lora_path,
            "default_lora_available": bool(self.default_lora_path),
            "active_lora_path": DiffusionService._active_lora_path,
            "active_lora_scale": DiffusionService._active_lora_scale,
            "lora_compatibility_warning": DiffusionService._lora_compatibility_warning,
        }

    def warmup(self) -> dict:
        """Force model load and return status."""
        _ = self.pipeline
        return self.get_status()

    def _try_load_pipeline(self, model_name: str):
        """Attempt to load a model pipeline by name."""
        from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler

        model_name_lower = model_name.lower()
        tiny_like_model = any(token in model_name_lower for token in ["tiny", "segmind/tiny-sd"])

        base_kwargs = {
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": True,
        }
        if self.hf_token:
            base_kwargs["token"] = self.hf_token
        if self.device == "cuda":
            base_kwargs["variant"] = "fp16"

        # First try safetensors for speed/security; if missing, retry with pytorch binaries.
        try:
            pipeline = AutoPipelineForText2Image.from_pretrained(
                model_name,
                use_safetensors=True,
                **base_kwargs,
            )
        except OSError as e:
            error_text = str(e).lower()
            if "safetensors" not in error_text:
                raise
            logger.warning(
                "Model %s has no safetensors weights, retrying with use_safetensors=False",
                model_name,
            )
            pipeline = AutoPipelineForText2Image.from_pretrained(
                model_name,
                use_safetensors=False,
                **base_kwargs,
            )

        pipeline = pipeline.to(self.device)

        if self.device == "cuda":
            try:
                pipeline.enable_xformers_memory_efficient_attention()
            except Exception:
                logger.info("xformers not available")

        if self.device == "cpu" and tiny_like_model:
            # Tiny community models can have component shape quirks with safety checker.
            if hasattr(pipeline, "safety_checker"):
                pipeline.safety_checker = None
            if hasattr(pipeline, "requires_safety_checker"):
                pipeline.requires_safety_checker = False
            return pipeline

        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        return pipeline

    def _load_model(self):
        """Load diffusion model with CPU fallback on MemoryError."""
        DiffusionService._requested_model_name = self.model_name
        DiffusionService._fallback_used = False

        primary_model = self.model_name
        should_try_primary = not (
            self.device == "cpu"
            and (
                self.profile == "fast"
                or self.skip_primary_cpu_model
                or primary_model == self.cpu_fallback_model_name
            )
        )

        logger.info(
            "Diffusion profile=%s device=%s primary=%s fallback=%s try_primary=%s",
            self.profile,
            self.device,
            primary_model,
            self.cpu_fallback_model_name,
            should_try_primary,
        )

        if should_try_primary:
            try:
                logger.info("Loading model: %s on %s", primary_model, self.device)
                DiffusionService._pipeline = self._try_load_pipeline(primary_model)
                DiffusionService._loaded_model_name = primary_model
                DiffusionService._load_error = None
                logger.info("Model loaded successfully")
                return
            except MemoryError as e:
                logger.exception("Failed to load model (MemoryError)")
                if self.device != "cpu":
                    DiffusionService._pipeline = None
                    DiffusionService._loaded_model_name = None
                    DiffusionService._load_error = (
                        "Out of RAM while loading diffusion model. "
                        "Increase Windows virtual memory, use a lighter CPU model, or use GPU."
                    )
                    raise RuntimeError(DiffusionService._load_error) from e
            except Exception as e:
                DiffusionService._pipeline = None
                DiffusionService._loaded_model_name = None
                DiffusionService._load_error = f"{type(e).__name__}: {str(e)}"
                logger.exception("Failed to load model")
                raise RuntimeError(f"Model load failed: {type(e).__name__}: {str(e)}") from e

        if self.device == "cpu":
            try:
                if should_try_primary:
                    logger.warning(
                        "Primary CPU model unavailable. Falling back to lighter model: %s",
                        self.cpu_fallback_model_name,
                    )
                else:
                    logger.info("Using fast CPU fallback model directly: %s", self.cpu_fallback_model_name)

                DiffusionService._pipeline = self._try_load_pipeline(self.cpu_fallback_model_name)
                DiffusionService._loaded_model_name = self.cpu_fallback_model_name
                DiffusionService._fallback_used = True
                DiffusionService._load_error = None
                logger.info("Fallback model loaded successfully")
                return
            except Exception as fallback_error:
                DiffusionService._pipeline = None
                DiffusionService._loaded_model_name = None
                DiffusionService._load_error = (
                    "CPU model load failed (including fallback): "
                    f"{type(fallback_error).__name__}: {str(fallback_error)}"
                )
                logger.exception("Fallback model load failed")
                raise RuntimeError(DiffusionService._load_error) from fallback_error

        DiffusionService._pipeline = None
        DiffusionService._loaded_model_name = None
        DiffusionService._load_error = "Unknown model loading failure"
        raise RuntimeError(DiffusionService._load_error)

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
        lora_path: Optional[str] = None,
        use_default_lora: bool = True,
        lora_scale: float = 1.0,
    ) -> List[Image.Image]:
        """Generate images from prompt with optional LoRA."""
        width = max(256, (width // 8) * 8)
        height = max(256, (height // 8) * 8)

        if mode == "turbo":
            num_inference_steps = min(num_inference_steps, 4)
            guidance_scale = 0.0
        elif mode == "standard":
            num_inference_steps = min(num_inference_steps, 8)
            guidance_scale = max(guidance_scale, 7.5)
        elif mode == "quality":
            num_inference_steps = max(num_inference_steps, 20)
            guidance_scale = max(guidance_scale, 7.5)

        # CPU safety/performance guardrails for local testing.
        if self.device == "cpu":
            # Cap resolution to keep latency reasonable on CPU.
            if DiffusionService._fallback_used:
                width = min(width, 256)
                height = min(height, 256)
                num_inference_steps = min(num_inference_steps, 2)
            else:
                width = min(width, 512)
                height = min(height, 512)
                num_inference_steps = min(num_inference_steps, 4)

            # Keep guidance cheap on CPU.
            guidance_scale = min(guidance_scale, 3.0)

        if seed is None:
            import random
            seed = random.randint(0, 2**31 - 1)

        generator = torch.Generator(device=self.device).manual_seed(int(seed))

        resolved_lora_path = lora_path
        if not resolved_lora_path and use_default_lora:
            resolved_lora_path = self.default_lora_path

        loaded_lora = False
        pipeline = self.pipeline
        DiffusionService._lora_compatibility_warning = None

        if resolved_lora_path:
            loaded_model_name = (DiffusionService._loaded_model_name or "").lower()
            likely_sdxl_base = "sdxl" in loaded_model_name
            if not likely_sdxl_base:
                DiffusionService._lora_compatibility_warning = (
                    "Requested LoRA path may be SDXL-trained, but current loaded base model is "
                    f"'{DiffusionService._loaded_model_name}'. LoRA was skipped to avoid incompatibility."
                )
                logger.warning(DiffusionService._lora_compatibility_warning)
            else:
                try:
                    self.load_custom_lora(resolved_lora_path, scale=lora_scale)
                    loaded_lora = True
                    logger.info("LoRA loaded from %s (scale=%.2f)", resolved_lora_path, lora_scale)
                except Exception as e:
                    logger.warning(
                        "Failed to load LoRA from %s: %s — generating with base model",
                        resolved_lora_path,
                        e,
                    )

        try:
            images = []
            for i in range(num_images):
                logger.info("Generating image %d/%d", i + 1, num_images)
                result = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    num_images_per_prompt=1,
                )
                images.extend(result.images)
        finally:
            if loaded_lora:
                try:
                    self.unload_lora()
                    logger.info("LoRA unloaded after generation")
                except Exception as e:
                    logger.warning("LoRA unload warning: %s", e)

        return images

    def load_custom_lora(self, lora_path: str, scale: float = 1.0):
        """Load LoRA adapter from a local folder path."""
        path = Path(lora_path)
        if not self._is_valid_lora_dir(path):
            raise ValueError(
                f"Invalid LoRA path: {lora_path}. "
                "Expected adapter_config.json and adapter_model.safetensors/bin"
            )

        self.pipeline.load_lora_weights(str(path), adapter_name="active_lora")
        self.pipeline.set_adapters(["active_lora"], adapter_weights=[float(scale)])

        DiffusionService._active_lora_path = str(path)
        DiffusionService._active_lora_scale = float(scale)
        logger.info("Loaded custom LoRA from %s", path)

    def unload_lora(self):
        """Unload currently active LoRA."""
        self.pipeline.unload_lora_weights()
        DiffusionService._active_lora_path = None
        DiffusionService._active_lora_scale = None
        logger.info("Unloaded LoRA weights")
