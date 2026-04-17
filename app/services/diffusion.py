"""
Diffusion Service - Stable Diffusion XL Turbo Inference
Uses diffusers pipeline with optional trained LoRA integration.
"""
import gc
import logging
import os
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image

from app.services.lora_manager import get_lora_manager

logger = logging.getLogger(__name__)


class DiffusionService:
    _instance = None
    _pipeline = None
    _load_error = None
    _loaded_model_name = None
    _lora_compatibility_warning = None
    _requested_model_name = None
    _fallback_used = False

    def __init__(self, model_name: str = "stabilityai/sdxl-turbo"):
        # ── Device detection with env override ─────────────────────────────────
        force = os.getenv("FORCE_DEVICE", "").lower()
        if force in ("cpu", "cuda"):
            self.device = force
        else:
            # Auto-detect GPU (try nvidia-smi first for Windows reliability)
            has_gpu = False
            try:
                import subprocess
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5,
                )
                has_gpu = r.returncode == 0 and r.stdout.strip().isdigit()
            except Exception:
                pass
            if not has_gpu:
                has_gpu = torch.cuda.is_available()
            self.device = "cuda" if has_gpu else "cpu"

        self.gpu_model_name = os.getenv("GPU_MODEL", model_name)
        self.cpu_model_name = os.getenv("CPU_MODEL", "segmind/tiny-sd")
        self.profile = os.getenv("LOCAL_INFER_PROFILE", "fast").lower()

        # Limit threads so loading is more stable on low-memory systems
        import os as _threading_os
        torch_num_threads = min(4, _threading_os.cpu_count() or 4)
        torch.set_num_threads(torch_num_threads)
        logger.info("[CPU] Thread count capped to %d to reduce memory pressure", torch_num_threads)

        if self.device == "cuda":
            self.model_name = self.gpu_model_name
        else:
            # CPU: use lightweight model directly — never attempt large SDXL on low-RAM CPU
            self.model_name = self.cpu_model_name

        self.hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._lora_manager = get_lora_manager()

    @staticmethod
    def _is_valid_lora_dir(path: Path) -> bool:
        if not path or not path.exists() or not path.is_dir():
            return False
        config = path / "adapter_config.json"
        has_weights = (path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists()
        return config.exists() and has_weights

    @property
    def pipeline(self):
        if DiffusionService._pipeline is None:
            if DiffusionService._load_error:
                raise RuntimeError(f"Model is unavailable: {DiffusionService._load_error}")
            self._load_model()
        return DiffusionService._pipeline

    def get_status(self) -> dict:
        """Return model + LoRA status."""
        lora_status = self._lora_manager.get_status()
        return {
            "device": self.device,
            "profile": self.profile,
            "target_model": self.model_name,
            "requested_model": DiffusionService._requested_model_name,
            "loaded": DiffusionService._pipeline is not None,
            "loaded_model": DiffusionService._loaded_model_name,
            "load_error": DiffusionService._load_error,
            "lora": {
                "discovered": lora_status["discovered"],
                "loaded_adapters": lora_status["loaded_adapters"],
                "active_weights": lora_status["active_weights"],
                "outputs_dir": lora_status["outputs_dir"],
            },
        }

    def warmup(self, auto_load_loras: bool = None) -> dict:
        """
        Pre-warm the base diffusion pipeline.

        Args:
            auto_load_loras: if True, also pre-load all 3 LoRA adapters after the base model.
                             Default: reads from LORA_AUTO_LOAD env var (default: False for speed).
                             LoRAs are always loaded lazily on first generation anyway.
        """
        _ = self.pipeline  # ensure base model is loaded

        if auto_load_loras is None:
            auto_load_loras = os.getenv("LORA_AUTO_LOAD", "false").lower() in {"1", "true", "yes"}

        if auto_load_loras:
            self._auto_load_all_loras()

        return self.get_status()

    def _auto_load_all_loras(self):
        """Load all 3 trained LoRA adapters into the pipeline simultaneously."""
        available = self._lora_manager.available_types()
        if not available:
            logger.warning("No LoRA adapters found for auto-load")
            return

        logger.info("Auto-loading %d LoRA adapters: %s", len(available), available)
        pipeline = self.pipeline

        # Load adapters one by one using stack=True
        for lora_type in available:
            ok = self._lora_manager.load_adapter(
                lora_type,
                pipeline,
                scale=1.0,
                stack=True,
            )
            if ok:
                logger.info("  [OK] %s", lora_type)
            else:
                logger.warning("  [SKIP] %s (failed to load)", lora_type)

    def _try_load_pipeline(self, model_name: str, load_direct_to_device: bool = False):
        """Attempt to load a diffusion pipeline by name."""
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

        # Aggressively free memory before loading on CPU to reduce MemoryError
        if self.device == "cpu":
            gc.collect()

        base_kwargs = {}
        if self.hf_token:
            base_kwargs["token"] = self.hf_token
        if self.device == "cuda":
            base_kwargs["variant"] = "fp16"
            base_kwargs["torch_dtype"] = self.dtype
        else:
            # CPU: load fp32. Do NOT pass device="cpu" to from_pretrained —
            # it is ignored and raises a warning. Let automatic CPU placement work.
            base_kwargs["torch_dtype"] = torch.float32

        # Load with StableDiffusionPipeline (stable on Windows CPU).
        try:
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_name,
                use_safetensors=True,
                safety_checker=None,
                requires_safety_checker=False,
                **base_kwargs,
            )
            if self.device != "cuda":
                # Move to CPU explicitly; on CPU-only systems this just
                # confirms placement without extra memory copy.
                pipeline = pipeline.to(self.device)
        except (OSError, MemoryError) as e:
            DiffusionService._pipeline = None
            DiffusionService._loaded_model_name = None
            DiffusionService._load_error = f"{type(e).__name__}: {str(e)}"
            logger.exception("Model load failed")
            raise

        if self.device == "cuda":
            try:
                pipeline.enable_xformers_memory_efficient_attention()
            except Exception:
                logger.info("xformers not available")
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

        if self.device == "cpu":
            if hasattr(pipeline, "safety_checker"):
                pipeline.safety_checker = None
            if hasattr(pipeline, "requires_safety_checker"):
                pipeline.requires_safety_checker = False
        return pipeline

    def _load_model(self):
        """Load the diffusion model (pre-selected in __init__ based on device)."""
        DiffusionService._requested_model_name = self.model_name
        DiffusionService._fallback_used = False

        logger.info(
            "Loading model: %s on %s",
            self.model_name,
            self.device,
        )

        try:
            DiffusionService._pipeline = self._try_load_pipeline(self.model_name)
            DiffusionService._loaded_model_name = self.model_name
            DiffusionService._load_error = None
            logger.info("Model loaded successfully")
        except WindowsError as e:
            DiffusionService._pipeline = None
            DiffusionService._loaded_model_name = None
            DiffusionService._load_error = f"WindowsError (access violation): {str(e)}"
            logger.error("Model load failed due to Windows access violation: %s", e)
            raise
        except Exception as e:
            DiffusionService._pipeline = None
            DiffusionService._loaded_model_name = None
            DiffusionService._load_error = f"{type(e).__name__}: {str(e)}"
            logger.exception("Model load failed")
            raise

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
        use_default_lora: bool = False,
        lora_scale: float = 1.0,
        lora_stack: bool = False,
    ) -> List[Image.Image]:
        """
        Generate images from prompt with optional LoRA adapter.

        LoRA resolution order (first valid wins):
          1. lora_path (explicit directory path — overrides everything)
          2. lora_type (logo_2d | logo_3d | poster — uses trained adapters)
          3. None (base model, no LoRA)
        """
        # Enforce 8-pixel alignment (required by diffusion models)
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

        # CPU: cap resolution and steps for speed
        if self.device == "cpu":
            width = min(width, 512)
            height = min(height, 512)
            num_inference_steps = min(num_inference_steps, 4)
            guidance_scale = min(guidance_scale, 3.0)

        if seed is None:
            import random
            seed = random.randint(0, 2**31 - 1)

        generator = torch.Generator(device=self.device).manual_seed(int(seed))

        # ── LoRA decision & pipeline selection ─────────────────────────────────
        # 1. Resolve which LoRA key we want (if any)
        if lora_path:
            lora_key = "_custom_path"
        elif lora_type:
            lora_key = lora_type
        else:
            lora_key = None

        wants_lora = lora_key is not None

        # 2. Determine current pipeline model compatibility
        current_model = DiffusionService._loaded_model_name or ""
        is_sdxl = current_model.lower() in (
            "stabilityai/sdxl-turbo",
            "stabilityai/stable-diffusion-xl-base-1.0",
        )

        needs_sdxl = wants_lora and not is_sdxl

        # 3a. If SDXL needed but not loaded → reload pipeline
        if needs_sdxl:
            logger.info(
                "LoRA '%s' requires SDXL — reloading pipeline from '%s' -> '%s' (device=%s)",
                lora_key, current_model, self.gpu_model_name, self.device,
            )
            # Aggressively free memory before reloading on CPU
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            DiffusionService._pipeline = None
            DiffusionService._loaded_model_name = None
            self.model_name = self.gpu_model_name
            try:
                DiffusionService._pipeline = self._try_load_pipeline(self.model_name)
                DiffusionService._loaded_model_name = self.model_name
                logger.info("Pipeline reloaded with '%s'", self.model_name)
            except Exception as e:
                logger.error("SDXL load failed (%s) — falling back to original model without LoRA.", e)
                DiffusionService._pipeline = None
                DiffusionService._loaded_model_name = None
                gc.collect()
                DiffusionService._pipeline = self._try_load_pipeline(self.cpu_model_name)
                DiffusionService._loaded_model_name = self.cpu_model_name
                wants_lora = False  # MUST skip LoRA — base model incompatible

        # 3b. Reload self.pipeline with fresh reference
        pipeline = self.pipeline

        # 4. Validate adapter exists AND pipeline is SDXL-compatible before loading
        lora_loaded = False
        if wants_lora and lora_key:
            adapter_info = self._lora_manager._discovered.get(lora_key)
            if not adapter_info:
                logger.warning("LoRA key '%s' not in discovered adapters — skipping.", lora_key)
                wants_lora = False
            else:
                # Final SDXL check after pipeline reload
                model_after = DiffusionService._loaded_model_name or ""
                is_sdxl_now = model_after.lower() in (
                    "stabilityai/sdxl-turbo",
                    "stabilityai/stable-diffusion-xl-base-1.0",
                )
                if not is_sdxl_now:
                    logger.warning(
                        "Pipeline is '%s' (not SDXL) — cannot load LoRA '%s'. Skipping.",
                        model_after, lora_key,
                    )
                    wants_lora = False
                else:
                    if lora_key == "_custom_path":
                        lora_loaded = self._lora_manager.load_adapter(
                            lora_key, pipeline, scale=lora_scale, stack=lora_stack,
                        )
                    else:
                        lora_loaded = self._lora_manager.load_adapter(
                            lora_key, pipeline, scale=lora_scale, stack=lora_stack,
                        )
                    if lora_loaded:
                        logger.info("LoRA '%s' loaded (scale=%.2f, stack=%s)", lora_key, lora_scale, lora_stack)

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
            if lora_loaded:
                self._lora_manager.unload_all(pipeline)
                logger.info("LoRA adapters unloaded after generation")

        return images

    def load_custom_lora(self, lora_path: str, scale: float = 1.0):
        """Load LoRA adapter from a local folder path via LoRAManager."""
        self._lora_manager.load_adapter("_custom_path", self.pipeline, scale=scale, stack=False)
        logger.info("Loaded custom LoRA from %s", lora_path)

    def unload_lora(self):
        """Unload all LoRA adapters from pipeline."""
        self._lora_manager.unload_all(self.pipeline)
        logger.info("LoRA weights unloaded")
