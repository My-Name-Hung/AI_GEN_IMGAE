"""
Unified Inference Service - Automatically selects the best available backend.
Priority: CUDA → CPU (PyTorch)

If neither GPU nor working CPU pipeline is available, returns a
"NoGPUError" that the API layer translates into a clear user message.
"""
import gc
import logging
import os
import random
from typing import List, Optional, Dict

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ─── Backend detection ─────────────────────────────────────────────────────────

def detect_backend() -> str:
    """
    Detect best available inference backend.
    Returns: "cuda" | "cpu"
    """
    # 1. Check NVIDIA CUDA
    try:
        result = __import__("subprocess").run(
            ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip().isdigit():
            gpu_count = int(result.stdout.strip())
            if gpu_count > 0 and torch.cuda.is_available():
                logger.info("Backend: CUDA (NVIDIA GPU detected)")
                return "cuda"
    except Exception:
        pass

    if torch.cuda.is_available():
        logger.info("Backend: CUDA (PyTorch CUDA available)")
        return "cuda"

    # 2. CPU — PyTorch on CPU is the safest fallback
    logger.info("Backend: CPU (PyTorch CPU)")
    return "cpu"


# ─── Unified Inference Service ──────────────────────────────────────────────────

class InferenceService:
    """
    Unified inference service with automatic backend selection.

    Backends:
      - cuda:      PyTorch + CUDA (fastest, requires NVIDIA GPU)
      - openvino:  OpenVINO + Intel GPU (good for integrated Intel GPUs)
      - cpu:       PyTorch on CPU (slowest, most compatible)

    Usage:
      service = InferenceService()
      images = service.generate("a modern logo", lora_type="lora_logo_2d")
    """

    def __init__(self, force_backend: Optional[str] = None):
        self.backend = force_backend or detect_backend()
        self._impl: Optional["InferenceBackend"] = None
        self._unavailable_reason: Optional[str] = None
        self._impl = self._init_backend()

    def _init_backend(self) -> "InferenceBackend":
        if self.backend == "cuda":
            return _CUDABackend()
        else:
            return _CPUBackend()

    def get_status(self) -> dict:
        extra = self._impl.get_status() if self._impl else {}
        return {
            "backend": self.backend,
            "impl": type(self._impl).__name__ if self._impl else None,
            "unavailable_reason": self._unavailable_reason,
            **extra,
        }

    def warmup(self) -> dict:
        if self._impl:
            return self._impl.warmup()
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
        Generate images using the best available backend.

        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt
            width, height: Resolution (multiple of 8)
            num_inference_steps: Denoising steps
            guidance_scale: CFG scale
            seed: Random seed for reproducibility
            num_images: Number of images to generate
            mode: "turbo" | "standard" | "quality"
            lora_type: LoRA adapter type
            lora_path: Custom LoRA path
            lora_scale: LoRA weight
            lora_stack: Stack LoRAs

        Returns:
            List of PIL Images
        """
        return self._impl.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            num_images=num_images,
            mode=mode,
            lora_type=lora_type,
            lora_path=lora_path,
            lora_scale=lora_scale,
            lora_stack=lora_stack,
        )

    def unload_lora(self):
        self._impl.unload_lora()


# ─── Backend interface ───────────────────────────────────────────────────────────

class InferenceBackend:
    """Abstract interface for inference backends."""

    backend_name: str = "base"

    def get_status(self) -> dict:
        return {"model": getattr(self, "_model_name", None)}

    def warmup(self) -> dict:
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
        raise NotImplementedError

    def unload_lora(self):
        pass


# ─── CUDA Backend ───────────────────────────────────────────────────────────────

class _CUDABackend(InferenceBackend):
    """PyTorch + CUDA backend (NVIDIA GPU). Fastest, full LoRA support."""

    backend_name = "cuda"

    def __init__(self):
        from app.services.diffusion import DiffusionService
        self._svc = DiffusionService()
        self._model_name = self._svc.gpu_model_name
        logger.info("[CUDA Backend] Using GPU model: %s", self._svc.gpu_model_name)

    def get_status(self) -> dict:
        return {
            "model": self._model_name,
            "device": "cuda",
            "pipeline": type(self._svc.pipeline).__name__,
        }

    def warmup(self) -> dict:
        self._svc.warmup()
        return self.get_status()

    def generate(self, **kwargs) -> List[Image.Image]:
        return self._svc.generate(**kwargs)

    def unload_lora(self):
        self._svc.unload_lora()


# ─── OpenVINO Backend ──────────────────────────────────────────────────────────

class _OpenVINOBackend(InferenceBackend):
    """
    OpenVINO GPU backend (Intel Integrated/Discrete GPU).
    Uses optimum-intel OVStableDiffusionPipeline.
    LoRA support via set_adapters() if supported.
    """

    backend_name = "openvino"

    def __init__(self):
        self._model_name = os.getenv("GPU_MODEL", "stabilityai/sdxl-turbo")
        self._device = "GPU"
        self._pipeline = None
        self._lora_manager = None
        self._lora_loaded = False
        self._cpu_fallback = None  # lazy-loaded CPU backend for memory failures

        try:
            from app.services.lora_manager import get_lora_manager
            self._lora_manager = get_lora_manager()
        except Exception as e:
            logger.warning("LoRAManager not available: %s", e)

        logger.info("[OpenVINO Backend] Using model: %s on GPU", self._model_name)

    def _get_pipeline(self):
        # If CPU fallback was triggered (OOM during load), delegate to CPU backend
        if self._cpu_fallback is not None:
            return self._cpu_fallback._svc.pipeline
        if self._pipeline is None:
            self._load_pipeline()
        return self._pipeline

    def _load_pipeline(self):
        from optimum.intel.openvino import OVStableDiffusionPipeline
        import openvino as ov

        hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")

        # Configure OpenVINO for low-memory GPU usage
        # These settings reduce memory footprint on integrated Intel GPUs
        core = ov.Core()
        try:
            core.set_property("GPU", {
                "PERFORMANCE_HINT": "LATENCY",
                "NUM_STREAMS": 1,
                "INFERENCE_PRECISION_HINT": "f16",
                "ENABLE_MMAP": "YES",
            })
            logger.info("[OpenVINO] GPU memory optimizations applied: LATENCY hint, 1 stream, FP16")
        except Exception as opt_err:
            logger.warning("[OpenVINO] Could not apply GPU memory optimizations: %s", opt_err)

        load_kwargs = {"device": self._device}
        if hf_token:
            load_kwargs["token"] = hf_token

        logger.info("[OpenVINO] Loading pipeline: %s on %s", self._model_name, self._device)
        try:
            self._pipeline = OVStableDiffusionPipeline.from_pretrained(
                self._model_name,
                export=False,
                **load_kwargs,
            )
            logger.info("[OpenVINO] Pipeline loaded successfully on %s", self._device)
        except Exception as e:
            err_msg = str(e).lower()
            if "paging file" in err_msg or "1455" in err_msg or "out of memory" in err_msg or "oom" in err_msg:
                logger.error(
                    "[OpenVINO] Pipeline load failed due to insufficient memory (os error 1455). "
                    "Auto-fallback to CPU backend for this session."
                )
                self._cpu_fallback = _CPUBackend()
                self._pipeline = None  # mark as unavailable
            else:
                logger.error("[OpenVINO] Pipeline load failed: %s", e)
                raise

    def get_status(self) -> dict:
        lora_status = {}
        if self._lora_manager:
            try:
                lora_status = self._lora_manager.get_status()
            except Exception:
                lora_status = {}

        return {
            "model": self._model_name,
            "device": self._device,
            "backend": "openvino",
            "lora_loaded": self._lora_loaded,
            "lora": lora_status,
        }

    def warmup(self) -> dict:
        _ = self._get_pipeline()
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
        # Enforce 8-pixel alignment
        width = max(256, (width // 8) * 8)
        height = max(256, (height // 8) * 8)

        # Mode overrides
        if mode == "turbo":
            num_inference_steps = min(num_inference_steps, 4)
            guidance_scale = 0.0
        elif mode == "standard":
            num_inference_steps = min(num_inference_steps, 8)
            guidance_scale = max(guidance_scale, 3.5)
        elif mode == "quality":
            num_inference_steps = max(num_inference_steps, 20)
            guidance_scale = max(guidance_scale, 7.5)

        # OpenVINO CPU: cap resolution
        if self._device == "CPU":
            width = min(width, 512)
            height = min(height, 512)
            num_inference_steps = min(num_inference_steps, 4)

        # Seed
        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        # If pipeline failed to load due to OOM, delegate to CPU backend
        if self._cpu_fallback is not None:
            logger.warning(
                "[OpenVINO] GPU pipeline unavailable (OOM). Delegating to CPU backend."
            )
            return self._cpu_fallback.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                num_images=num_images,
                mode=mode,
                lora_type=lora_type,
                lora_path=lora_path,
                lora_scale=lora_scale,
                lora_stack=lora_stack,
            )

        pipeline = self._get_pipeline()

        # LoRA loading via set_adapters (optimum-intel supports this)
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
                            "[OpenVINO] LoRA '%s' loaded (scale=%.2f)",
                            lora_key, lora_scale,
                        )
                        self._lora_loaded = True
                except Exception as e:
                    logger.warning("[OpenVINO] LoRA load failed: %s", e)
                    lora_loaded = False

        try:
            images = []
            for i in range(num_images):
                logger.info(
                    "[OpenVINO] Generating image %d/%d (seed=%d, steps=%d, device=%s)",
                    i + 1, num_images, seed + i, num_inference_steps, self._device,
                )
                try:
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
                except Exception as gen_err:
                    err_lower = str(gen_err).lower()
                    # Catch OOM during generation (not just pipeline loading)
                    if any(kw in err_lower for kw in ["out of memory", "oom", "paging", "1455", "cudamalloc"]):
                        logger.warning(
                            "[OpenVINO] OOM during generation at %dx%d — retrying with 512x512 on CPU backend.",
                            width, height,
                        )
                        return self._cpu_fallback.generate(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            width=512,
                            height=512,
                            num_inference_steps=min(num_inference_steps, 4),
                            guidance_scale=guidance_scale,
                            seed=seed,
                            num_images=num_images,
                            mode=mode,
                            lora_type=lora_type,
                            lora_path=lora_path,
                            lora_scale=lora_scale,
                            lora_stack=lora_stack,
                        )
                    raise
        finally:
            if lora_loaded and self._lora_manager:
                try:
                    self._lora_manager.unload_all(pipeline)
                    self._lora_loaded = False
                except Exception:
                    pass

        return images

    def unload_lora(self):
        if self._lora_manager and self._lora_loaded:
            try:
                self._lora_manager.unload_all(self._pipeline)
                self._lora_loaded = False
            except Exception:
                pass


# ─── CPU Backend ───────────────────────────────────────────────────────────────

class _CPUBackend(InferenceBackend):
    """
    PyTorch CPU backend — now uses DiffusionService directly.
    This enables full LoRA support on CPU by sharing the same
    DiffusionService pipeline that has LoRA integration built in.
    """

    # Fallback chain: tiny-SD first (fastest/lightest) > SD 1.5 Turbo > SDXL Turbo.
    # SDXL is only attempted last since it needs 6-7 GB RAM on CPU.
    CPU_MODEL_FALLBACK_CHAIN = [
        "segmind/tiny-sd",
        "stabilityai/sd-turbo",
        "stabilityai/sdxl-turbo",
    ]

    def __init__(self):
        # Force CPU device before DiffusionService reads GPU availability
        original_force = os.environ.get("FORCE_DEVICE")
        os.environ["FORCE_DEVICE"] = "cpu"

        # Limit threads to reduce RAM footprint during model loading
        import os as _threading_os
        torch_num_threads = min(4, _threading_os.cpu_count() or 4)
        torch.set_num_threads(torch_num_threads)

        try:
            from app.services.diffusion import DiffusionService

            loaded_model = None
            load_error = None

            for model_name in _CPUBackend.CPU_MODEL_FALLBACK_CHAIN:
                try:
                    os.environ["CPU_MODEL"] = model_name
                    # Reset singleton so it reinitializes with new model
                    DiffusionService._pipeline = None
                    DiffusionService._load_error = None
                    DiffusionService._loaded_model_name = None
                    DiffusionService._instance = None

                    svc = DiffusionService()
                    # Verify it actually loaded on CPU
                    if svc.device != "cpu":
                        raise RuntimeError(
                            f"DiffusionService landed on {svc.device} instead of cpu"
                        )
                    self._svc = svc
                    self._model_name = svc.model_name
                    loaded_model = model_name
                    logger.info(
                        "[CPU Backend] Loaded %s (device=cpu, threads=%d)",
                        model_name, torch_num_threads,
                    )
                    break
                except (OSError, MemoryError) as e:
                    load_error = f"{model_name}: {e}"
                    logger.warning(
                        "[CPU Backend] Model '%s' failed (%s) — trying next after GC...",
                        model_name, e,
                    )
                    DiffusionService._pipeline = None
                    DiffusionService._load_error = None
                    DiffusionService._loaded_model_name = None
                    DiffusionService._instance = None
                    # Free memory before next attempt
                    gc.collect()
                except Exception as e:
                    load_error = f"{model_name}: {e}"
                    logger.warning(
                        "[CPU Backend] Model '%s' failed (%s) — trying next...",
                        model_name, e,
                    )
                    DiffusionService._pipeline = None
                    DiffusionService._load_error = None
                    DiffusionService._loaded_model_name = None
                    DiffusionService._instance = None

            if loaded_model is None:
                DiffusionService._load_error = (
                    f"No working CPU model found. Tried: {self.CPU_MODEL_FALLBACK_CHAIN}. "
                    f"Last error: {load_error}. "
                    f"Recommendation: close other apps to free RAM, "
                    f"or set CPU_MODEL=segmind/tiny-sd in .env"
                )
                self._svc = None
                self._model_name = "unavailable"
                logger.error("[CPU Backend] %s", DiffusionService._load_error)

        finally:
            if original_force is not None:
                os.environ["FORCE_DEVICE"] = original_force
            elif "FORCE_DEVICE" in os.environ:
                del os.environ["FORCE_DEVICE"]
            # CPU_MODEL is intentionally left set so DiffusionService uses it

    backend_name = "cpu"

    def get_status(self) -> dict:
        return {
            "model": self._model_name,
            "device": "cpu",
            "lora_support": True,
            "note": "DiffusionService CPU (LoRA enabled)",
        }

    def warmup(self) -> dict:
        if self._svc is not None:
            self._svc.warmup()
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
        if self._svc is None:
            raise RuntimeError(
                f"CPU inference unavailable: {DiffusionService._load_error or 'unknown error'}"
            )

        # Pass all params (including LoRA) directly to DiffusionService
        # — no subprocess wrapper, no model= parameter
        return self._svc.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            num_images=num_images,
            mode=mode,
            lora_type=lora_type,
            lora_path=lora_path,
            lora_scale=lora_scale,
            lora_stack=lora_stack,
        )

    def unload_lora(self):
        pass


# ─── Module-level singleton ───────────────────────────────────────────────────

_inference_service: Optional[InferenceService] = None


def get_inference_service(force_backend: Optional[str] = None) -> InferenceService:
    global _inference_service
    if _inference_service is None:
        requested = force_backend or os.getenv("INFERENCE_BACKEND")
        detected = detect_backend()

        # Prefer CUDA if available — ignore INFERENCE_BACKEND if it says "cpu"
        # but the system has a GPU. This prevents accidental CPU-only override
        # when GPU is available (e.g. Kaggle, Colab, local GPU machines).
        if detected == "cuda" and requested == "cpu":
            logger.warning(
                "INFERENCE_BACKEND=cpu is set but CUDA GPU is available. "
                "Auto-overriding to CUDA for better performance."
            )
            backend = "cuda"
        else:
            backend = requested or detected

        # If OpenVINO is requested but optimum-intel is missing, fall back to CPU
        if backend == "openvino":
            try:
                import optimum.intel.openvino  # noqa: F401
            except ImportError:
                logger.warning(
                    "INFERENCE_BACKEND=openvino is set but optimum-intel is not installed. "
                    "Falling back to CPU. Run: pip install optimum[openvino]"
                )
                backend = "cpu"

        _inference_service = InferenceService(force_backend=backend)
        logger.info(
            "InferenceService initialized: backend=%s, device=%s",
            _inference_service.backend,
            type(_inference_service._impl).__name__,
        )
    return _inference_service
