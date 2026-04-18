"""
LoRA Manager - Auto-discovers and manages trained LoRA adapters.
Supports lazy-load, multi-adapter stacking, and fast switching.
No GPU/memory is used during discovery — only during actual loading.
"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class LoRAInfo:
    """Metadata for a discovered LoRA."""
    __slots__ = ("name", "lora_type", "path", "rank", "alpha", "base_model", "loaded", "adapter_name")

    def __init__(
        self,
        name: str,
        lora_path: Path,
        lora_type_key: str,
        rank: int = 8,
        alpha: int = 16,
        base_model: str = "stabilityai/sdxl-turbo",
    ):
        self.name = name
        self.lora_type = lora_type_key
        self.path = lora_path
        self.rank = rank
        self.alpha = alpha
        self.base_model = base_model
        self.loaded = False
        self.adapter_name = name  # already has "lora_" prefix in lora_type


class LoRAManager:
    """
    Manages all trained LoRA adapters with lazy-loading and multi-adapter support.

    Directory convention:
        outputs/{lora_type}/final/unet_lora/
            adapter_config.json
            adapter_model.safetensors

    Auto-discovery: scans outputs/ at __init__ time (no GPU needed).
    Lazy-load: adapters are loaded into the pipeline only when first used.
    Multi-adapter: supports loading multiple adapters simultaneously with weights.
    """

    def __init__(self):
        self._discovered: Dict[str, LoRAInfo] = {}
        self._loaded_adapters: List[str] = []
        self._active_weights: Dict[str, float] = {}
        self._base_dir = self._resolve_outputs_dir()
        self._discover_all()
        logger.info(
            "LoRAManager initialized: %d adapters found — %s",
            len(self._discovered),
            list(self._discovered.keys()),
        )

    # ------------------------------------------------------------------ #
    # Discovery                                                            #
    # ------------------------------------------------------------------ #

    def _resolve_outputs_dir(self) -> Path:
        env_override = os.getenv("LORA_OUTPUTS_DIR", "")
        if env_override:
            return Path(env_override)
        # lora_manager.py is at app/services/
        # parents[0]=services, parents[1]=app, parents[2]=project_root (AI_GEN/)
        here = Path(__file__).resolve().parents[2]
        return here / "outputs"

    def _discover_all(self):
        """Scan outputs/ for all valid LoRA adapters (no GPU, no loading)."""
        if not self._base_dir.exists():
            logger.warning("LoRA outputs dir not found: %s", self._base_dir)
            return

        lora_types = ["lora_logo_2d", "lora_logo_3d", "lora_poster"]

        for lora_type in lora_types:
            lora_path = self._base_dir / lora_type / "final" / "unet_lora"
            if self._is_valid_lora(lora_path):
                info = self._read_adapter_info(lora_path, lora_type)
                self._discovered[lora_type] = info
                logger.info("  Discovered: %s @ %s", lora_type, lora_path)
            else:
                logger.info("  Not found: %s", lora_path)

    @staticmethod
    def _is_valid_lora(path: Path) -> bool:
        if not path or not path.exists() or not path.is_dir():
            return False
        config = path / "adapter_config.json"
        has_weights = (path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists()
        return config.exists() and has_weights

    @staticmethod
    def _read_adapter_info(lora_path: Path, lora_type: str) -> LoRAInfo:
        cfg = {}
        config_path = lora_path / "adapter_config.json"
        try:
            with open(config_path, encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            pass

        rank = cfg.get("r", cfg.get("lora_r", 8))
        alpha = cfg.get("lora_alpha", 16)
        base_model = cfg.get("base_model_name_or_path", "stabilityai/sdxl-turbo")

        return LoRAInfo(
            name=lora_type,
            lora_path=lora_path,
            lora_type_key=lora_type,
            rank=rank,
            alpha=alpha,
            base_model=base_model,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def list_adapters(self) -> List[dict]:
        """Return list of all discovered adapters with their metadata."""
        return [
            {
                "type": info.lora_type,
                "path": str(info.path),
                "rank": info.rank,
                "alpha": info.alpha,
                "base_model": info.base_model,
                "loaded": info.loaded,
            }
            for info in self._discovered.values()
        ]

    def get_adapter_path(self, lora_type: str) -> Optional[str]:
        """Return the resolved path for a lora_type, or None if not found."""
        info = self._discovered.get(lora_type)
        return str(info.path) if info else None

    def available_types(self) -> List[str]:
        """Return all available LoRA type identifiers."""
        return list(self._discovered.keys())

    def load_adapter(
        self,
        lora_type: str,
        pipeline,
        scale: float = 1.0,
        stack: bool = False,
    ) -> bool:
        """
        Load a LoRA adapter into the pipeline.

        Args:
            lora_type: one of ["lora_logo_2d", "lora_logo_3d", "lora_poster"]
            pipeline: the diffusers pipeline instance
            scale: adapter weight scale (0.0–2.0)
            stack: if True, add to existing adapters; if False, replace all

        Returns:
            True if loaded successfully, False otherwise.
        """
        info = self._discovered.get(lora_type)
        if not info:
            logger.error("Unknown LoRA type: %s. Available: %s", lora_type, self.available_types())
            return False

        if not self._is_valid_lora(info.path):
            logger.error("LoRA path is invalid: %s", info.path)
            return False

        adapter_name = info.adapter_name

        if stack and self._loaded_adapters:
            new_adapters = list(self._loaded_adapters) + [adapter_name]
            new_weights = [self._active_weights.get(a, 1.0) for a in self._loaded_adapters] + [float(scale)]
            pipeline.load_lora_weights(str(info.path), adapter_name=adapter_name)
            pipeline.set_adapters(new_adapters, adapter_weights=new_weights)
            self._loaded_adapters = new_adapters
            self._active_weights[adapter_name] = float(scale)
        else:
            self._unload_all_from_pipeline(pipeline)
            pipeline.load_lora_weights(str(info.path), adapter_name=adapter_name)
            pipeline.set_adapters([adapter_name], adapter_weights=[float(scale)])
            self._loaded_adapters = [adapter_name]
            self._active_weights = {adapter_name: float(scale)}

        info.loaded = True
        logger.info(
            "Loaded LoRA '%s' (scale=%.2f) — stacked=%s — adapters: %s",
            lora_type, scale, stack, self._loaded_adapters,
        )
        return True

    def _unload_all_from_pipeline(self, pipeline):
        """Safely unload all LoRA adapters from pipeline."""
        try:
            if hasattr(pipeline, "unload_lora_weights"):
                pipeline.unload_lora_weights()
            elif hasattr(pipeline, "disable_lora"):
                pipeline.disable_lora()
        except Exception as e:
            logger.warning("LoRA unload warning: %s", e)
        finally:
            for info in self._discovered.values():
                info.loaded = False
            self._loaded_adapters = []
            self._active_weights = {}

    def unload_all(self, pipeline):
        """Unload all adapters from pipeline and reset state."""
        self._unload_all_from_pipeline(pipeline)
        logger.info("All LoRA adapters unloaded")

    def set_adapter_weights(self, pipeline, weights: Dict[str, float]):
        """
        Adjust the weight scale of currently-loaded adapters in-place.
        weights: {"lora_logo_2d": 0.8, "lora_poster": 0.5, ...}
        """
        active = [a for a in self._loaded_adapters if a in {info.adapter_name for info in self._discovered.values()}]
        if not active:
            return

        resolved_weights = [
            float(weights.get(adapter, self._active_weights.get(adapter, 1.0)))
            for adapter in active
        ]
        for adapter in active:
            self._active_weights[adapter] = weights.get(adapter, self._active_weights.get(adapter, 1.0))

        pipeline.set_adapters(active, adapter_weights=resolved_weights)
        logger.info("Updated adapter weights: %s", self._active_weights)

    def get_status(self) -> dict:
        """Return full status of all adapters."""
        return {
            "outputs_dir": str(self._base_dir),
            "discovered": self.list_adapters(),
            "loaded_adapters": list(self._loaded_adapters),
            "active_weights": dict(self._active_weights),
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_lora_manager: Optional[LoRAManager] = None


def get_lora_manager() -> LoRAManager:
    """Lazily create and return the global LoRAManager instance."""
    global _lora_manager
    if _lora_manager is None:
        _lora_manager = LoRAManager()
    return _lora_manager
