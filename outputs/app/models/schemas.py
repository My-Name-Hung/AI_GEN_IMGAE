"""
Pydantic Models for AI Designer System
Request/Response schemas
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
from datetime import datetime


class GenerationMode(str, Enum):
    TURBO = "turbo"
    STANDARD = "standard"
    QUALITY = "quality"


class LoraType(str, Enum):
    LOGO_2D = "lora_logo_2d"
    LOGO_3D = "lora_logo_3d"
    POSTER = "lora_poster"
    BASE = "base"  # no LoRA, use base model


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000, description="Text prompt for generation")
    negative_prompt: Optional[str] = Field(default="low quality, blurry, distorted, deformed", max_length=500)
    width: int = Field(default=1024, ge=256, le=1024)
    height: int = Field(default=1024, ge=256, le=1024)
    num_inference_steps: int = Field(default=4, ge=1, le=50)
    guidance_scale: float = Field(default=0.0, ge=0.0, le=20.0)
    seed: Optional[int] = Field(default=None)
    num_images: int = Field(default=1, ge=1, le=4)
    mode: GenerationMode = Field(default=GenerationMode.TURBO)

    # LoRA adapter selection (new — recommended over lora_path)
    lora_type: Optional[LoraType] = Field(
        default=None,
        description="LoRA adapter type: 'lora_logo_2d' | 'lora_logo_3d' | 'lora_poster' | 'base'",
    )
    lora_scale: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="LoRA adapter weight scale (0.0–2.0)",
    )
    lora_stack: bool = Field(
        default=False,
        description="If true, stack lora_type with existing adapters instead of replacing",
    )

    # Legacy path-based override (takes precedence if non-null)
    lora_path: Optional[str] = Field(
        default=None,
        description="Path to LoRA weights directory (overrides lora_type if set)",
    )
    use_default_lora: bool = Field(
        default=False,
        description="Use default LoRA path from env when lora_path and lora_type are both empty",
    )

    enable_vectorization: bool = Field(default=False, description="Export SVG vector alongside PNG")
    layout_aware: bool = Field(default=False, description="Run CLIP quality scoring on generated images")
    save_outputs: bool = Field(
        default=True,
        description="If true, save generated PNG files under outputs/generated",
    )
    output_subdir: Optional[str] = Field(
        default=None,
        max_length=120,
        description="Optional sub-folder name under outputs/generated",
    )


class GenerationResponse(BaseModel):
    success: bool
    images: List[str] = Field(description="Base64 encoded images")
    svg_paths: Optional[List[str]] = None
    metadata: dict
    clip_scores: Optional[List[float]] = None
    analysis: Optional[dict] = Field(default=None, description="Smart analysis breakdown from prompt analysis")


class VectorizeRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded PNG/JPG image")
    color_quantization: int = Field(
        default=16, ge=2, le=256,
        description="Number of color layers in the output SVG (2–256)"
    )
    simplify_tolerance: float = Field(
        default=1.5, ge=0.1, le=10.0,
        description="RDP simplification tolerance — lower = more detail, higher = cleaner paths"
    )
    bezier_smoothness: float = Field(
        default=0.25, ge=0.0, le=1.0,
        description="Bezier smoothing factor — 0 = sharp corners, 1 = very smooth curves"
    )
    output_dir: Optional[str] = Field(
        default=None,
        description="Optional directory to save output.svg and output_layers.json"
    )
    filename_base: Optional[str] = Field(
        default="vectorized",
        description="Base filename for saved output files"
    )


class VectorizeResponse(BaseModel):
    success: bool
    svg_content: Optional[str] = Field(default=None, description="Full SVG markup string")
    png_base64: Optional[str] = Field(default=None, description="Quantized PNG reference (base64)")
    layers_json: Optional[dict] = Field(default=None, description="Layer metadata JSON")
    metadata: dict = Field(default_factory=dict)


class TrainRequest(BaseModel):
    model_config = {'protected_namespaces': ()}

    dataset_path: str = Field(..., description="Path to training dataset")
    base_model: str = Field(default="stabilityai/sdxl-turbo", description="Base model name")
    output_dir: str = Field(default="./outputs")
    num_train_epochs: int = Field(default=10, ge=1, le=100)
    per_device_train_batch_size: int = Field(default=2, ge=1, le=16)
    learning_rate: float = Field(default=1e-4, gt=0)
    gradient_accumulation_steps: int = Field(default=4, ge=1, le=32)
    rank: int = Field(default=16, ge=4, le=128, description="LoRA rank")
    alpha: int = Field(default=16, ge=4, le=128)
    resolution: int = Field(default=512, ge=256, le=1024)
    validation_prompt: Optional[str] = None
    save_steps: int = Field(default=100, ge=10)
    allow_cpu: bool = Field(default=False, description="Allow CPU training (very slow)")


class AutoTrainRequest(BaseModel):
    """One-shot training: auto-detect dataset + run full pipeline.

    Passes raw_dataset_path so paired_poster_import is called internally.
    All training params are optional — smart defaults applied per field.
    """
    model_config = {'protected_namespaces': ()}

    raw_dataset_path: str = Field(
        ...,
        description="Path to raw images (e.g. ./dataset with POSTER (n).png + .txt)"
    )
    output_dir: str = Field(default="./outputs", description="Where to save checkpoints")
    style_prefix: str = Field(
        default="poster_style,",
        description="Caption prefix to help LoRA activate on specific prompts"
    )
    merge_short_title: bool = Field(
        default=False,
        description="Also read POSTER (n)(1).txt short title and append to caption"
    )
    base_model: str = Field(default="stabilityai/sdxl-turbo")
    rank: int = Field(default=8, ge=4, le=128)
    alpha: int = Field(default=16, ge=4, le=128)
    learning_rate: float = Field(default=1e-4, gt=0)
    num_train_epochs: int = Field(default=10, ge=1, le=100)
    per_device_train_batch_size: int = Field(default=1, ge=1, le=16)
    gradient_accumulation_steps: int = Field(default=8, ge=1, le=32)
    resolution: int = Field(default=512, ge=256, le=1024)
    save_steps: int = Field(default=100, ge=10)
    validation_prompt: str = Field(
        default="poster_style, modern movie poster, cinematic lighting, high detail"
    )
    allow_cpu: bool = Field(default=False)


class TrainResponse(BaseModel):
    success: bool
    job_id: str
    status: str
    message: str
    checkpoint_path: Optional[str] = None


class AutoTrainResponse(BaseModel):
    success: bool
    job_id: str
    status: str
    message: str
    processed_images: int = 0
    lora_output_path: Optional[str] = None
    final_checkpoint: Optional[str] = None


class LayoutAnalyzeRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded PNG image")
    detect_text: bool = Field(default=True, description="Enable text region detection")
    detect_icons: bool = Field(default=True, description="Enable icon/logo region detection")
    suggest_layout: bool = Field(default=True, description="Generate composition guides")
    composition_mode: str = Field(
        default="thirds",
        description="Composition mode for suggest_composition: centered | thirds | diagonal | split",
    )
    output_dir: Optional[str] = Field(
        default=None,
        description="Optional directory to save layout_schema.json",
    )


class LayoutAnalyzeResponse(BaseModel):
    success: bool
    layout: dict = Field(default_factory=dict)
    regions: List[dict] = Field(default_factory=list)
    texts: List[dict] = Field(default_factory=list)
    icons: List[dict] = Field(default_factory=list)
    background_zones: List[dict] = Field(default_factory=list)
    contrast_zones: List[dict] = Field(default_factory=list)
    image_size: dict = Field(default_factory=dict)
    summary: str = ""
    saved_schema: Optional[str] = None
