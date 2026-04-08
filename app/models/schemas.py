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


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500, description="Text prompt for generation")
    negative_prompt: Optional[str] = Field(default="low quality, blurry, distorted", max_length=500)
    width: int = Field(default=1024, ge=256, le=1024)
    height: int = Field(default=1024, ge=256, le=1024)
    num_inference_steps: int = Field(default=4, ge=1, le=50)
    guidance_scale: float = Field(default=0.0, ge=0.0, le=20.0)
    seed: Optional[int] = Field(default=None)
    num_images: int = Field(default=1, ge=1, le=4)
    mode: GenerationMode = Field(default=GenerationMode.TURBO)
    enable_vectorization: bool = Field(default=False)
    layout_aware: bool = Field(default=False)
    lora_path: Optional[str] = Field(
        default=None,
        description="Path to LoRA weights directory (e.g. /lora_poster/final/unet_lora)"
    )
    use_default_lora: bool = Field(
        default=True,
        description="Use DEFAULT_LORA_PATH or common local LoRA path when lora_path is empty"
    )
    lora_scale: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="LoRA adapter weight scale"
    )
    save_outputs: bool = Field(
        default=False,
        description="If true, save generated PNG files under outputs/generated"
    )
    output_subdir: Optional[str] = Field(
        default=None,
        max_length=120,
        description="Optional sub-folder name under outputs/generated"
    )


class GenerationResponse(BaseModel):
    success: bool
    images: List[str] = Field(description="Base64 encoded images")
    svg_paths: Optional[List[str]] = None
    metadata: dict
    clip_scores: Optional[List[float]] = None


class VectorizeRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded PNG image")
    output_format: str = Field(default="svg", pattern="^(svg|png)$")
    color_quantization: int = Field(default=16, ge=2, le=256)
    simplify_tolerance: float = Field(default=1.0, ge=0.1, le=10.0)


class VectorizeResponse(BaseModel):
    success: bool
    svg_content: Optional[str] = None
    png_base64: Optional[str] = None
    layers_json: Optional[dict] = None
    metadata: dict


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


class LayoutAnalysisRequest(BaseModel):
    image_base64: str
    detect_text: bool = True
    detect_icons: bool = True
    suggest_layout: bool = True


class LayoutAnalysisResponse(BaseModel):
    success: bool
    layout: dict
    regions: List[dict]
    texts: List[dict]
    icons: List[dict]
