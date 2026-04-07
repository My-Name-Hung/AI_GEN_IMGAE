"""
Backend Utils - Helper functions
"""
import base64
import io
import hashlib
from pathlib import Path
from typing import Optional
from PIL import Image
import torch
import numpy as np


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def base64_to_image(base64_str: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data))


def image_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """Convert PIL Image to bytes"""
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    return buffered.getvalue()


def bytes_to_image(image_bytes: bytes) -> Image.Image:
    """Convert bytes to PIL Image"""
    return Image.open(io.BytesIO(image_bytes))


def generate_seed() -> int:
    """Generate random seed"""
    return int(np.random.randint(0, 2**32))


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def resize_image(
    image: Image.Image,
    target_size: tuple,
    maintain_aspect: bool = True
) -> Image.Image:
    """Resize image with optional aspect ratio maintenance"""
    if maintain_aspect:
        image.thumbnail(target_size, Image.Resampling.LANCZOS)
        return image
    else:
        return image.resize(target_size, Image.Resampling.LANCZOS)


def get_device_info() -> dict:
    """Get information about available compute device"""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_total"] = torch.cuda.get_device_properties(0).total_memory
        info["cuda_memory_allocated"] = torch.cuda.memory_allocated(0)
        info["cuda_memory_cached"] = torch.cuda.memory_reserved(0)
    
    return info


def clear_cache():
    """Clear GPU cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
