"""
Image Export Service - Multi-format Image Processing
Handles conversion between various image formats with quality optimization.
"""
import io
import base64
import logging
from typing import Optional, Tuple, Dict, Any
from PIL import Image

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {
    "png": {"extension": ".png", "mime": "image/png"},
    "jpg": {"extension": ".jpg", "mime": "image/jpeg"},
    "jpeg": {"extension": ".jpeg", "mime": "image/jpeg"},
    "webp": {"extension": ".webp", "mime": "image/webp"},
    "bmp": {"extension": ".bmp", "mime": "image/bmp"},
    "tiff": {"extension": ".tiff", "mime": "image/tiff"},
    "ico": {"extension": ".ico", "mime": "image/x-icon"},
    "pdf": {"extension": ".pdf", "mime": "application/pdf"},
    "svg": {"extension": ".svg", "mime": "image/svg+xml"},
}


class ImageExportService:
    """
    Handles image format conversion and export.
    Supports PNG, JPG, WEBP, BMP, TIFF, ICO, PDF, SVG.
    """

    @staticmethod
    def _decode_base64_image(encoded_data: str) -> Image.Image:
        """Decode base64 string to PIL Image."""
        try:
            image_data = base64.b64decode(encoded_data)
            return Image.open(io.BytesIO(image_data))
        except Exception as e:
            raise ValueError(f"Không thể giải mã ảnh: {str(e)}")

    @staticmethod
    def _encode_image(img: Image.Image, format: str, **kwargs) -> str:
        """Encode PIL Image to base64 string."""
        buffer = io.BytesIO()
        save_kwargs = {}

        if format.upper() in ("JPEG", "JPG"):
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            save_kwargs["quality"] = kwargs.get("quality", 95)
            save_kwargs["optimize"] = kwargs.get("optimize", True)
        elif format.upper() == "WEBP":
            save_kwargs["quality"] = kwargs.get("quality", 90)
            save_kwargs["method"] = 6
        elif format.upper() == "PNG":
            save_kwargs["optimize"] = kwargs.get("optimize", True)

        img.save(buffer, format=format.upper(), **save_kwargs)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def get_image_info(encoded_data: str) -> Dict[str, Any]:
        """Extract image metadata without full decode."""
        try:
            image_data = base64.b64decode(encoded_data[:100] + "==")
            img = Image.open(io.BytesIO(image_data))
            return {
                "width": img.width,
                "height": img.height,
                "mode": img.mode,
                "format": img.format,
            }
        except Exception:
            return {"error": "Cannot read image info"}

    @classmethod
    def export_image(
        cls,
        image_base64: str,
        output_format: str,
        quality: int = 95,
        max_size: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Export image to specified format.

        Args:
            image_base64: Base64 encoded image data
            output_format: Target format (png, jpg, webp, bmp, tiff, ico, pdf, svg)
            quality: JPEG/WebP quality (1-100)
            max_size: Optional (width, height) to resize

        Returns:
            Dict with base64 result, metadata, and format info
        """
        format_lower = output_format.lower()

        if format_lower not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Định dạng không được hỗ trợ: {output_format}. "
                f"Hỗ trợ: {', '.join(SUPPORTED_FORMATS.keys())}"
            )

        img = cls._decode_base64_image(image_base64)
        original_size = img.size

        if max_size:
            img.thumbnail(max_size, Image.Resampling.LANCZOS)

        if format_lower == "svg":
            return cls._convert_to_vector(encoded_data=image_base64, **kwargs)

        exported = cls._encode_image(img, format_lower, quality=quality)

        return {
            "success": True,
            "data": exported,
            "format": format_lower,
            "extension": SUPPORTED_FORMATS[format_lower]["extension"],
            "mime_type": SUPPORTED_FORMATS[format_lower]["mime"],
            "metadata": {
                "original_size": original_size,
                "exported_size": img.size,
                "original_mode": img.mode,
                "quality": quality,
            },
        }

    @classmethod
    def _convert_to_vector(
        cls,
        encoded_data: Optional[str] = None,
        image: Optional[Image.Image] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Convert raster image to SVG (placeholder - uses vectorizer service)."""
        try:
            from app.services.vectorizer import VectorizerService

            if encoded_data:
                img = cls._decode_base64_image(encoded_data)
            elif image:
                img = image
            else:
                raise ValueError("No image data provided")

            vectorizer = VectorizerService()
            result = vectorizer.vectorize(img)

            return {
                "success": True,
                "data": result.get("svg_content", ""),
                "format": "svg",
                "extension": ".svg",
                "mime_type": "image/svg+xml",
                "metadata": {
                    "original_size": img.size,
                    "vector_paths": result.get("num_paths", 0),
                    "colors": result.get("num_colors", 0),
                },
            }
        except Exception as e:
            logger.warning(f"SVG conversion fallback: {e}")
            return {
                "success": False,
                "error": f"Không thể chuyển sang SVG: {str(e)}",
                "format": "svg",
                "data": None,
            }

    @classmethod
    def export_thumbnail(
        cls,
        image_base64: str,
        size: Tuple[int, int] = (128, 128),
    ) -> str:
        """Export as thumbnail for preview."""
        img = cls._decode_base64_image(image_base64)
        img.thumbnail(size, Image.Resampling.LANCZOS)
        return cls._encode_image(img, "PNG")

    @classmethod
    def export_batch(
        cls,
        image_base64: str,
        formats: list,
        quality: int = 90,
    ) -> Dict[str, Any]:
        """
        Export single image to multiple formats at once.

        Args:
            image_base64: Source image
            formats: List of target formats
            quality: Quality for lossy formats

        Returns:
            Dict mapping format name to export result
        """
        results = {}
        errors = {}

        for fmt in formats:
            try:
                result = cls.export_image(
                    image_base64, fmt, quality=quality
                )
                results[fmt] = {
                    "success": True,
                    "data": result["data"],
                }
            except Exception as e:
                results[fmt] = {"success": False, "error": str(e)}
                errors[fmt] = str(e)

        return {
            "success": len(errors) == 0,
            "exports": results,
            "errors": errors if errors else None,
            "source_size": cls.get_image_info(image_base64),
        }


def get_export_service() -> ImageExportService:
    """Get export service instance."""
    return ImageExportService()
