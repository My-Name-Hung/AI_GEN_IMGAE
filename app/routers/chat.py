"""
Chat Router - Conversational AI Interface
Provides chat + export APIs.
Chat uses cloud AI first and local fallback when cloud fails.
"""
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging

from app.services.neural_inference import get_neural_service, NeuralServiceError
from app.services.export_service import get_export_service, SUPPORTED_FORMATS

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    conversation_id: Optional[str] = Field(default=None, max_length=100)
    system_hint: Optional[str] = Field(default=None, max_length=500)


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    model: str
    timestamp: str


class ExportRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image")
    format: str = Field(
        ...,
        description=f"Output format. Supported: {', '.join(SUPPORTED_FORMATS.keys())}",
    )
    quality: int = Field(default=95, ge=1, le=100)
    max_width: Optional[int] = Field(default=None, ge=16, le=4096)
    max_height: Optional[int] = Field(default=None, ge=16, le=4096)


class BatchExportRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image")
    formats: List[str] = Field(..., min_length=1, max_length=8)
    quality: int = Field(default=90, ge=1, le=100)


class ExportResponse(BaseModel):
    success: bool
    data: Optional[str] = None
    format: str
    extension: str
    mime_type: str
    metadata: Dict[str, Any]
    error: Optional[str] = None


class BatchExportResponse(BaseModel):
    success: bool
    exports: Dict[str, Any]
    errors: Optional[Dict[str, str]] = None
    source_size: Dict[str, Any]


class ImageInfoResponse(BaseModel):
    width: int
    height: int
    mode: str
    format: Optional[str] = None


def _local_fallback_chat_response(message: str) -> str:
    """Very light local fallback when cloud AI is unavailable."""
    msg = message.strip().lower()
    if any(k in msg for k in ["tạo ảnh", "generate", "poster", "logo"]):
        return (
            "Cloud AI đang lỗi tạm thời nên mình chuyển sang local fallback. "
            "Bạn có thể mô tả rõ style/màu/chữ, mình sẽ tối ưu prompt để tạo ảnh local tốt hơn."
        )
    if any(k in msg for k in ["xin chào", "hello", "hi"]):
        return "Chào bạn! Cloud AI đang tạm lỗi nên mình phản hồi local fallback. Bạn cần hỗ trợ gì về thiết kế?"
    return (
        "Cloud AI đang tạm không khả dụng, mình đã chuyển sang local fallback cho chat. "
        "Bạn có thể hỏi tiếp, hoặc thử lại sau để nhận phản hồi đầy đủ từ cloud AI."
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a chat message to the assistant.

    Priority:
    1) Cloud chat API
    2) Local fallback response when cloud fails
    """
    service = get_neural_service()
    conversation_id = request.conversation_id or "default"

    try:
        response = service.chat(
            prompt=request.message,
            conversation_id=conversation_id,
        )
        return ChatResponse(
            response=response["response"],
            conversation_id=response["conversation_id"],
            model=response.get("model", "cloud_ai"),
            timestamp=response["timestamp"],
        )

    except (NeuralServiceError, RuntimeError, Exception) as e:
        logger.warning("Cloud chat failed, fallback local: %s", str(e))
        from datetime import datetime

        return ChatResponse(
            response=_local_fallback_chat_response(request.message),
            conversation_id=conversation_id,
            model="local-fallback",
            timestamp=datetime.now().isoformat(),
        )


@router.post("/chat/clear")
async def clear_chat(conversation_id: Optional[str] = None):
    """Clear conversation history."""
    try:
        service = get_neural_service()
        service.clear_history(conversation_id)
        return {"success": True, "message": "Đã xóa lịch sử hội thoại"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/capabilities")
async def get_capabilities():
    """Get neural service capabilities."""
    try:
        service = get_neural_service()
        return service.get_capabilities()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export", response_model=ExportResponse)
async def export_image(request: ExportRequest):
    """
    Export an image to specified format.

    Supported formats:
    - png, jpg, webp, bmp, tiff, ico
    - pdf (single page)
    - svg (vector conversion)
    """
    try:
        service = get_export_service()

        max_size = None
        if request.max_width or request.max_height:
            max_size = (
                request.max_width or 4096,
                request.max_height or 4096,
            )

        result = service.export_image(
            image_base64=request.image_base64,
            output_format=request.format,
            quality=request.quality,
            max_size=max_size,
        )

        return ExportResponse(
            success=result.get("success", False),
            data=result.get("data"),
            format=result["format"],
            extension=result["extension"],
            mime_type=result["mime_type"],
            metadata=result.get("metadata", {}),
            error=result.get("error"),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Export failed")
        raise HTTPException(status_code=500, detail=f"Lỗi xuất file: {str(e)}")


@router.post("/export/batch", response_model=BatchExportResponse)
async def batch_export(request: BatchExportRequest):
    """
    Export single image to multiple formats at once.

    Useful for generating all needed formats in one request.
    """
    try:
        service = get_export_service()

        result = service.export_batch(
            image_base64=request.image_base64,
            formats=request.formats,
            quality=request.quality,
        )

        return BatchExportResponse(**result)

    except Exception as e:
        logger.exception("Batch export failed")
        raise HTTPException(status_code=500, detail=f"Lỗi xuất hàng loạt: {str(e)}")


@router.get("/export/formats")
async def get_supported_formats():
    """Get list of all supported export formats."""
    return {
        "formats": [
            {
                "name": fmt,
                "extension": info["extension"],
                "mime": info["mime"],
                "lossy": fmt in ("jpg", "jpeg", "webp"),
                "supports_quality": fmt in ("jpg", "jpeg", "webp"),
                "supports_resize": True,
            }
            for fmt, info in SUPPORTED_FORMATS.items()
        ]
    }


@router.post("/image/info")
async def get_image_info(image_base64: str = Body(..., embed=True)):
    """Get image metadata without processing."""
    try:
        service = get_export_service()
        info = service.get_image_info(image_base64)
        if "error" in info:
            raise HTTPException(status_code=400, detail=info["error"])
        return ImageInfoResponse(**info)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export/thumbnail")
async def export_thumbnail(
    image_base64: str = Body(...),
    size: int = Body(default=128, ge=16, le=512),
):
    """Generate thumbnail preview of image."""
    try:
        service = get_export_service()
        thumbnail = service.export_thumbnail(
            image_base64=image_base64,
            size=(size, size),
        )
        return {
            "success": True,
            "data": thumbnail,
            "format": "png",
            "size": size,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
