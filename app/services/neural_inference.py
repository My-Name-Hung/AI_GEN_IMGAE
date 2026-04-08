"""
Neural Inference Service - Cloud AI processing
Uses cloud API for chat and image generation, with local fallback handled by routers.
"""
import os
import json
import logging
import urllib.request
import urllib.error
from typing import List, Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Cloud API configuration
_CLUSTER_CONFIG = {
    "primary": "https://generativelanguage.googleapis.com",
    "timeout": 90,
    "max_retries": 2,
}


def _get_api_credentials() -> Optional[str]:
    """Get API key from environment with multiple fallback names."""
    return (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("NEURAL_NET_API_KEY")
        or os.getenv("LOCAL_MODEL_TOKEN")
        or os.getenv("INFERENCE_BACKEND_KEY")
        or os.getenv("AI_PROCESSING_TOKEN")
    )


def _get_text_model_id() -> str:
    """Get configured cloud text model."""
    return os.getenv(
        "GEMINI_TEXT_MODEL",
        os.getenv("NEURAL_MODEL_ID", os.getenv("INFERENCE_MODEL", "gemini-2.5-flash-lite")),
    )


def _get_image_model_id() -> str:
    """Get configured cloud image model."""
    return os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-lite")


class NeuralInferenceService:
    """
    Cloud-first inference service.
    - chat(): text responses
    - generate_images(): image generation (base64 png/jpeg from cloud API)
    """

    _instance = None
    _session_history: List[Dict[str, Any]] = []
    _max_history = 20

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._api_key = _get_api_credentials()
        self._text_model_id = _get_text_model_id()
        self._image_model_id = _get_image_model_id()
        self._system_context = self._load_system_context()
        logger.info(
            "Neural service initialized with text model=%s image model=%s",
            self._text_model_id,
            self._image_model_id,
        )

    def _load_system_context(self) -> str:
        return """Bạn là một AI Assistant chuyên nghiệp với khả năng:
1. Trả lời câu hỏi về thiết kế đồ họa, logo, poster
2. Mô tả chi tiết ý tưởng thiết kế khi được yêu cầu
3. Hỗ trợ brainstorm ý tưởng sáng tạo
4. Giải thích các khái niệm về nghệ thuật, màu sắc, typography
5. Khi user yêu cầu tạo/xuất file ảnh, trả lời về format phù hợp

NGUYÊN TẮC:
- Trả lời ngắn gọn, thân thiện, chuyên nghiệp
- Dùng tiếng Việt, có thể kết hợp tiếng Anh cho thuật ngữ
- Khi không biết, nói thẳng và đề xuất hướng đi
- Không bịa đặt thông tin"""

    def _build_chat_payload(self, prompt: str, history: List[Dict]) -> Dict:
        contents = []
        for msg in history[-self._max_history:]:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        return {
            "contents": contents,
            "generationConfig": {
                "temperature": 0.7,
                "topP": 0.95,
                "topK": 40,
                "maxOutputTokens": 4096,
            },
            "systemInstruction": {"parts": [{"text": self._system_context}]},
        }

    def _build_image_payload(self, prompt: str, num_images: int = 1) -> Dict:
        # Gemini image preview models typically support image output through responseModalities.
        # Keep config compact and robust across model revisions.
        return {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                "candidateCount": max(1, min(int(num_images), 4)),
            },
        }

    def _call_cloud_api(self, payload: Dict, model_id: str) -> Dict:
        if not self._api_key:
            raise RuntimeError(
                "Cloud API key not configured. Set GEMINI_API_KEY (or GOOGLE_API_KEY/NEURAL_NET_API_KEY)."
            )

        base_url = _CLUSTER_CONFIG["primary"]
        url = f"{base_url}/v1beta/models/{model_id}:generateContent?key={self._api_key}"

        request_body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=request_body,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "AIDesignerBackend/1.0",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=_CLUSTER_CONFIG["timeout"]) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else "{}"
            try:
                error_data = json.loads(error_body)
                error_msg = self._map_error(e.code, error_data)
            except Exception:
                error_msg = self._map_error(e.code, {"raw": error_body})
            raise NeuralServiceError(error_msg, e.code)
        except urllib.error.URLError as e:
            raise NeuralServiceError(f"Không kết nối được cloud API: {e.reason}", -1)
        except Exception as e:
            raise NeuralServiceError(f"Cloud processing error: {str(e)}", -1)

    def _map_error(self, status_code: int, error_data: Dict) -> str:
        error_str = json.dumps(error_data).lower()

        if status_code == 429 or "quota" in error_str or "rate" in error_str:
            return "Cloud AI đang quá tải/quá quota. Hệ thống sẽ chuyển fallback local."
        if "token" in error_str or "limit" in error_str:
            return "Yêu cầu quá dài. Vui lòng rút gọn prompt."
        if status_code == 400 or "invalid" in error_str:
            return "Yêu cầu không hợp lệ. Kiểm tra lại nội dung/prompt."
        if status_code == 403 or "permission" in error_str:
            return "Cloud API bị từ chối quyền truy cập."
        if status_code == 401 or "auth" in error_str:
            return "Cloud API key không hợp lệ hoặc thiếu quyền."
        if status_code >= 500 or "internal" in error_str:
            return "Cloud AI server lỗi tạm thời. Hệ thống sẽ fallback local."

        return f"Lỗi cloud AI (mã {status_code})."

    def chat(self, prompt: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        if not conversation_id:
            conversation_id = "default"

        if conversation_id not in [h.get("conversation_id") for h in self._session_history]:
            self._session_history.append({"conversation_id": conversation_id, "messages": []})

        conv = next(
            (h for h in self._session_history if h.get("conversation_id") == conversation_id),
            None,
        )
        if conv:
            conv["messages"].append({"role": "user", "content": prompt})

        history = conv["messages"] if conv else []
        payload = self._build_chat_payload(prompt, history)

        result = self._call_cloud_api(payload, self._text_model_id)
        response_text = self._extract_response_text(result)

        if conv:
            conv["messages"].append({"role": "assistant", "content": response_text})

        return {
            "response": response_text,
            "conversation_id": conversation_id,
            "model": self._text_model_id,
            "timestamp": datetime.now().isoformat(),
            "provider": "cloud_ai",
        }

    def generate_images(self, prompt: str, num_images: int = 1) -> Dict[str, Any]:
        """Generate images from cloud API and return base64 payloads."""
        payload = self._build_image_payload(prompt, num_images=num_images)
        result = self._call_cloud_api(payload, self._image_model_id)

        images = self._extract_images_base64(result)
        if not images:
            raise NeuralServiceError("Cloud AI không trả về ảnh. Hệ thống sẽ fallback local.", 502)

        return {
            "images": images[: max(1, min(int(num_images), 4))],
            "model": self._image_model_id,
            "provider": "gemini",
            "timestamp": datetime.now().isoformat(),
        }

    def _extract_response_text(self, api_result: Dict) -> str:
        try:
            candidates = api_result.get("candidates", [])
            if not candidates:
                return "Không nhận được phản hồi từ hệ thống."

            parts = candidates[0].get("content", {}).get("parts", [])
            texts = [p.get("text", "") for p in parts if p.get("text")]
            return " ".join(texts) if texts else "Phản hồi trống từ hệ thống."
        except (KeyError, IndexError, TypeError) as e:
            logger.warning("Response extraction error: %s", e)
            return "Đã xảy ra lỗi khi xử lý phản hồi."

    def _extract_images_base64(self, api_result: Dict) -> List[str]:
        """Extract inline image data from cloud response."""
        extracted: List[str] = []
        candidates = api_result.get("candidates", [])

        for candidate in candidates:
            parts = candidate.get("content", {}).get("parts", [])
            for part in parts:
                inline = part.get("inlineData") or part.get("inline_data")
                if not isinstance(inline, dict):
                    continue
                data = inline.get("data")
                mime_type = inline.get("mimeType") or inline.get("mime_type", "")
                if data and str(mime_type).startswith("image/"):
                    extracted.append(data)

        return extracted

    def clear_history(self, conversation_id: Optional[str] = None):
        if conversation_id:
            self._session_history = [
                h for h in self._session_history if h.get("conversation_id") != conversation_id
            ]
        else:
            self._session_history = []
        logger.info("Cleared history for: %s", conversation_id or "all")

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "chat": {
                "provider": "gemini",
                "model": self._text_model_id,
            },
            "image_generation": {
                "provider": "gemini",
                "model": self._image_model_id,
                "fallback": "local_diffusion",
            },
            "max_tokens": 4096,
        }


class NeuralServiceError(Exception):
    """Custom exception for neural service errors."""

    def __init__(self, message: str, status_code: int = -1):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


def get_neural_service() -> NeuralInferenceService:
    """Get singleton neural service instance."""
    return NeuralInferenceService()
