"""
Local Chat Service - Template-based design assistant responses.
Used when no cloud AI API key is configured.
"""
import re
import random
from datetime import datetime
from typing import List, Dict, Any


class LocalChatService:
    """
    Lightweight local chat for design assistant responses.
    Covers: greetings, design advice, tool guidance.
    Falls back to: helpful redirect for complex queries.
    """

    _INSTANCE = None

    def __new__(cls):
        if cls._INSTANCE is None:
            cls._INSTANCE = super().__new__(cls)
            cls._INSTANCE._initialized = False
        return cls._INSTANCE

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._history: Dict[str, List[Dict[str, str]]] = {}

    # ─── Intent patterns ─────────────────────────────────────────────────────

    _GREETING_PATTERNS = [
        re.compile(r"^(xin chào|chào|hi|hello|hey|alo|chào bạn)\b", re.I),
        re.compile(r"^(bạn là ai|ai vậy|who are you|what are you)\b", re.I),
    ]
    _LOGO_PATTERNS = [
        re.compile(r"\b(logo|thương hiệu|brand|nhãn hiệu)\b", re.I),
    ]
    _POSTER_PATTERNS = [
        re.compile(r"\b(poster|áp phích|banner|quảng cáo)\b", re.I),
    ]
    _COLOR_PATTERNS = [
        re.compile(r"\b(màu|color|palette|tông màu|màu sắc|phối màu)\b", re.I),
    ]
    _TYPOGRAPHY_PATTERNS = [
        re.compile(r"\b(font|chữ|typography|kiểu chữ|text style)\b", re.I),
    ]
    _TOOL_PATTERNS = [
        re.compile(r"\b(cách|dùng|làm sao|hướng dẫn|how to|guide|tutorial)\b", re.I),
    ]
    _IMAGE_PATTERNS = [
        re.compile(r"\b(tạo ảnh|tao anh|generate|vẽ|draw|create image)\b", re.I),
    ]

    # ─── Response templates ──────────────────────────────────────────────────

    @staticmethod
    def _greeting_response() -> str:
        greetings = [
            "Chào bạn! Mình là Design AI — trợ lý thiết kế logo và poster. "
            "Bạn có thể nhắn prompt như 'tạo ảnh logo tối giản màu xanh' hoặc hỏi về nguyên tắc thiết kế nhé!",
            "Xin chào! Mình là AI Designer. "
            "Cứ mô tả ý tưởng logo/poster bằng tiếng Việt, mình sẽ tạo ảnh cho bạn ngay!",
            "Hey! Mình có thể giúp bạn tạo logo, poster hoặc tư vấn thiết kế. "
            "Hãy thử: 'tạo ảnh poster công nghệ tông xanh dương' nhé!",
        ]
        return random.choice(greetings)

    @staticmethod
    def _logo_response() -> str:
        tips = [
            "**Logo tốt cần có:**\n"
            "• Tính đơn giản — nhận diện được ở nhiều kích thước\n"
            "• Tính khác biệt — không nhầm với đối thủ\n"
            "• Tính phù hợp — đúng ngành, đúng đối tượng\n"
            "• Tính thời gian — không lỗi thời nhanh",
            "**Nguyên tắc thiết kế logo:**\n"
            "1. **Less is more** — càng đơn giản càng dễ nhớ\n"
            "2. Chọn 1-2 màu chủ đạo\n"
            "3. Font chữ đồng nhất, dễ đọc\n"
            "4. Đảm bảo hoạt động tốt ở cả trắng đen",
            "**Mình gợi ý bạn thử:**\n"
            "• Logo tối giản (minimalist) — phù hợp tech/startup\n"
            "• Logo chữ (wordmark) — phù hợp thương hiệu lớn\n"
            "• Logo biểu tượng (emblem) — phù hợp tổ chức/câu lạc bộ",
        ]
        return random.choice(tips)

    @staticmethod
    def _poster_response() -> str:
        tips = [
            "**Poster hiệu quả cần:**\n"
            "• Hình ảnh nổi bật, thu hút ánh nhìn\n"
            "• Tiêu đề ngắn gọn, dễ đọc từ xa\n"
            "• Contrast tốt giữa chữ và nền\n"
            "• Hierarchy rõ ràng: tiêu đề → mô tả → CTA",
            "**Xu hướng poster 2025:**\n"
            "• Phong cách brutalist — font to, góc cạnh\n"
            "• Retro/vintage — phong cách hoài cổ\n"
            "• Minimal neon — màu nổi trên nền tối\n"
            "• 3D render — mockup thực tế",
            "**Bố cục poster chuẩn:**\n"
            "• Rule of thirds — chia ảnh 3 phần\n"
            "• Trọng tâm có thể là tâm ảnh hoặc điểm giao\n"
            "• Khoảng trắng đủ để mắt nghỉ ngơi",
        ]
        return random.choice(tips)

    @staticmethod
    def _color_response() -> str:
        tips = [
            "**Chọn bảng màu cho thương hiệu:**\n"
            "1. Màu chính (primary) — cảm xúc chủ đạo\n"
            "2. Màu phụ (secondary) — bổ sung, tương phản\n"
            "3. Màu trung tính (neutral) — nền, chữ\n\n"
            "**Gợi ý theo ngành:**\n"
            "• Tech: xanh dương + xám + trắng\n"
            "• Nội thất: be + nâu + xanh olive\n"
            "• F&B: đỏ/cam + vàng + trắng\n"
            "• Y tế: xanh lá + trắng + xanh nhạt",
            "**Công cụ chọn màu:**\n"
            "• Coolors.co — random palette generator\n"
            "• Adobe Color — tạo từ ảnh\n"
            "• Material Design palette — hệ thống màu chuẩn\n\n"
            "Bạn muốn mình tạo logo với tông màu cụ thể nào?",
            "**60-30-10 rule:**\n"
            "• 60% — màu nền trung tính\n"
            "• 30% — màu phụ\n"
            "• 10% — màu nhấn (accent)\n\n"
            "Tỷ lệ này tạo cảm giác cân bằng và chuyên nghiệp.",
        ]
        return random.choice(tips)

    @staticmethod
    def _typography_response() -> str:
        tips = [
            "**Chọn font cho logo/poster:**\n"
            "• Sans-serif (Arial, Helvetica) — hiện đại, sạch sẽ\n"
            "• Serif (Times, Georgia) — cổ điển, đáng tin cậy\n"
            "• Display (các font trang trí) — sáng tạo, nổi bật\n\n"
            "**Tip:** Font chữ trong logo nên là 1-2 font duy nhất, "
            "không nên mix quá nhiều.",
            "**Font đang trending:**\n"
            "• Inter, Poppins, Montserrat — sans-serif phổ biến\n"
            "• Playfair Display, Cormorant — serif thanh lịch\n"
            "• Bebas Neue, Oswald — condensed cho poster\n\n"
            "Bạn thích phong cách nào? Mình sẽ gợi ý font cụ thể.",
        ]
        return random.choice(tips)

    @staticmethod
    def _tool_response(user_msg: str) -> str:
        return (
            "Mình là AI Designer — công cụ tạo logo và poster bằng AI.\n\n"
            "**Cách dùng:**\n"
            "1. Gõ prompt mô tả ý tưởng (VD: 'tạo ảnh logo tech startup tông xanh dương')\n"
            "2. Chọn model phù hợp (logo 2D / logo 3D / poster)\n"
            "3. Nhấn gửi → AI tạo ảnh cho bạn\n"
            "4. Tải PNG hoặc chuyển sang SVG vector\n\n"
            "**Mẹo prompt tốt:**\n"
            "• Mô tả rõ: chủ đề + màu sắc + phong cách\n"
            "• Thêm: 'minimalist', 'modern', 'flat design'\n"
            "• Thêm: 'on white background', 'transparent background'\n\n"
            "Bạn muốn tạo ảnh gì?"
        )

    @staticmethod
    def _image_request_response() -> str:
        return (
            "Mình hiểu bạn muốn tạo ảnh! "
            "Hãy gõ prompt theo cú pháp: **'tạo ảnh [mô tả chi tiết]'** "
            "và mình sẽ sinh logo/poster cho bạn.\n\n"
            "Ví dụ:\n"
            "• `tạo ảnh logo coffee shop phong cách vintage`\n"
            "• `tạo ảnh poster sự kiện công nghệ tông xanh neon`\n"
            "• `tạo ảnh logo 3D công ty bất động sản cao cấp`"
        )

    @staticmethod
    def _fallback_response(user_msg: str) -> str:
        fallbacks = [
            f"Mình nghe thấy '{user_msg[:60]}...' — bạn cần mình hỗ trợ gì về thiết kế?\n\n"
            "Mình có thể:\n"
            "• Tạo logo theo mô tả\n"
            "• Tư vấn màu sắc, typography\n"
            "• Gợi ý bố cục poster",
            f"Ý tưởng hay đó! Mình có thể giúp bạn hiện thực hóa.\n\n"
            "Hãy thử prompt cụ thể hơn, ví dụ:\n"
            "• 'tạo ảnh logo startup AI tông xanh dương'\n"
            "• 'tạo ảnh poster sự kiện tech với màu neon'",
        ]
        return random.choice(fallbacks)

    # ─── Main dispatch ────────────────────────────────────────────────────────

    def chat(self, message: str, conversation_id: str = "default") -> Dict[str, Any]:
        """
        Generate a local response based on message intent.
        Returns dict compatible with the chat router response format.
        """
        msg = message.strip()
        lower = msg.lower()

        # Store in history
        if conversation_id not in self._history:
            self._history[conversation_id] = []
        self._history[conversation_id].append({"role": "user", "content": msg})

        response_text = ""

        # Priority: image generation intent → guide user to generation
        if self._IMAGE_PATTERNS[0].search(lower) or self._IMAGE_PATTERNS[1].search(lower):
            response_text = self._image_request_response()
        # Greeting
        elif any(p.search(lower) for p in self._GREETING_PATTERNS):
            response_text = self._greeting_response()
        # Logo topic
        elif any(p.search(lower) for p in self._LOGO_PATTERNS):
            response_text = self._logo_response()
        # Poster topic
        elif any(p.search(lower) for p in self._POSTER_PATTERNS):
            response_text = self._poster_response()
        # Color topic
        elif any(p.search(lower) for p in self._COLOR_PATTERNS):
            response_text = self._color_response()
        # Typography topic
        elif any(p.search(lower) for p in self._TYPOGRAPHY_PATTERNS):
            response_text = self._typography_response()
        # Tool/how-to
        elif any(p.search(lower) for p in self._TOOL_PATTERNS):
            response_text = self._tool_response(msg)
        # Fallback
        else:
            response_text = self._fallback_response(msg)

        self._history[conversation_id].append({"role": "assistant", "content": response_text})

        return {
            "response": response_text,
            "conversation_id": conversation_id,
            "model": "local-chat",
            "timestamp": datetime.now().isoformat(),
            "provider": "local",
        }

    def clear_history(self, conversation_id: str = None):
        """Clear conversation history."""
        if conversation_id:
            self._history.pop(conversation_id, None)
        else:
            self._history.clear()

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "chat": {
                "provider": "local",
                "model": "template-based",
                "description": "Design assistant with logo/poster guidance",
            },
            "image_generation": {
                "provider": "local",
                "model": "SDXL-Turbo + LoRA",
                "adapters": ["lora_logo_2d", "lora_logo_3d", "lora_poster"],
            },
        }


def get_local_chat_service() -> LocalChatService:
    return LocalChatService()
