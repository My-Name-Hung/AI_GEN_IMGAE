"""
Smart Prompt Analyzer - Auto-detect LoRA type and optimize generation parameters
from natural language prompts. No ML models needed — pure keyword/heuristic analysis.
"""
import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Optimized generation parameters derived from prompt analysis."""
    lora_type: str  # "lora_logo_2d" | "lora_logo_3d" | "lora_poster" | "base"
    mode: str       # "turbo" | "standard" | "quality"
    num_inference_steps: int
    guidance_scale: float
    width: int
    height: int
    negative_prompt: str
    enhanced_prompt: str
    style_keywords: List[str] = field(default_factory=list)
    subject_type: str = "unknown"
    confidence: float = 1.0


class SmartPromptAnalyzer:
    """
    Analyzes natural language prompts to auto-select:
    - LoRA adapter (logo_2d / logo_3d / poster / base)
    - Generation mode (turbo / standard / quality)
    - Inference steps + CFG scale
    - Resolution (1:1 / 16:9 / 9:16 / 4:3)
    - Enhanced prompt with style keywords
    - Targeted negative prompt
    """

    # ─── Keyword sets ──────────────────────────────────────────────────────────

    # Logo 2D patterns: flat, minimal, geometric, clean
    LOGO_2D_TRIGGERS = [
        # Vietnamese
        r"\blogo\b", r"\bthương hiệu\b", r"\bnhãn hiệu\b", r"\bbrand\b",
        r"\bbiểu tượng\b", r"\bicon\b", r"\bsymbol\b",
        r"\btối giản\b", r"\bminimal\b", r"\bflat\b", r"\bgeometric\b",
        r"\bclean\b", r"\bsimple\b", r"\blucid\b", r"\bđơn giản\b",
        r"\bvector\b", r"\bchìm\b", r"\betched\b", r"\bembossed\b",
        r"\bfood\b", r"\bđồ ăn\b", r"\brestaurant\b", r"\bcafe\b",
        r"\bcoffee\b", r"\bcoffee shop\b", r"\bnhà hàng\b",
        r"\bfashion\b", r"\bthời trang\b", r"\bclothing\b", r"\bapparel\b",
        r"\btech\b", r"\btechnology\b", r"\bcông nghệ\b", r"\b startup\b",
        r"\bSaaS\b", r"\bsoftware\b", r"\bapp\b", r"\bplatform\b",
        r"\bhealth\b", r"\bsức khỏe\b", r"\by tế\b", r"\bmedical\b",
        r"\bfitness\b", r"\bgym\b", r"\bthể thao\b", r"\bsport\b",
        r"\beducation\b", r"\bgiáo dục\b", r"\bschool\b", r"\buniversity\b",
        r"\bfinance\b", r"\btài chính\b", r"\bbank\b", r"\bfintech\b",
        r"\breal estate\b", r"\bbất động sản\b", r"\bproperty\b",
        r"\blogo chữ\b", r"\bwordmark\b", r"\blettering\b",
        # English
        r"flat design", r"minimalist logo", r"brand logo", r"company logo",
        r"business logo", r"shop logo", r"store logo", r"app icon",
    ]

    # Logo 3D patterns: 3D, render, realistic, metallic, glass, neon
    LOGO_3D_TRIGGERS = [
        # Vietnamese
        r"\b3D\b", r"\b3d\b", r"\brender\b", r"\bchiều\b",
        r"\bchrome\b", r"\bmetallic\b", r"\bmetal\b", r"\bvàng gold\b",
        r"\bgold\b", r"\bsilver\b", r"\bbạc\b", r"\bbronze\b",
        r"\bglass\b", r"\bkính\b", r"\btransparent\b", r"\btrong suốt\b",
        r"\bneon\b", r"\bled\b", r"\bglowing\b", r"\bsáng\b",
        r"\bhologram\b", r"\bholographic\b", r"\biridescent\b",
        r"\bglassmorphism\b", r"\bglossy\b", r"\bgloss\b",
        r"\bmatte\b", r"\brough\b", r"\bholographic\b",
        r"\bhightech\b", r"\bhi-tech\b", r"\bfuturistic\b",
        r"\btương lai\b", r"\bfuture\b", r"\bsci-fi\b",
        r"\bluxury\b", r"\bluxe\b", r"\bcao cấp\b", r"\bcông ti\b",
        r"\bgame\b", r"\bgame studio\b", r"\bdev studio\b",
        # English
        r"3d logo", r"3d render", r"metallic", r"metallic texture",
        r"gold texture", r"glass effect", r"neon glow", r"neon sign",
        r"realistic render", r"product render", r"cg render",
        r"glossy finish", r"matte finish", r"holographic effect",
        r"luxury brand", r"premium", r"exclusive",
        r"isometric", r"2.5d", r"pseudo-3d",
    ]

    # Poster patterns: movie, event, banner, cinematic, editorial
    POSTER_TRIGGERS = [
        # Vietnamese
        r"\bposter\b", r"\báp phích\b", r"\bbanner\b", r"\bquảng cáo\b",
        r"\breclame\b", r"\bleaflet\b", r"\bbrochure\b", r"\btờ rơi\b",
        r"\bphim\b", r"\bmovie\b", r"\bcinema\b", r"\bđiện ảnh\b",
        r"\bgame\b", r"\bgaming\b", r"\bsự kiện\b", r"\bevent\b",
        r"\bhội thảo\b", r"\bworkshop\b", r"\bconcert\b", r"\b âm nhạc\b",
        r"\bfestival\b", r"\blễ hội\b", r"\bexpo\b", r"\bexhibition\b",
        r"\btrưng bày\b", r"\bconcert\b", r"\btour\b",
        r"\bấn phẩm\b", r"\bpublishing\b", r"\beditorial\b",
        r"\btạp chí\b", r"\bmagazine\b", r"\bcover\b",
        r"\bbook cover\b", r"\bsách\b", r"\bebook\b",
        r"\bsocial media\b", r"\bfacebook\b", r"\binstagram\b",
        r"\bthumbnail\b", r"\byoutube\b",
        r"\bcd\b", r"\bdvd\b", r"\bproduct\b", r"\bsản phẩm\b",
        r"\bpackaging\b", r"\bbao bì\b", r"\blabel\b", r"\bnhãn\b",
        # English
        r"movie poster", r"film poster", r"cinematic poster",
        r"event poster", r"concert poster", r"festival poster",
        r"gaming poster", r"stream poster", r"youtube thumbnail",
        r"social media banner", r"instagram post", r"facebook ad",
        r"flyer design", r"brochure", r"print design",
        r"book cover", r"album cover", r"cd cover",
        r"product advertisement", r"ad banner", r"web banner",
        r"editorial design", r"magazine layout",
    ]

    # Quality/detail keywords → use quality mode
    QUALITY_TRIGGERS = [
        r"\bchất lượng cao\b", r"\bcao cấp\b", r"\b4K\b", r"\b8K\b",
        r"\bHD\b", r"\bsharp\b", r"\bsắc nét\b", r"\bchi tiết\b",
        r"\bhighly detailed\b", r"\bintricate\b", r"\belaborate\b",
        r"\brealistic\b", r"\bphotorealistic\b", r"\bhyperrealistic\b",
        r"\bphotorealistic render\b", r"\bmasterpiece\b",
        r"\baward-winning\b", r"\bcompetition\b",
        r"\bstunning\b", r"\bimpressive\b", r"\bprofessional\b",
        r"\bcommercial\b", r"\bprint-ready\b", r"\bproduction\b",
        r"\bcinematic lighting\b", r"\bvolumetric\b", r"\bray tracing\b",
    ]

    # Speed keywords → use turbo mode
    SPEED_TRIGGERS = [
        r"\bnhanh\b", r"\bquick\b", r"\bsnapshot\b", r"\bsdraft\b",
        r"\bsketchy\b", r"\brough\b", r"\brough draft\b",
        r"\bplaceholder\b", r"\bdraft\b", r"\bthumbnail\b",
        r"\bconcept\b", r"\bideas\b", r"\bbrainstorm\b",
    ]

    # ─── Style keywords ───────────────────────────────────────────────────────

    STYLE_PATTERNS = {
        "minimalist": [
            r"\bminimal\b", r"\btối giản\b", r"\bclean\b", r"\bsimple\b",
            r"\bflat design\b", r"\bless is more\b", r"\bsleek\b",
        ],
        "modern": [
            r"\bmodern\b", r"\bhiện đại\b", r"\bcontemporary\b",
            r"\btrendy\b", r"\bcurrent\b", r"\btendency\b",
        ],
        "vintage": [
            r"\bvintage\b", r"\bretro\b", r"\bclassic\b", r"\bcổ điển\b",
            r"\bold school\b", r"\bhoài cổ\b", r"\bnostalgic\b",
        ],
        "neon": [
            r"\bneon\b", r"\bled\b", r"\bglowing\b", r"\bfuturistic\b",
            r"\bsáng\b", r"\btương lai\b", r"\bcyber\b",
        ],
        "brutalist": [
            r"\bbrutalist\b", r"\bbrutalism\b", r"\blocky\b",
            r"\bgóc cạnh\b", r"\bthick\b", r"\bbold\b",
        ],
        "corporate": [
            r"\bcorporate\b", r"\bprofessional\b", r"\bbusiness\b",
            r"\bdoanh nghiệp\b", r"\benterprise\b", r"\bcompany\b",
        ],
        "playful": [
            r"\bplayful\b", r"\bfun\b", r"\bvui\b", r"\bcolorful\b",
            r"\bcheerful\b", r"\bhappy\b", r"\bquirky\b", r"\bwhimsical\b",
        ],
        "luxury": [
            r"\bluxury\b", r"\bluxe\b", r"\bcao cấp\b", r"\bpremium\b",
            r"\bexpensive\b", r"\bhigh-end\b", r"\belegant\b", r"\bđẳng cấp\b",
        ],
        "tech": [
            r"\btech\b", r"\btechnology\b", r"\bdigital\b", r"\bcyber\b",
            r"\bAI\b", r"\brobotic\b", r"\bcircuit\b", r"\bdata\b",
        ],
        "nature": [
            r"\bnature\b", r"\bnatural\b", r"\btự nhiên\b", r"\borganic\b",
            r"\bleaf\b", r"\btree\b", r"\bplant\b", r"\bgreen\b", r"\bforest\b",
        ],
    }

    # ─── Subject type detection ──────────────────────────────────────────────

    SUBJECT_PATTERNS = {
        "tech": [r"\btech\b", r"\bAI\b", r"\bsoftware\b", r"\bapp\b", r"\bstartup\b",
                  r"\bIT\b", r"\bcomputer\b", r"\bcode\b", r"\bcông nghệ\b"],
        "food": [r"\bfood\b", r"\bđồ ăn\b", r"\brestaurant\b", r"\bcafe\b", r"\bcoffee\b",
                  r"\bnhà hàng\b", r"\bfood\b", r"\bdrink\b", r"\bbeverage\b"],
        "fashion": [r"\bfashion\b", r"\bthời trang\b", r"\bclothing\b", r"\bshop\b",
                     r"\bstore\b", r"\bretail\b", r"\bskin\b", r"\bbeauty\b"],
        "fitness": [r"\bfitness\b", r"\bgym\b", r"\bsport\b", r"\bworkout\b",
                     r"\bhealth\b", r"\byoga\b", r"\bthể thao\b"],
        "finance": [r"\bfinance\b", r"\bbank\b", r"\bfintech\b", r"\bmoney\b",
                     r"\btài chính\b", r"\binvestment\b"],
        "real_estate": [r"\breal estate\b", r"\bbất động sản\b", r"\bproperty\b",
                         r"\bhome\b", r"\bhousing\b", r"\bapartment\b"],
        "music": [r"\bmusic\b", r"\bâm nhạc\b", r"\bband\b", r"\balbum\b",
                   r"\bconcert\b", r"\bproducer\b", r"\bdj\b"],
        "gaming": [r"\bgame\b", r"\bgaming\b", r"\bplaystation\b", r"\bxbox\b",
                    r"\bsteam\b", r"\besport\b"],
        "medical": [r"\bmedical\b", r"\by tế\b", r"\bhealth\b", r"\bpharma\b",
                     r"\bhospital\b", r"\bclinic\b"],
        "education": [r"\beducation\b", r"\bgiáo dục\b", r"\bschool\b",
                       r"\buniversity\b", r"\bcourse\b", r"\blms\b"],
    }

    # ─── Aspect ratio / size detection ──────────────────────────────────────

    ASPECT_PATTERNS = {
        (1216, 832): [r"16:9", r"wide\b", r"\brộng\b", r"\blandscape\b",
                       r"\bfacebook\b", r"\byoutube\b", r"\bweb\banner",
                       r"\bwebsite\b", r"\bdesktop\b", r"\bposter horizontal"],
        (832, 1216): [r"9:16", r"vertical\b", r"\bdọc\b", r"\bportrait\b",
                       r"\bstory\b", r"\binstagram story\b", r"\btiktok\b",
                       r"\bsocial media\b", r"\breel\b", r"\bmobile\b"],
        (1024, 768): [r"4:3", r"\bclassic\b", r"\bstandard\b",
                       r"\bpresentation\b", r"\bslide\b"],
        (1024, 1024): [r"1:1", r"\bsquare\b", r"\bvuông\b",
                         r"\blogo\b", r"\bicon\b", r"\bavatar\b",
                         r"\bprofile\b", r"\bthumbnail\b", r"\bbadge\b"],
    }

    # ─── Enhanced negative prompts ─────────────────────────────────────────────

    BASE_NEGATIVE = (
        "low quality, blurry, distorted, deformed, bad anatomy, "
        "extra limbs, watermark, text, logo, signature, "
        "cropped, out of frame, worst quality, low resolution, "
        "jpeg artifacts, noise, grainy"
    )

    NEGATIVE_PRESETS = {
        "logo_2d": (
            "low quality, blurry, distorted, deformed, bad anatomy, "
            "extra limbs, watermark, text, logo, signature, "
            "3d render, realistic, shadow, depth, 2d abstraction, "
            "cartoon, illustration style, drawing, painting"
        ),
        "logo_3d": (
            "low quality, blurry, flat, 2d, drawing, painting, "
            "illustration, cartoon, anime, watermark, text, logo, "
            "faded, low poly, ugly, distorted, deformed"
        ),
        "poster": (
            "low quality, blurry, distorted, deformed, bad anatomy, "
            "watermark, text overlay, cluttered, busy, low resolution, "
            "jpeg artifacts, amateur, unprofessional, "
            "logo prominent, icon, small image"
        ),
        "base": BASE_NEGATIVE,
    }

    # ─── Color detection ──────────────────────────────────────────────────────

    COLOR_PATTERNS = {
        "red": [r"\bđỏ\b", r"\bred\b", r"\btím đậm\b"],
        "blue": [r"\bxanh dương\b", r"\bblue\b", r"\bnavy\b"],
        "green": [r"\bxanh lá\b", r"\bgreen\b", r"\bforest\b", r"\bteal\b"],
        "yellow": [r"\bvàng\b", r"\byellow\b", r"\bgold\b", r"\borange\b"],
        "black": [r"\bđen\b", r"\bblack\b"],
        "white": [r"\btrắng\b", r"\bwhite\b"],
        "purple": [r"\btím\b", r"\bpurple\b", r"\bviolet\b"],
        "pink": [r"\bhồng\b", r"\bpink\b"],
        "neon": [r"\bneon\b", r"\bglowing\b", r"\bbright\b", r"\bfluorescent\b"],
        "pastel": [r"\bpastel\b", r"\bsoft\b", r"\blight color\b"],
    }

    # ─── Main analysis method ─────────────────────────────────────────────────

    def analyze(self, prompt: str) -> GenerationConfig:
        """
        Analyze a natural language prompt and return optimized GenerationConfig.

        Detection priority for LoRA type:
          1. logo_3d triggers → lora_logo_3d
          2. poster triggers → lora_poster
          3. logo_2d triggers → lora_logo_2d
          4. else → base model

        Returns GenerationConfig with all fields populated.
        """
        p = prompt.strip()
        p_lower = p.lower()

        # ── Detect LoRA type ─────────────────────────────────────────────────
        lora_type, confidence = self._detect_lora_type(p_lower)

        # ── Detect style keywords ───────────────────────────────────────────
        style_keywords = self._detect_styles(p_lower)

        # ── Detect subject type ─────────────────────────────────────────────
        subject_type = self._detect_subject(p_lower)

        # ── Detect quality vs speed intent ───────────────────────────────────
        quality_intent = any(re.search(p, p_lower) for p in self.QUALITY_TRIGGERS)
        speed_intent = any(re.search(p, p_lower) for p in self.SPEED_TRIGGERS)

        # ── Determine mode ────────────────────────────────────────────────────
        if quality_intent and not speed_intent:
            mode = "quality"
        elif speed_intent and not quality_intent:
            mode = "turbo"
        else:
            mode = "standard"  # default

        # ── Determine steps + CFG ────────────────────────────────────────────
        if mode == "turbo":
            num_steps = 1
            cfg = 0.0
        elif mode == "standard":
            num_steps = 4
            cfg = 7.5
        else:  # quality
            num_steps = 8
            cfg = 8.0

        # ── Detect resolution ────────────────────────────────────────────────
        width, height = self._detect_resolution(p_lower)

        # ── Generate negative prompt ─────────────────────────────────────────
        negative_prompt = self.NEGATIVE_PRESETS.get(lora_type, self.BASE_NEGATIVE)

        # ── Generate enhanced prompt ──────────────────────────────────────────
        enhanced_prompt = self._enhance_prompt(p, lora_type, style_keywords)

        logger.info(
            "Prompt analysis: lora=%s mode=%s steps=%d cfg=%.1f "
            "size=%dx%d subject=%s styles=%s",
            lora_type, mode, num_steps, cfg, width, height,
            subject_type, style_keywords,
        )

        return GenerationConfig(
            lora_type=lora_type,
            mode=mode,
            num_inference_steps=num_steps,
            guidance_scale=cfg,
            width=width,
            height=height,
            negative_prompt=negative_prompt,
            enhanced_prompt=enhanced_prompt,
            style_keywords=style_keywords,
            subject_type=subject_type,
            confidence=confidence,
        )

    # ─── Detection helpers ────────────────────────────────────────────────────

    def _detect_lora_type(self, p_lower: str) -> tuple[str, float]:
        """Return (lora_type, confidence) based on keyword matching."""
        scores = {
            "lora_logo_3d": 0.0,
            "lora_poster": 0.0,
            "lora_logo_2d": 0.0,
        }

        for pat in self.LOGO_3D_TRIGGERS:
            if re.search(pat, p_lower):
                scores["lora_logo_3d"] += 1

        for pat in self.POSTER_TRIGGERS:
            if re.search(pat, p_lower):
                scores["lora_poster"] += 1

        for pat in self.LOGO_2D_TRIGGERS:
            if re.search(pat, p_lower):
                scores["lora_logo_2d"] += 1

        if scores["lora_logo_3d"] > 0:
            total = sum(scores.values())
            confidence = scores["lora_logo_3d"] / total if total > 0 else 1.0
            return "lora_logo_3d", min(confidence * 1.5, 1.0)

        if scores["lora_poster"] > 0:
            total = sum(scores.values())
            confidence = scores["lora_poster"] / total if total > 0 else 1.0
            return "lora_poster", min(confidence * 1.5, 1.0)

        if scores["lora_logo_2d"] > 0:
            total = sum(scores.values())
            confidence = scores["lora_logo_2d"] / total if total > 0 else 1.0
            return "lora_logo_2d", min(confidence * 1.5, 1.0)

        return "base", 0.5

    def _detect_styles(self, p_lower: str) -> List[str]:
        """Return list of detected style keywords."""
        found = []
        for style, patterns in self.STYLE_PATTERNS.items():
            if any(re.search(p, p_lower) for p in patterns):
                found.append(style)
        return found[:3]  # max 3 styles

    def _detect_subject(self, p_lower: str) -> str:
        """Return the detected subject category."""
        for subject, patterns in self.SUBJECT_PATTERNS.items():
            if any(re.search(p, p_lower) for p in patterns):
                return subject
        return "general"

    def _detect_resolution(self, p_lower: str) -> tuple[int, int]:
        """Detect aspect ratio / resolution from prompt keywords."""
        for (w, h), patterns in self.ASPECT_PATTERNS.items():
            if any(re.search(p, p_lower) for p in patterns):
                return w, h
        return 1024, 1024  # default 1:1

    def _enhance_prompt(
        self,
        original: str,
        lora_type: str,
        styles: List[str],
    ) -> str:
        """
        Build an enhanced prompt that includes style keywords
        and subject context to maximize generation quality.
        """
        parts = [original.strip()]

        # Add LoRA activation prefix
        if lora_type == "lora_logo_2d":
            parts.insert(0, "logo design, flat style, clean graphic")
        elif lora_type == "lora_logo_3d":
            parts.insert(0, "3D logo render, realistic material")
        elif lora_type == "lora_poster":
            parts.insert(0, "professional poster design, editorial")

        # Add style keywords
        style_map = {
            "minimalist": "minimalist, clean, simple",
            "modern": "modern, contemporary, trendy",
            "vintage": "vintage, retro, classic style",
            "neon": "neon glow, cyberpunk, futuristic",
            "brutalist": "brutalist, bold, impactful",
            "corporate": "corporate, professional, business",
            "playful": "playful, colorful, fun",
            "luxury": "luxury, premium, high-end",
            "tech": "tech, digital, futuristic",
            "nature": "natural, organic, eco-friendly",
        }
        for style in styles:
            if style in style_map:
                parts.append(style_map[style])

        # Add quality anchors
        parts.append("high detail, sharp, professional quality, no watermark")

        return ", ".join(filter(None, parts))


# ─── Module-level singleton ───────────────────────────────────────────────────

_analyzer: Optional[SmartPromptAnalyzer] = None


def get_prompt_analyzer() -> SmartPromptAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = SmartPromptAnalyzer()
    return _analyzer
