"""
Vectorizer Service - PNG/JPG to SVG conversion.
Uses OpenCV contour tracing + Ramer-Douglas-Peucker path simplification
+ color quantization for clean SVG output.
Fully offline — no third-party API calls.
"""
import io
import json
import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ─── Bezier & path helpers ────────────────────────────────────────────────────

def _smooth_bezier(
    points: List[Tuple[int, int]],
    smoothness: float = 0.25,
) -> str:
    """
    Convert ordered polygon points to smooth cubic Bezier SVG path.
    smoothness controls how much the control points deviate (0.0 = straight lines only).
    """
    if len(points) < 2:
        return ""
    if len(points) == 2:
        return f"M {points[0][0]} {points[0][1]} L {points[1][0]} {points[1][1]}"

    n = len(points)
    path = f"M {points[0][0]} {points[0][1]}"

    for i in range(n - 1):
        p0 = points[(i - 1) % n]
        p1 = points[i]
        p2 = points[i + 1]
        p3 = points[(i + 2) % n] if i + 2 < n else points[i + 1]

        # Catmull-Rom → cubic Bezier conversion
        cp1x = p1[0] + smoothness * (p2[0] - p0[0]) / 6
        cp1y = p1[1] + smoothness * (p2[1] - p0[1]) / 6
        cp2x = p2[0] - smoothness * (p3[0] - p1[0]) / 6
        cp2y = p2[1] - smoothness * (p3[1] - p1[1]) / 6

        path += f" C {cp1x:.1f} {cp1y:.1f} {cp2x:.1f} {cp2y:.1f} {p2[0]} {p2[1]}"

    path += " Z"
    return path


def _rdp_simplify(points: List[Tuple[float, float]], epsilon: float) -> List[Tuple[float, float]]:
    """
    Ramer-Douglas-Peucker path simplification.
    Reduces the number of points while preserving shape within epsilon tolerance.
    """
    if len(points) < 3:
        return points

    # Find point with max distance from line segment
    start, end = points[0], points[-1]
    max_dist = 0.0
    max_idx = 0

    for i, pt in enumerate(points[1:-1], start=1):
        d = _perp_dist(pt, start, end)
        if d > max_dist:
            max_dist = d
            max_idx = i

    if max_dist > epsilon:
        left = _rdp_simplify(points[:max_idx + 1], epsilon)
        right = _rdp_simplify(points[max_idx:], epsilon)
        return left[:-1] + right
    return [start, end]


def _perp_dist(
    pt: Tuple[float, float],
    line_start: Tuple[float, float],
    line_end: Tuple[float, float],
) -> float:
    """Perpendicular distance from point to line segment."""
    dx, dy = line_end[0] - line_start[0], line_end[1] - line_start[1]
    if dx == 0 and dy == 0:
        return math.hypot(pt[0] - line_start[0], pt[1] - line_start[1])
    t = max(0.0, min(1.0, (
        (pt[0] - line_start[0]) * dx + (pt[1] - line_start[1]) * dy
    ) / (dx * dx + dy * dy)))
    proj_x = line_start[0] + t * dx
    proj_y = line_start[1] + t * dy
    return math.hypot(pt[0] - proj_x, pt[1] - proj_y)


def _largest_enclosing_rect(pts: List[Tuple[int, int]], w: int, h: int) -> Optional[Dict]:
    """Compute axis-aligned bounding rect for a point set."""
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x, y = min(xs), min(ys)
    bw, bh = max(xs) - x, max(ys) - y
    return {"x": int(x), "y": int(y), "w": int(bw), "h": int(bh)}


# ─── Color quantization ─────────────────────────────────────────────────────────

def _quantize_pil(image: Image.Image, num_colors: int) -> Tuple[Image.Image, Dict]:
    """
    Quantize image using PIL adaptive palette.
    Returns quantized image + mapping of palette index → RGB.
    """
    quantized = image.quantize(colors=num_colors, method=2)  # 2=FASTOCTREE (best quality)
    palette = quantized.getpalette()
    idx_to_rgb = {}
    if palette:
        for i in range(min(num_colors, 256)):
            r, g, b = palette[i * 3], palette[i * 3 + 1], palette[i * 3 + 2]
            idx_to_rgb[i] = (r, g, b)
    return quantized, idx_to_rgb


def _detect_dominant_colors(image: Image.Image, num_colors: int) -> List[Tuple[int, int, int]]:
    """
    Detect dominant colors using K-means in RGB space.
    More accurate than PIL palette for color analysis.
    """
    img_small = image.resize((128, 128), Image.Resampling.LANCZOS)
    arr = np.array(img_small, dtype=np.float32).reshape(-1, 3)
    try:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(arr, num_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        colors = sorted(centers.astype(np.uint8).tolist())
        return [(int(c[0]), int(c[1]), int(c[2])) for c in colors]
    except Exception:
        return [(0, 0, 0), (128, 128, 128), (255, 255, 255)]


# ─── Contour extraction per color ──────────────────────────────────────────────

def _extract_color_contours(
    image: Image.Image,
    target_rgb: Tuple[int, int, int],
    tolerance: int = 15,
    min_area: float = 4.0,
) -> List[List[Tuple[int, int]]]:
    """
    Find contours of all regions matching a target RGB color (±tolerance).
    Returns list of contours, each as list of (x, y) points.
    """
    arr = np.array(image.convert("RGB"))
    r, g, b = target_rgb

    # Build mask within color tolerance
    lower = np.array([max(0, r - tolerance), max(0, g - tolerance), max(0, b - tolerance)], dtype=np.uint8)
    upper = np.array([min(255, r + tolerance), min(255, g + tolerance), min(255, b + tolerance)], dtype=np.uint8)
    mask = cv2.inRange(arr, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        # Simplify contour
        epsilon = 1.5
        approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
        pts = [(int(p[0][0]), int(p[0][1])) for p in approx]
        if len(pts) >= 3:
            results.append(pts)

    return results


def _extract_grayscale_contours(
    image: Image.Image,
    bg_white: bool = True,
    min_area: float = 4.0,
) -> Tuple[List[List[Tuple[int, int]]], int]:
    """
    Extract contours from grayscale image.
    If bg_white=True: finds dark regions; else: finds light regions.
    Returns (contours, dominant_bg_brightness).
    """
    gray = np.array(image.convert("L"))
    h, w = gray.shape
    total_pixels = h * w

    median_bg = int(np.median(gray))
    dark_mask = gray < median_bg if bg_white else gray > median_bg

    dark_arr = (dark_mask.astype(np.uint8)) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark_arr = cv2.morphologyEx(dark_arr, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(dark_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        epsilon = 1.5
        approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
        pts = [(int(p[0][0]), int(p[0][1])) for p in approx]
        if len(pts) >= 3:
            results.append(pts)

    return results, median_bg


# ─── SVG generation ─────────────────────────────────────────────────────────────

def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def _generate_svg(
    layers: List[Dict],
    w: int,
    h: int,
    simplify_eps: float,
    bezier_smooth: float,
    bg_color: Optional[str],
) -> str:
    """Build complete SVG string from color layers."""
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg"',
        f'     xmlns:xlink="http://www.w3.org/1999/xlink"',
        f'     width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
    ]

    # Background
    bg = bg_color or "#ffffff"
    parts.append(f'  <rect width="{w}" height="{h}" fill="{bg}"/>')

    # Layers sorted by pixel count (small on top = darker, usually foreground)
    sorted_layers = sorted(layers, key=lambda l: l["pixel_count"])

    for layer in sorted_layers:
        color = layer["color"]
        color_hex = _rgb_to_hex(*color) if isinstance(color, (list, tuple)) else str(color)
        paths = layer["paths"]
        if not paths:
            continue

        parts.append(f'  <g id="layer_{layer["layer_id"]}" fill="{color_hex}">')
        for path_pts in paths:
            if len(path_pts) < 3:
                continue
            # Simplify first, then smooth
            simplified = _rdp_simplify(path_pts, epsilon=simplify_eps)
            if len(simplified) < 3:
                continue
            d = _smooth_bezier(simplified, smoothness=bezier_smooth)
            if d:
                parts.append(f'    <path d="{d}"/>')
        parts.append('  </g>')

    parts.append('</svg>')
    return "\n".join(parts)


# ─── Main service ──────────────────────────────────────────────────────────────

class VectorizerService:
    """
    Converts raster images (PNG/JPG) to SVG vector format.

    Pipeline:
      1. Color quantization (PIL median-cut or K-means)
      2. Per-color layer extraction (OpenCV inRange contours)
      3. Path simplification (RDP algorithm)
      4. Bezier smoothing (Catmull-Rom splines)
      5. SVG assembly with grouped layers
      6. Side-car JSON metadata

    Exports:
      output.svg         — full SVG with all layers
      output_layers.json — layer metadata (colors, paths, areas)
    """

    def __init__(
        self,
        default_colors: int = 16,
        default_simplify_eps: float = 1.5,
        default_bezier_smooth: float = 0.25,
    ):
        self.default_colors = max(2, min(default_colors, 256))
        self.default_simplify_eps = default_simplify_eps
        self.default_bezier_smooth = default_bezier_smooth

    def vectorize(
        self,
        image: Image.Image,
        color_quantization: int = 16,
        simplify_tolerance: float = 1.5,
        bezier_smoothness: float = 0.25,
        output_dir: Optional[str] = None,
        filename_base: str = "output",
    ) -> Dict:
        """
        Main vectorization entry point.

        Args:
            image: PIL Image
            color_quantization: Number of color layers (2–256)
            simplify_tolerance: RDP epsilon (0.5–5.0, lower = more detail)
            bezier_smoothness: Bezier smoothing (0 = sharp, 1 = very smooth)
            output_dir: Optional directory to save output.svg + output_layers.json
            filename_base: Base name for output files

        Returns:
            {
                "svg_content": str,
                "png_base64": str,
                "layers_json": dict,
                "metadata": {...}
            }
        """
        t0 = time.time()
        color_quantization = max(2, min(color_quantization, 256))

        # Normalize image
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")
        elif image.mode == "RGBA":
            # Composite on white background
            bg = Image.new("RGB", image.size, (255, 255, 255))
            bg.paste(image, mask=image.split()[3])
            image = bg

        w, h = image.size

        # Detect dominant colors
        dominant_colors = _detect_dominant_colors(image, color_quantization)

        layers: List[Dict] = []
        total_pixels = w * h

        for i, color in enumerate(dominant_colors):
            contours = _extract_color_contours(image, color, tolerance=20, min_area=4.0)
            if not contours:
                continue

            # Compute pixel coverage
            mask = np.zeros((h, w), dtype=np.uint8)
            for pts in contours:
                arr = np.array([pts], dtype=np.int32)
                cv2.fillPoly(mask, arr, 255)
            pixel_count = int(np.count_nonzero(mask))

            # Skip near-background layers with very small coverage
            if pixel_count < total_pixels * 0.001:
                continue

            # Simplify paths
            simplified_paths = []
            for pts in contours:
                simp = _rdp_simplify(pts, epsilon=simplify_tolerance)
                if len(simp) >= 3:
                    simplified_paths.append(simp)

            if simplified_paths:
                layers.append({
                    "layer_id": i,
                    "color": color,
                    "color_hex": _rgb_to_hex(*color),
                    "paths": simplified_paths,
                    "num_paths": len(simplified_paths),
                    "pixel_count": pixel_count,
                    "coverage_pct": round(pixel_count / total_pixels * 100, 3),
                })

        if not layers:
            logger.warning("No contours extracted — image may be too uniform")
            # Fallback: try grayscale contour extraction
            contours, bg_val = _extract_grayscale_contours(image, bg_white=True)
            if contours:
                simplified_paths = [
                    _rdp_simplify(p, simplify_tolerance)
                    for p in contours
                    if len(_rdp_simplify(p, simplify_tolerance)) >= 3
                ]
                layers.append({
                    "layer_id": 0,
                    "color": (0, 0, 0),
                    "color_hex": "#000000",
                    "paths": simplified_paths,
                    "num_paths": len(simplified_paths),
                    "pixel_count": sum(len(p) for p in simplified_paths),
                    "coverage_pct": 50.0,
                })

        # Determine background color (most common, usually largest layer)
        bg_color = "#ffffff"
        if layers:
            largest = max(layers, key=lambda l: l["pixel_count"])
            bg_color = largest.get("color_hex", "#ffffff")

        # Generate SVG
        svg_content = _generate_svg(
            layers, w, h,
            simplify_eps=simplify_tolerance,
            bezier_smooth=bezier_smoothness,
            bg_color=bg_color,
        )

        # Quantized PNG for reference
        quantized_pil, _ = _quantize_pil(image, color_quantization)
        buf = io.BytesIO()
        quantized_pil.save(buf, format="PNG")
        png_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        layers_meta = {
            "version": "1.0",
            "source_size": {"width": w, "height": h},
            "num_colors": len(layers),
            "background": bg_color,
            "color_quantization": color_quantization,
            "layers": [
                {
                    "layer_id": l["layer_id"],
                    "color_hex": l["color_hex"],
                    "num_paths": l["num_paths"],
                    "pixel_count": l["pixel_count"],
                    "coverage_pct": l["coverage_pct"],
                }
                for l in layers
            ],
        }

        # Save output files
        saved_files = {}
        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            svg_path = out_path / f"{filename_base}.svg"
            svg_path.write_text(svg_content, encoding="utf-8")
            saved_files["svg"] = str(svg_path)

            json_path = out_path / f"{filename_base}_layers.json"
            json_path.write_text(json.dumps(layers_meta, indent=2, ensure_ascii=False), encoding="utf-8")
            saved_files["layers_json"] = str(json_path)

            # Also save the quantized PNG
            png_path = out_path / f"{filename_base}_quantized.png"
            quantized_pil.save(png_path)
            saved_files["png"] = str(png_path)

            logger.info(
                "Vectorization complete: %d layers, %d paths → %s",
                len(layers), sum(l["num_paths"] for l in layers), svg_path,
            )

        elapsed = time.time() - t0

        return {
            "svg_content": svg_content,
            "png_base64": png_base64,
            "layers_json": layers_meta,
            "metadata": {
                "original_size": (w, h),
                "num_colors": len(layers),
                "total_paths": sum(l["num_paths"] for l in layers),
                "background": bg_color,
                "color_quantization": color_quantization,
                "saved_files": saved_files,
                "elapsed_seconds": round(elapsed, 2),
            },
        }

    def vectorize_from_bytes(
        self,
        image_bytes: bytes,
        **kwargs,
    ) -> Dict:
        """Convenience: vectorize from raw image bytes."""
        image = Image.open(io.BytesIO(image_bytes))
        return self.vectorize(image, **kwargs)

    def vectorize_from_path(
        self,
        image_path: str,
        **kwargs,
    ) -> Dict:
        """Convenience: vectorize from file path."""
        image = Image.open(image_path)
        return self.vectorize(image, **kwargs)


# ─── Module-level singleton ────────────────────────────────────────────────────

_vectorizer_service: Optional[VectorizerService] = None


def get_vectorizer_service() -> VectorizerService:
    global _vectorizer_service
    if _vectorizer_service is None:
        _vectorizer_service = VectorizerService()
    return _vectorizer_service


# ─── Import base64 lazily (used only in this module) ─────────────────────────
import base64
