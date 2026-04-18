"""
Layout Service - OpenCV-based image layout analysis.
Detects text regions, icon/logo regions, background zones.
Suggests composition guides: rule-of-thirds, contrast zones, safe margins.
"""
import logging
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ─── Layout presets ──────────────────────────────────────────────────────────

ASPECT_PRESETS = {
    "1:1": {"width": 1024, "height": 1024},
    "4:3": {"width": 1024, "height": 768},
    "16:9": {"width": 1216, "height": 832},
    "9:16": {"width": 832, "height": 1216},
    "3:2": {"width": 1056, "height": 704},
    "2:3": {"width": 704, "height": 1056},
}


# ─── Main service ──────────────────────────────────────────────────────────────

class LayoutService:
    """
    Analyzes image layout: detects text/icon/background regions,
    generates composition guides, and outputs a structured JSON schema.

    Fully offline — uses only OpenCV + NumPy + PIL.
    """

    def __init__(self):
        self._cv2_available = self._check_cv2()
        if not self._cv2_available:
            logger.warning("OpenCV (cv2) not available — layout analysis disabled")
        self._text_detector = _TextRegionHeuristic()
        self._icon_detector = _IconRegionHeuristic()
        self._contrast_finder = _ContrastZoneFinder()

    @staticmethod
    def _check_cv2() -> bool:
        try:
            cv2.__version__
            return True
        except Exception:
            return False

    # ─── Public API ────────────────────────────────────────────────────────────
    
    def analyze(
        self,
        image: Image.Image,
        detect_text: bool = True,
        detect_icons: bool = True,
        suggest_layout: bool = True,
        output_path: Optional[str] = None,
    ) -> Dict:
        """
        Full layout analysis pipeline.
        
        Args:
            image: PIL Image (any mode)
            detect_text: Enable text region detection
            detect_icons: Enable icon/logo region detection
            suggest_layout: Generate composition guides
            output_path: Optional path to save annotated debug image
            
        Returns:
            Layout schema dict:
            {
                "regions": [...],
                "texts": [...],
                "icons": [...],
                "background_zones": [...],
                "layout": { "rule_of_thirds": {...}, "margins": {...}, "grid": {...} },
                "contrast_zones": [...],
                "image_size": {"width": int, "height": int},
                "summary": str
            }
        """
        if not self._cv2_available:
            return {"error": "OpenCV not available", "regions": [], "texts": [], "icons": []}

        img_rgb = np.array(image.convert("RGB"))
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        h, w = img_gray.shape

        regions: List[Dict] = []
        texts: List[Dict] = []
        icons: List[Dict] = []
        
        if detect_text:
            texts = self._text_detector.find(img_gray, img_rgb, w, h)
            regions.extend(texts)
        
        if detect_icons:
            icons = self._icon_detector.find(img_rgb, img_gray, img_hsv, w, h)
            regions.extend(icons)

        # Background detection: largest uniform region (usually background)
        bg_zone = self._detect_background(img_rgb, img_gray, w, h)
        bg_zones = [bg_zone] if bg_zone else []

        contrast_zones = []
        if suggest_layout:
            contrast_zones = self._contrast_finder.find(img_rgb, img_gray, w, h)

        layout = self._generate_layout(h, w, regions, contrast_zones) if suggest_layout else {}

        result = {
            "regions": regions,
            "texts": texts,
            "icons": icons,
            "background_zones": bg_zones,
            "layout": layout,
            "contrast_zones": contrast_zones,
            "image_size": {"width": w, "height": h},
            "summary": self._summarize(regions, texts, icons, h, w),
        }

        if output_path:
            self._save_debug_visualization(result, img_rgb, output_path)

        return result

    def analyze_from_path(self, image_path: str, **kwargs) -> Dict:
        """Convenience: load image from file path and analyze."""
        try:
            image = Image.open(image_path).convert("RGB")
            return self.analyze(image, **kwargs)
        except Exception as e:
            logger.error("Failed to load image from %s: %s", image_path, e)
            return {"error": str(e), "regions": [], "texts": [], "icons": []}

    def suggest_composition(
        self,
        image: Image.Image,
        mode: str = "centered",
    ) -> Dict:
        """
        Suggest text/logo placement positions for an empty canvas.

        Args:
            image: PIL Image (the canvas, used only for size)
            mode: "centered" | "thirds" | "diagonal" | "split"

        Returns:
            Placement guide dict with x, y, width, height, alignment, label
        """
        w, h = image.size
        sw, sh = self._safe_margin(w, h)

        layouts = {
            "centered": [
                {"x": sw, "y": sh, "width": w - 2 * sw, "height": h - 2 * sh,
                 "alignment": "center", "label": "centered_content"},
            ],
            "thirds": [
                {"x": sw, "y": sh, "width": w // 3 - sw, "height": h - 2 * sh,
                 "alignment": "left", "label": "thirds_left"},
                {"x": w * 2 // 3, "y": sh, "width": w // 3 - sw, "height": h - 2 * sh,
                 "alignment": "right", "label": "thirds_right"},
                {"x": w // 3, "y": h // 3, "width": w // 3, "height": h // 3,
                 "alignment": "center", "label": "thirds_center"},
            ],
            "diagonal": [
                {"x": sw, "y": sh, "width": w * 2 // 3 - sw, "height": h - 2 * sh,
                 "alignment": "left", "label": "diagonal_main"},
                {"x": w // 2, "y": h // 4, "width": w // 2 - sw, "height": h // 2,
                 "alignment": "right", "label": "diagonal_secondary"},
            ],
            "split": [
                {"x": sw, "y": sh, "width": w // 2 - sw * 2, "height": h - 2 * sh,
                 "alignment": "left", "label": "split_left"},
                {"x": w // 2 + sw, "y": sh, "width": w // 2 - sw * 2, "height": h - 2 * sh,
                 "alignment": "right", "label": "split_right"},
            ],
        }

        placements = layouts.get(mode, layouts["centered"])

        return {
            "mode": mode,
            "canvas_size": {"width": w, "height": h},
            "safe_margin": {"x": sw, "y": sh},
            "placements": placements,
        }

    def export_schema(self, analysis: Dict, output_path: str) -> str:
        """
        Save layout analysis as a JSON schema file.
        Includes all regions, composition guides, and metadata.
        """
        schema = {
            "version": "1.0",
            "type": "layout_schema",
            "image_size": analysis.get("image_size", {}),
            "regions": analysis.get("regions", []),
            "texts": analysis.get("texts", []),
            "icons": analysis.get("icons", []),
            "background_zones": analysis.get("background_zones", []),
            "contrast_zones": analysis.get("contrast_zones", []),
            "layout": analysis.get("layout", {}),
            "summary": analysis.get("summary", ""),
        }
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)
        logger.info("Layout schema saved to %s", path)
        return str(path)

    # ─── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _safe_margin(w: int, h: int) -> Tuple[int, int]:
        margin = int(min(w, h) * 0.05)
        return margin, margin

    def _generate_layout(
        self,
        h: int,
        w: int,
        regions: List[Dict],
        contrast_zones: List[Dict],
    ) -> Dict:
        """Build composition guide: rule-of-thirds + safe margins."""
        thirds_x = [w // 3, 2 * w // 3]
        thirds_y = [h // 3, 2 * h // 3]

        intersections = [
            {"x": thirds_x[0], "y": thirds_y[0], "name": "top_left",  "weight": 1.2},
            {"x": thirds_x[1], "y": thirds_y[0], "name": "top_right", "weight": 1.0},
            {"x": thirds_x[0], "y": thirds_y[1], "name": "bot_left",  "weight": 1.0},
            {"x": thirds_x[1], "y": thirds_y[1], "name": "bot_right", "weight": 1.1},
        ]

        grid_lines = {
            "vertical": [{"x": tx, "ratio": i + 1} for i, tx in enumerate(thirds_x)],
            "horizontal": [{"y": ty, "ratio": i + 1} for i, ty in enumerate(thirds_y)],
        }

        # Find best text overlay zone (high contrast near center)
        best_text_zone = None
        if contrast_zones:
            center_x, center_y = w / 2, h / 2
            best = min(
                contrast_zones,
                key=lambda z: abs(z["bbox"][0] - center_x) + abs(z["bbox"][1] - center_y),
                default=None,
            )
            if best:
                best_text_zone = {
                    "bbox": best["bbox"],
                    "contrast": best["contrast"],
                    "reason": "nearest_to_center",
                }

        return {
            "rule_of_thirds": {
                "lines": grid_lines,
                "intersections": intersections,
            },
            "center": {"x": w // 2, "y": h // 2},
            "margins": {
                "safe_zone": int(min(w, h) * 0.05),
                "inner_margin": int(min(w, h) * 0.02),
                "bleed": int(min(w, h) * 0.01),
            },
            "recommended_text_zone": best_text_zone,
            "aspect_ratios": ASPECT_PRESETS,
        }

    @staticmethod
    def _detect_background(
        img_rgb: np.ndarray,
        img_gray: np.ndarray,
        w: int,
        h: int,
    ) -> Optional[Dict]:
        """Detect the dominant background region using K-means on 2x2 grid."""
        try:
            zones = []
            step_x, step_y = w // 4, h // 4
            for row in range(2):
                for col in range(2):
                    x0, y0 = col * step_x + step_x, row * step_y + step_y
                    x1, y1 = x0 + step_x, y0 + step_y
                    patch = img_gray[y0:y1, x0:x1]
                    if patch.size > 0:
                        mean_val = float(np.mean(patch))
                        std_val = float(np.std(patch))
                        zones.append({
                            "x": int(x0), "y": int(y0),
                            "w": int(step_x), "h": int(step_y),
                            "mean_brightness": mean_val,
                            "std_brightness": std_val,
                            "is_uniform": std_val < 20,
                        })

            # Find most uniform zone → likely background
            uniform = [z for z in zones if z["is_uniform"]]
            if uniform:
                bg = min(uniform, key=lambda z: z["std_brightness"])
                return {
                    "bbox": [bg["x"], bg["y"], bg["w"], bg["h"]],
                    "mean_brightness": bg["mean_brightness"],
                    "std_brightness": bg["std_brightness"],
                }
        except Exception as e:
            logger.warning("Background detection failed: %s", e)
        return None

    @staticmethod
    def _summarize(
        regions: List[Dict],
        texts: List[Dict],
        icons: List[Dict],
        h: int,
        w: int,
    ) -> str:
        total = len(regions)
        text_count = len(texts)
        icon_count = len(icons)
        bg_count = total - text_count - icon_count
        ratio = round(w / max(h, 1), 2)
        return (
            f"Image {w}×{h} (ratio {ratio}). "
            f"Detected {total} regions: {text_count} text, "
            f"{icon_count} icon/logo, {bg_count} background/other."
        )

    def _save_debug_visualization(
        self,
        analysis: Dict,
        img_rgb: np.ndarray,
        output_path: str,
    ) -> None:
        """Draw layout guides on image for debugging."""
        try:
            vis = img_rgb.copy()
            h, w = vis.shape[:2]

            # Draw thirds grid
            for tx in [w // 3, 2 * w // 3]:
                cv2.line(vis, (tx, 0), (tx, h), (0, 200, 0), 1)
            for ty in [h // 3, 2 * h // 3]:
                cv2.line(vis, (0, ty), (w, ty), (0, 200, 0), 1)

            # Draw intersection points
            for inter in analysis["layout"].get("rule_of_thirds", {}).get("intersections", []):
                cx, cy = inter["x"], inter["y"]
                cv2.circle(vis, (cx, cy), 8, (0, 255, 0), -1)
                cv2.putText(vis, inter["name"], (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # Draw text regions
            for r in analysis["texts"]:
                x, y, bw, bh = r["bbox"]
                cv2.rectangle(vis, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
                cv2.putText(vis, "T", (x + 3, y + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Draw icon regions
            for r in analysis["icons"]:
                x, y, bw, bh = r["bbox"]
                cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 165, 255), 2)
                cv2.putText(vis, "I", (x + 3, y + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

            # Draw contrast zones
            for zone in analysis.get("contrast_zones", []):
                x, y, bw, bh = zone["bbox"]
                alpha = int(min(zone["contrast"] * 100, 80))
                color = (0, 255 - alpha, alpha) if alpha < 128 else (255 - alpha, 0, alpha)
                cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, 1)

            cv2.imwrite(str(output_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            logger.info("Debug visualization saved to %s", output_path)
        except Exception as e:
            logger.warning("Failed to save debug visualization: %s", e)


# ─── Text region heuristic detector ─────────────────────────────────────────

class _TextRegionHeuristic:
    """
    Detects text-like regions using:
    - Horizontal edge density (text has strong horizontal edges)
    - Aspect ratio (text boxes are usually wider than tall)
    - Vertical projection profile
    """

    def find(
        self,
        gray: np.ndarray,
        color: np.ndarray,
        w: int,
        h: int,
    ) -> List[Dict]:
        results = []

        # Method 1: Horizontal Sobel → find horizontal edge bands
        sobel_h = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        abs_sobel_h = np.abs(sobel_h)
        norm = cv2.normalize(abs_sobel_h, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        thresh = cv2.threshold(norm, 30, 255, cv2.THRESH_BINARY)[1]

        # Find horizontal runs → text rows
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 8, 3))
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw > 40 and ch > 8 and 0.3 < (cw / max(ch, 1)) < 20:
                contrast = self._local_contrast(gray, x, y, cw, ch)
                results.append({
                    "type": "text",
                    "bbox": [int(x), int(y), int(cw), int(ch)],
                    "center": [int(x + cw / 2), int(y + ch / 2)],
                    "aspect_ratio": round(cw / max(ch, 1), 2),
                    "contrast": round(contrast, 3),
                    "confidence": round(min(contrast * 2, 0.95), 3),
                    "detection_method": "horizontal_edges",
                })

        # Method 2: Vertical projection — find low-variance rows (uniform text)
        proj = np.var(gray, axis=1)
        row_thresh = np.percentile(proj, 30) if len(proj) > 0 else 0
        text_rows = np.where(proj < row_thresh)[0]

        if len(text_rows) > 5:
            row_groups = np.split(text_rows, np.where(np.diff(text_rows) > 5)[0] + 1)
            for group in row_groups:
                if len(group) < 3:
                    continue
                ry1, ry2 = int(group[0]), int(group[-1])
                col_thresh = np.percentile(np.var(gray[ry1:ry2 + 1, :], axis=0), 60) if (ry2 - ry1) > 0 else 0
                text_cols = np.where(np.var(gray[ry1:ry2 + 1, :], axis=0) > col_thresh)[0]
                if len(text_cols) > 10:
                    cx1 = int(text_cols[0])
                    cx2 = int(text_cols[-1])
                    bbox = [cx1, ry1, cx2 - cx1, ry2 - ry1]
                    if bbox[2] > 30 and bbox[3] > 6:
                        results.append({
                            "type": "text",
                            "bbox": bbox,
                            "center": [cx1 + (cx2 - cx1) // 2, ry1 + (ry2 - ry1) // 2],
                            "aspect_ratio": round((cx2 - cx1) / max(ry2 - ry1, 1), 2),
                            "confidence": 0.65,
                            "detection_method": "projection_profile",
                        })

        # Deduplicate overlapping boxes
        return self._deduplicate(results)

    @staticmethod
    def _local_contrast(gray: np.ndarray, x: int, y: int, bw: int, bh: int) -> float:
        """Measure local contrast inside a bounding box."""
        x0, y0 = max(0, x - 2), max(0, y - 2)
        x1, y1 = min(gray.shape[1], x + bw + 2), min(gray.shape[0], y + bh + 2)
        patch = gray[y0:y1, x0:x1]
        if patch.size == 0:
            return 0.0
        return float(np.std(patch) / 128.0)

    @staticmethod
    def _deduplicate(boxes: List[Dict]) -> List[Dict]:
        """Merge boxes that overlap > 50%."""
        if not boxes:
            return []
        sorted_boxes = sorted(boxes, key=lambda b: b["bbox"][2] * b["bbox"][3], reverse=True)
        kept = []
        for box in sorted_boxes:
            x1, y1, w1, h1 = box["bbox"]
            overlap = False
            for k in kept:
                x2, y2, w2, h2 = k["bbox"]
                ox = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                oy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                inter = ox * oy
                union = w1 * h1 + w2 * h2 - inter
                if union > 0 and (inter / union) > 0.5:
                    overlap = True
                    break
            if not overlap:
                kept.append(box)
        return kept


# ─── Icon / logo region detector ─────────────────────────────────────────────

class _IconRegionHeuristic:
    """
    Detects icon/logo regions using:
    - Solidity (ratio of area to convex hull area)
    - Bounded size range (icons are not too big or too small)
    - Color blob detection via contour analysis
    """

    def find(
        self,
        color: np.ndarray,
        gray: np.ndarray,
        hsv: np.ndarray,
        w: int,
        h: int,
    ) -> List[Dict]:
        results = []

        # Method 1: Adaptive threshold → find solid blobs
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 5,
        )
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        img_area = w * h
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (img_area * 0.001 < area < img_area * 0.5):
                continue
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            if solidity < 0.3:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            # Skip very rectangular (likely text) regions
            ar = bw / max(bh, 1)
            if 0.3 < ar < 10:
                continue
            results.append({
                "type": "icon",
                "bbox": [int(x), int(y), int(bw), int(bh)],
                "center": [int(x + bw / 2), int(y + bh / 2)],
                "area": round(float(area), 1),
                "solidity": round(float(solidity), 3),
                "confidence": round(float(solidity * 0.8), 3),
                "detection_method": "adaptive_threshold",
            })

        # Method 2: Color blob detection via K-means in color space
        small = cv2.resize(color, (w // 2, h // 2))
        pixels = small.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        try:
            _, labels, centers = cv2.kmeans(pixels, 8, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
            labels = labels.reshape(small.shape[:2])
            for i, center in enumerate(centers.astype(int)):
                mask = (labels == i).astype(np.uint8) * 255
                contours2, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours2:
                    area = cv2.contourArea(cnt)
                    if area < img_area * 0.001 or area > img_area * 0.3:
                        continue
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    if bw < 20 or bh < 20:
                        continue
                    results.append({
                        "type": "icon",
                        "bbox": [int(x * 2), int(y * 2), int(bw * 2), int(bh * 2)],
                        "center": [int((x + bw // 2) * 2), int((y + bh // 2) * 2)],
                        "color": [int(c) for c in center],
                        "area": round(float(area * 4), 1),
                        "confidence": 0.55,
                        "detection_method": "color_blob",
                    })
        except Exception as e:
            logger.warning("K-means blob detection failed: %s", e)

        return _IconRegionHeuristic._deduplicate(results)

    @staticmethod
    def _deduplicate(boxes: List[Dict]) -> List[Dict]:
        """Keep largest non-overlapping boxes."""
        if not boxes:
            return []
        sorted_boxes = sorted(boxes, key=lambda b: b.get("area", 0), reverse=True)
        kept = []
        for box in sorted_boxes:
            if box.get("area", 0) < 100:
                continue
            x1, y1, w1, h1 = box["bbox"]
            overlap = False
            for k in kept:
                x2, y2, w2, h2 = k["bbox"]
                ox = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                oy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                inter = ox * oy
                if inter > 0:
                    overlap = True
                    break
            if not overlap:
                kept.append(box)
        return kept


# ─── Contrast zone finder ─────────────────────────────────────────────────────

class _ContrastZoneFinder:
    """
    Finds high-contrast zones suitable for text overlay.
    Uses Laplacian variance for sharpness and local contrast.
    """

    def find(
        self,
        color: np.ndarray,
        gray: np.ndarray,
        w: int,
        h: int,
    ) -> List[Dict]:
        zones = []

        # Grid-based analysis
        grid_x, grid_y = 8, 8
        step_x = w // grid_x
        step_y = h // grid_y

        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        laplacian = cv2.Laplacian(blurred, ddepth=cv2.CV_64F)

        for gy in range(grid_y):
            for gx in range(grid_x):
                x0 = gx * step_x
                y0 = gy * step_y
                x1 = min(x0 + step_x, w)
                y1 = min(y0 + step_y, h)

                patch = gray[y0:y1, x0:x1]
                lap_patch = laplacian[y0:y1, x0:x1]

                if patch.size == 0:
                    continue

                mean_val = float(np.mean(patch))
                std_val = float(np.std(patch))
                lap_var = float(np.var(lap_patch))

                # Contrast: difference from neutral gray (128)
                contrast = abs(128 - mean_val) / 128.0
                # Sharpness bonus
                sharpness = min(lap_var / 5000.0, 1.0)
                combined = (contrast * 0.7 + sharpness * 0.3)

                if combined > 0.2 and std_val > 5:
                    zones.append({
                        "bbox": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
                        "center": [int(x0 + (x1 - x0) // 2), int(y0 + (y1 - y0) // 2)],
                        "mean_brightness": round(mean_val, 1),
                        "std_brightness": round(std_val, 1),
                        "contrast": round(contrast, 3),
                        "sharpness": round(sharpness, 3),
                        "combined_score": round(combined, 3),
                        "suitable_for_text": combined > 0.35,
                    })

        # Sort by combined score and keep top zones
        zones.sort(key=lambda z: z["combined_score"], reverse=True)
        return zones[:12]


# ─── Module-level singleton ───────────────────────────────────────────────────

_layout_service: Optional[LayoutService] = None


def get_layout_service() -> LayoutService:
    global _layout_service
    if _layout_service is None:
        _layout_service = LayoutService()
    return _layout_service
