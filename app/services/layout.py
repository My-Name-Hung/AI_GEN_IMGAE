"""
Layout Service - OpenCV-based image layout analysis
Detects text regions, icons, and suggests composition layouts
"""
import logging
import numpy as np
import cv2
from PIL import Image
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


class LayoutService:
    def __init__(self):
        """Initialize Layout Analysis Service"""
        self.thirds_width_ratio = 1/3
        self.thirds_height_ratio = 1/3
    
    def analyze(
        self,
        image: Image.Image,
        detect_text: bool = True,
        detect_icons: bool = True,
        suggest_layout: bool = True
    ) -> Dict:
        """
        Analyze image layout and detect regions.
        
        Args:
            image: PIL Image
            detect_text: Enable text region detection
            detect_icons: Enable icon detection
            suggest_layout: Generate layout suggestions
            
        Returns:
            Layout analysis dict with regions, texts, icons
        """
        img_array = np.array(image.convert("RGB"))
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        regions = []
        texts = []
        icons = []
        
        if detect_text:
            text_regions = self._detect_text_regions(gray, img_cv)
            texts.extend(text_regions)
            regions.extend(text_regions)
        
        if detect_icons:
            icon_regions = self._detect_icon_regions(img_cv)
            icons.extend(icon_regions)
            regions.extend(icon_regions)
        
        if suggest_layout:
            layout_suggestion = self._generate_layout_suggestion(img_cv)
        else:
            layout_suggestion = {}
        
        return {
            "regions": regions,
            "texts": texts,
            "icons": icons,
            "layout": layout_suggestion,
            "image_size": {"width": img_cv.shape[1], "height": img_cv.shape[0]}
        }
    
    def _detect_text_regions(self, gray: np.ndarray, color_img: np.ndarray) -> List[Dict]:
        """Detect text-like regions using edge detection and contour analysis"""
        regions = []
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h) if h > 0 else 0
            
            if w > 50 and h > 20 and 0.2 < aspect_ratio < 10:
                regions.append({
                    "type": "text",
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "center": [int(x + w/2), int(y + h/2)],
                    "confidence": 0.7
                })
        
        return regions
    
    def _detect_icon_regions(self, img: np.ndarray) -> List[Dict]:
        """Detect icon-like regions using color and shape analysis"""
        regions = []
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 500 < area < 50000:
                x, y, w, h = cv2.boundingRect(cnt)
                solidity = area / cv2.contourArea(cv2.convexHull(cnt)) if cv2.contourArea(cv2.convexHull(cnt)) > 0 else 0
                
                if solidity > 0.5:
                    regions.append({
                        "type": "icon",
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "center": [int(x + w/2), int(y + h/2)],
                        "area": float(area),
                        "confidence": float(solidity)
                    })
        
        return regions
    
    def _generate_layout_suggestion(self, img: np.ndarray) -> Dict:
        """Generate layout suggestions using rule of thirds"""
        h, w = img.shape[:2]
        
        thirds = []
        for i in range(1, 3):
            thirds.append({
                "vertical_line": int(w * i / 3),
                "position": "left" if i == 1 else "right"
            })
        for i in range(1, 3):
            thirds.append({
                "horizontal_line": int(h * i / 3),
                "position": "top" if i == 1 else "bottom"
            })
        
        intersections = [
            {"x": int(w/3), "y": int(h/3), "name": "top-left"},
            {"x": int(2*w/3), "y": int(h/3), "name": "top-right"},
            {"x": int(w/3), "y": int(2*h/3), "name": "bottom-left"},
            {"x": int(2*w/3), "y": int(2*h/3), "name": "bottom-right"}
        ]
        
        return {
            "rule_of_thirds": {
                "lines": thirds,
                "intersections": intersections
            },
            "center": {"x": w//2, "y": h//2},
            "margins": {
                "safe_zone": int(min(w, h) * 0.1),
                "inner_margin": int(min(w, h) * 0.05)
            }
        }
    
    def suggest_contrast_zones(self, image: Image.Image) -> List[Dict]:
        """Identify high-contrast zones for text overlay"""
        img_array = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        zones = []
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 30 and h > 30:
                mean_intensity = np.mean(gray[y:y+h, x:x+w])
                contrast = abs(128 - mean_intensity) / 128
                zones.append({
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "mean_intensity": float(mean_intensity),
                    "contrast": float(contrast),
                    "suitable_for_text": contrast > 0.2
                })
        
        return zones
