"""
Vectorizer Service - PNG/JPG to SVG conversion
Uses Potrace for bitmap tracing with color quantization
"""
import logging
import io
import base64
import numpy as np
from PIL import Image
from typing import Dict, Optional, List, Tuple

logger = logging.getLogger(__name__)


class VectorizerService:
    def __init__(self):
        """Initialize Vectorizer Service"""
        self.default_colors = 16
        self.default_tolerance = 1.0
    
    def vectorize(
        self,
        image: Image.Image,
        color_quantization: int = 16,
        simplify_tolerance: float = 1.0
    ) -> Dict:
        """
        Convert raster image to SVG vector format.
        
        Args:
            image: PIL Image
            color_quantization: Number of colors to reduce to
            simplify_tolerance: Path simplification tolerance
            
        Returns:
            Dict with SVG content, PNG base64, layers info
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        quantized = self._quantize_colors(image, color_quantization)
        
        layers = self._extract_layers(quantized)
        
        svg_content = self._create_svg(quantized, layers, simplify_tolerance)
        
        buffered = io.BytesIO()
        quantized.save(buffered, format="PNG")
        png_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {
            "svg_content": svg_content,
            "png_base64": png_base64,
            "layers_json": layers,
            "original_size": image.size,
            "num_colors": color_quantization,
            "num_paths": sum(len(layer["paths"]) for layer in layers)
        }
    
    def _quantize_colors(self, image: Image.Image, num_colors: int) -> Image.Image:
        """Reduce image to specified number of colors"""
        quantized = image.convert("P", palette=Image.ADAPTIVE, colors=num_colors)
        return quantized
    
    def _extract_layers(self, image: Image.Image) -> list:
        """Extract color layers for separate path generation"""
        layers = []
        
        palette = image.getpalette()
        if palette is None:
            return layers
        
        pixels = np.array(image)
        unique_colors = np.unique(pixels.reshape(-1), axis=0)
        
        for i, color_idx in enumerate(unique_colors[:32]):
            mask = (pixels == color_idx).all(axis=-1) if len(pixels.shape) > 2 else pixels == color_idx
            
            if mask.sum() > 100:
                color = (
                    palette[int(color_idx) * 3],
                    palette[int(color_idx) * 3 + 1],
                    palette[int(color_idx) * 3 + 2]
                )
                contours = self._find_contours(mask.astype(np.uint8) * 255)
                
                layers.append({
                    "color": color,
                    "color_index": int(color_idx),
                    "paths": contours,
                    "pixel_count": int(mask.sum())
                })
        
        return layers
    
    def _find_contours(self, binary_mask: np.ndarray) -> List[List[Tuple[int, int]]]:
        """Find contours in binary mask using OpenCV"""
        try:
            import cv2
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            path_data = []
            for cnt in contours:
                if cv2.contourArea(cnt) > 10:
                    epsilon = 0.01 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    points = [(int(p[0][0]), int(p[0][1])) for p in approx]
                    path_data.append(points)
            return path_data
        except ImportError:
            logger.warning("OpenCV not available for contour detection")
            return []
    
    def _create_svg(self, image: Image.Image, layers: List[Dict], tolerance: float) -> str:
        """Generate SVG from color layers"""
        width, height = image.size
        
        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            f'<rect width="100%" height="100%" fill="white"/>'
        ]
        
        for layer in layers:
            color = layer["color"]
            color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            
            svg_parts.append(f'<g fill="{color_hex}" stroke="{color_hex}" stroke-width="1">')
            
            for path_points in layer["paths"]:
                if len(path_points) >= 3:
                    path_d = self._points_to_path(path_points, tolerance)
                    svg_parts.append(f'  <path d="{path_d}"/>')
                elif len(path_points) == 2:
                    svg_parts.append(f'  <line x1="{path_points[0][0]}" y1="{path_points[0][1]}" x2="{path_points[1][0]}" y2="{path_points[1][1]}"/>')
            
            svg_parts.append('</g>')
        
        svg_parts.append('</svg>')
        return '\n'.join(svg_parts)
    
    def _points_to_path(self, points: List[Tuple[int, int]], tolerance: float) -> str:
        """Convert list of points to SVG path data using Bezier curves"""
        if len(points) < 2:
            return ""
        
        if len(points) == 2:
            return f"M {points[0][0]} {points[0][1]} L {points[1][0]} {points[1][1]}"
        
        path = f"M {points[0][0]} {points[0][1]}"
        
        for i in range(1, len(points) - 1):
            x0, y0 = points[i - 1]
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            
            cp1x = x1 - (x2 - x0) * tolerance * 0.1
            cp1y = y1 - (y2 - y0) * tolerance * 0.1
            cp2x = x1 + (x2 - x0) * tolerance * 0.1
            cp2y = y1 + (y2 - y0) * tolerance * 0.1
            
            path += f" Q {x1} {y1} {cp2x} {cp2y}"
        
        path += f" L {points[-1][0]} {points[-1][1]}"
        path += " Z"
        
        return path
