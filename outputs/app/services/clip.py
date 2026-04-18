"""
CLIP Service - Text-Image similarity scoring
Uses OpenAI CLIP ViT-L/14 for encoding and similarity computation
"""
import os
import torch
from PIL import Image
from typing import List
import logging
from transformers import CLIPProcessor, CLIPModel

logger = logging.getLogger(__name__)


class CLIPService:
    _model = None
    _processor = None
    
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):
        """
        Initialize CLIP Service for text-image alignment scoring.
        """
        self.model_name = model_name

        # Respect FORCE_DEVICE env var, else auto-detect
        force = os.getenv("FORCE_DEVICE", "").lower()
        if force == "cpu":
            self.device = "cpu"
        elif force == "cuda":
            self.device = "cuda"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
    
    @property
    def model(self):
        """Lazy load model"""
        if CLIPService._model is None:
            self._load_model()
        return CLIPService._model
    
    @property
    def processor(self):
        """Lazy load processor"""
        if CLIPService._processor is None:
            self._load_model()
        return CLIPService._processor
    
    def _load_model(self):
        """Load CLIP model with correct dtype for device."""
        try:
            logger.info(f"Loading CLIP model: {self.model_name} on {self.device} (dtype={self.dtype})")
            CLIPService._model = CLIPModel.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
            )
            CLIPService._processor = CLIPProcessor.from_pretrained(self.model_name)
            CLIPService._model = CLIPService._model.to(self.device)
            CLIPService._model.eval()
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLIP: {e}")
            raise
    
    def compute_score(self, image: Image.Image, text: str) -> float:
        """
        Compute CLIP similarity score between image and text.
        
        Args:
            image: PIL Image
            text: Text description
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        inputs = self.processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            score = probs[0][0].item()
        
        return score
    
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode image to CLIP embedding"""
        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_embeds = self.model.get_image_features(**inputs)
        
        return image_embeds
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text to CLIP embedding"""
        inputs = self.processor(
            text=[text],
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_embeds = self.model.get_text_features(**inputs)
        
        return text_embeds
    
    def compute_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Compute cosine similarity between two embeddings"""
        cos_sim = torch.nn.functional.cosine_similarity(emb1, emb2, dim=-1)
        return cos_sim.item()
