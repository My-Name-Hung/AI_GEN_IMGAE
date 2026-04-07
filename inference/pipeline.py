"""
Inference Pipeline - End-to-end inference script
Combines diffusion, CLIP scoring, and vectorization
"""
import argparse
import logging
import base64
import io
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image
import torch
import numpy as np

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    End-to-end inference pipeline for logo/poster generation.
    
    Pipeline stages:
    1. Prompt processing
    2. Image generation (SDXL Turbo)
    3. CLIP score filtering
    4. Optional layout injection
    5. Optional vectorization
    """
    
    def __init__(
        self,
        model_name: str = "stabilityai/sdxl-turbo",
        clip_model: str = "openai/clip-vit-large-patch14"
    ):
        """
        Initialize inference pipeline.
        
        Args:
            model_name: Stable Diffusion model
            clip_model: CLIP model for scoring
        """
        self.model_name = model_name
        self.clip_model_name = clip_model
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.diffusion_pipeline = None
        self.clip_model = None
        self.clip_processor = None
        
        self._load_models()
    
    def _load_models(self):
        """Load all models"""
        from diffusers import AutoPipelineForText2Image
        from transformers import CLIPModel, CLIPProcessor
        
        logger.info("Loading diffusion model...")
        self.diffusion_pipeline = AutoPipelineForText2Image.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype
        )
        self.diffusion_pipeline = self.diffusion_pipeline.to(self.device)
        
        if self.device == "cuda":
            self.diffusion_pipeline.enable_xformers_memory_efficient_attention()
        
        logger.info("Loading CLIP model...")
        self.clip_model = CLIPModel.from_pretrained(self.clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()
        
        logger.info("Models loaded successfully")
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        seed: Optional[int] = None,
        num_images: int = 1,
        min_clip_score: float = 0.0,
        enable_vectorization: bool = False,
        output_dir: str = "./output"
    ) -> Dict:
        """
        Generate images with optional CLIP filtering and vectorization.
        
        Args:
            prompt: Text description
            negative_prompt: Things to avoid
            width/height: Output dimensions
            num_inference_steps: Denoising steps
            guidance_scale: CFG scale
            seed: Random seed
            num_images: Number to generate
            min_clip_score: Minimum CLIP score threshold
            enable_vectorization: Enable SVG output
            output_dir: Save directory
            
        Returns:
            Generation results with images, scores, optional SVG
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if seed is None:
            import random
            # Use int32-safe range for cross-platform compatibility
            seed = random.randint(0, 2**31 - 1)
        
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        logger.info(f"Generating {num_images} image(s) with prompt: {prompt}")
        
        images = []
        clip_scores = []
        
        for i in range(num_images):
            logger.info(f"Generating image {i+1}/{num_images}")
            
            result = self.diffusion_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            )
            
            img = result.images[0]
            
            clip_score = self._compute_clip_score(img, prompt)
            clip_scores.append(clip_score)
            
            if clip_score >= min_clip_score:
                images.append(img)
                img.save(output_path / f"generated_{seed}_{i+1}.png")
                logger.info(f"Image {i+1} CLIP score: {clip_score:.4f}")
        
        result_data = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "num_generated": len(images),
            "images": [self._image_to_base64(img) for img in images],
            "clip_scores": clip_scores,
            "metadata": {
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale
            }
        }
        
        if enable_vectorization and images:
            logger.info("Vectorizing best image...")
            best_idx = np.argmax(clip_scores)
            svg_result = self._vectorize_image(images[best_idx])
            result_data["svg"] = svg_result["svg_content"]
            result_data["svg_base64"] = self._text_to_base64(svg_result["svg_content"])
        
        return result_data
    
    def _compute_clip_score(self, image: Image.Image, text: str) -> float:
        """Compute CLIP similarity score"""
        inputs = self.clip_processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            score = probs[0][0].item()
        
        return score
    
    def _vectorize_image(self, image: Image.Image) -> Dict:
        """Convert image to SVG"""
        from app.services.vectorizer import VectorizerService
        
        vectorizer = VectorizerService()
        result = vectorizer.vectorize(image)
        
        return result
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    def _text_to_base64(self, text: str) -> str:
        """Convert text to base64"""
        return base64.b64encode(text.encode("utf-8")).decode("utf-8")
    
    def load_custom_lora(self, lora_path: str):
        """Load custom LoRA weights"""
        self.diffusion_pipeline.load_lora_weights(lora_path)
        logger.info(f"Loaded custom LoRA from {lora_path}")
    
    def unload_lora(self):
        """Unload custom LoRA weights"""
        self.diffusion_pipeline.unload_lora_weights()
        logger.info("Unloaded LoRA weights")


def main():
    parser = argparse.ArgumentParser(description="Inference Pipeline for Logo Generation")
    
    parser.add_argument("--prompt", type=str, required=True,
                       help="Text prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default="low quality, blurry",
                       help="Negative prompt")
    parser.add_argument("--model", type=str, default="stabilityai/sdxl-turbo",
                       help="Model name")
    parser.add_argument("--lora", type=str, default=None,
                       help="Path to custom LoRA weights")
    parser.add_argument("--width", type=int, default=1024,
                       help="Image width")
    parser.add_argument("--height", type=int, default=1024,
                       help="Image height")
    parser.add_argument("--steps", type=int, default=4,
                       help="Inference steps")
    parser.add_argument("--guidance", type=float, default=0.0,
                       help="Guidance scale")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed")
    parser.add_argument("--num_images", type=int, default=1,
                       help="Number of images to generate")
    parser.add_argument("--min_clip_score", type=float, default=0.0,
                       help="Minimum CLIP score threshold")
    parser.add_argument("--vectorize", action="store_true",
                       help="Enable vectorization")
    parser.add_argument("--output", type=str, default="./output",
                       help="Output directory")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    pipeline = InferencePipeline(model_name=args.model)
    
    if args.lora:
        pipeline.load_custom_lora(args.lora)
    
    result = pipeline.generate(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        num_images=args.num_images,
        min_clip_score=args.min_clip_score,
        enable_vectorization=args.vectorize,
        output_dir=args.output
    )
    
    logger.info(f"Generation complete! Generated {result['num_generated']} images")
    logger.info(f"CLIP scores: {result['clip_scores']}")
    
    if "svg" in result:
        svg_path = Path(args.output) / "output.svg"
        with open(svg_path, "w") as f:
            f.write(result["svg"])
        logger.info(f"SVG saved to {svg_path}")


if __name__ == "__main__":
    main()
