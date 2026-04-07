"""
Data Pipeline - Auto-clean, auto-caption, auto-tag cho dataset logo/poster
Uses BLIP/Florence-2 for captioning, CLIP for tagging
"""
import logging
import json
import uuid
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image
import torch
import re

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Complete data processing pipeline for logo/poster dataset.

    Pipeline stages:
    1. Auto-clean (resize, RGB, remove corrupted)
    2. Auto-caption (BLIP/BLIP2/Florence-2)
    3. Auto-tag (CLIP)
    4. Metadata generation
    """

    def __init__(
        self,
        dataset_path: str,
        output_path: str,
        target_size: int = 1024,
        caption_model: str = "Salesforce/blip-image-captioning-base",
        tag_model: str = "openai/clip-vit-large-patch14",
        logo_folders: Optional[List[str]] = None,
        poster_folders: Optional[List[str]] = None,
    ):
        """
        Initialize Data Pipeline.

        Args:
            dataset_path: Path to raw images
            output_path: Path to save processed dataset
            target_size: Target image size (square)
            caption_model: BLIP model name
            tag_model: CLIP model name
            logo_folders: Folder names treated as logo class (relative to dataset_path)
            poster_folders: Folder names treated as poster class (relative to dataset_path)
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.target_size = target_size

        self.images_out = self.output_path / "images"
        self.captions_out = self.output_path / "captions"
        self.tags_out = self.output_path / "tags"

        self.images_out.mkdir(parents=True, exist_ok=True)
        self.captions_out.mkdir(parents=True, exist_ok=True)
        self.tags_out.mkdir(parents=True, exist_ok=True)

        self.caption_model_name = caption_model
        self.tag_model_name = tag_model

        self.caption_model = None
        self.caption_processor = None
        self.tag_model = None
        self.tag_processor = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.logo_folders = set((logo_folders or ["logo", "logos"]))
        self.poster_folders = set((poster_folders or ["poster", "posters"]))

        self.stats = {
            "total_processed": 0,
            "failed": 0,
            "skipped": 0
        }

    def load_models(self):
        """Load captioning and tagging models"""
        from transformers import BlipProcessor, BlipForConditionalGeneration
        from transformers import CLIPProcessor, CLIPModel

        logger.info("Loading captioning model...")
        self.caption_processor = BlipProcessor.from_pretrained(self.caption_model_name)
        self.caption_model = BlipForConditionalGeneration.from_pretrained(self.caption_model_name)
        self.caption_model = self.caption_model.to(self.device)
        self.caption_model.eval()

        logger.info("Loading tagging model...")
        self.tag_processor = CLIPProcessor.from_pretrained(self.tag_model_name)
        self.tag_model = CLIPModel.from_pretrained(self.tag_model_name)
        self.tag_model = self.tag_model.to(self.device)
        self.tag_model.eval()

        logger.info("Models loaded successfully")

    def process(self, image_extensions: List[str] = None, recursive: bool = True) -> Dict:
        """
        Process entire dataset.

        Args:
            image_extensions: List of valid extensions (e.g., ['.jpg', '.png'])
            recursive: Search images recursively in dataset_path

        Returns:
            Processing statistics and metadata
        """
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']

        self.load_models()

        image_files = []
        for ext in image_extensions:
            if recursive:
                image_files.extend(list(self.dataset_path.rglob(f"*{ext}")))
                image_files.extend(list(self.dataset_path.rglob(f"*{ext.upper()}")))
            else:
                image_files.extend(list(self.dataset_path.glob(f"*{ext}")))
                image_files.extend(list(self.dataset_path.glob(f"*{ext.upper()}")))

        image_files = sorted(list(set(image_files)))
        logger.info(f"Found {len(image_files)} images to process")

        metadata = {
            "version": "1.1",
            "total_images": 0,
            "image_size": self.target_size,
            "color_mode": "RGB",
            "images": [],
            "statistics": {
                "avg_caption_length": 0,
                "total_tags": 0,
                "unique_tags": []
            }
        }

        unique_tags = set()

        for img_path in image_files:
            try:
                result = self._process_single_image(img_path)

                if result:
                    metadata["images"].append(result)
                    unique_tags.update(result["tags"])
                    self.stats["total_processed"] += 1
                else:
                    self.stats["skipped"] += 1

            except Exception as e:
                logger.error(f"Failed to process {img_path}: {e}")
                self.stats["failed"] += 1

        metadata["total_images"] = len(metadata["images"])
        metadata["statistics"]["unique_tags"] = sorted(list(unique_tags))

        total_captions = sum(len(img["caption"]) for img in metadata["images"])
        metadata["statistics"]["avg_caption_length"] = (
            total_captions / len(metadata["images"]) if metadata["images"] else 0
        )
        metadata["statistics"]["total_tags"] = sum(
            len(img["tags"]) for img in metadata["images"]
        )

        metadata_path = self.output_path / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(
            "Processing complete. Total: %s, Failed: %s, Skipped: %s",
            self.stats['total_processed'],
            self.stats['failed'],
            self.stats['skipped']
        )

        return metadata

    def _process_single_image(self, img_path: Path) -> Optional[Dict]:
        """Process a single image through all pipeline stages"""

        img_id = str(uuid.uuid4())[:8]

        img = self._clean_image(img_path)
        if img is None:
            return None

        output_filename = f"{img_id}.png"
        output_path = self.images_out / output_filename
        img.save(output_path, format="PNG")

        style_token = self._infer_style_token(img_path)
        caption = self._generate_caption(img, style_token=style_token)
        tags = self._generate_tags(img)

        caption_file = self.captions_out / f"{img_id}.txt"
        with open(caption_file, "w", encoding="utf-8") as f:
            f.write(caption)

        tags_file = self.tags_out / f"{img_id}.json"
        with open(tags_file, "w", encoding="utf-8") as f:
            json.dump({"tags": tags, "style_token": style_token}, f, ensure_ascii=False, indent=2)

        return {
            "id": img_id,
            "filename": output_filename,
            "original_filename": img_path.name,
            "style_token": style_token,
            "caption": caption,
            "tags": tags,
            "width": self.target_size,
            "height": self.target_size,
            "format": "PNG"
        }

    def _clean_image(self, img_path: Path) -> Optional[Image.Image]:
        """Clean and normalize image"""
        try:
            img = Image.open(img_path)

            if img.mode not in ("RGB", "RGBA", "L"):
                return None

            if img.mode == "RGBA":
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            elif img.mode == "L":
                img = img.convert("RGB")

            if img.size[0] < 64 or img.size[1] < 64:
                return None

            img = img.resize(
                (self.target_size, self.target_size),
                Image.Resampling.LANCZOS
            )

            return img

        except Exception as e:
            logger.error(f"Image cleaning failed for {img_path}: {e}")
            return None

    def _infer_style_token(self, img_path: Path) -> str:
        """Infer style token from parent folders and filename."""
        lower_parts = [part.lower() for part in img_path.parts]
        stem = img_path.stem.lower()

        for part in lower_parts:
            if part in self.logo_folders:
                return "logo_style"
            if part in self.poster_folders:
                return "poster_style"

        if re.search(r"\blogo\b", stem):
            return "logo_style"
        if re.search(r"\bposter\b", stem):
            return "poster_style"

        return "logo_style"

    def _generate_caption(self, img: Image.Image, style_token: str = "logo_style") -> str:
        """Generate caption using BLIP and prepend style token."""
        try:
            inputs = self.caption_processor(
                img,
                return_tensors="pt"
            ).to(self.device)

            out = self.caption_model.generate(
                **inputs,
                max_length=120,
                num_beams=4,
                repetition_penalty=1.2
            )

            caption = self.caption_processor.decode(out[0], skip_special_tokens=True).strip()
            if not caption:
                caption = "minimal design"

            return f"{style_token}, {caption}"

        except Exception as e:
            logger.error(f"Caption generation failed: {e}")
            return f"{style_token}, clean modern design"

    def _generate_tags(self, img: Image.Image) -> List[str]:
        """Generate tags using CLIP"""
        try:
            candidate_labels = [
                "minimalist logo", "modern logo", "vintage logo",
                "corporate logo", "tech logo", "food logo", "fashion logo",
                "sports logo", "music logo", "artistic design",
                "geometric shape", "typography", "icon design",
                "badge logo", "emblem", "monogram", "text logo",
                "symbol mark", "abstract shape", "flat design",
                "gradient background", "solid background", "dark theme",
                "light theme", "colorful", "monochrome",
                "poster design", "event poster", "cinematic poster"
            ]

            inputs = self.tag_processor(
                text=candidate_labels,
                images=img,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.tag_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)

            top_indices = probs[0].topk(5)[1].tolist()
            tags = [candidate_labels[i] for i in top_indices]

            return tags

        except Exception as e:
            logger.error(f"Tag generation failed: {e}")
            return ["design"]

    def validate_dataset(self) -> Dict:
        """Validate processed dataset integrity"""
        issues = []

        if not self.images_out.exists():
            issues.append("Images directory missing")
            return {"valid": False, "issues": issues}

        image_files = list(self.images_out.glob("*.png"))
        metadata_path = self.output_path / "metadata.json"

        if not metadata_path.exists():
            issues.append("Metadata file missing")
            return {"valid": False, "issues": issues}

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        metadata_ids = {img["id"] for img in metadata["images"]}

        for img_file in image_files:
            img_id = img_file.stem
            if img_id not in metadata_ids:
                issues.append(f"Image {img_file.name} not in metadata")

        for img_meta in metadata["images"]:
            caption_file = self.captions_out / f"{img_meta['id']}.txt"
            if not caption_file.exists():
                issues.append(f"Caption missing for {img_meta['id']}")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "stats": {
                "images": len(image_files),
                "metadata_entries": len(metadata["images"])
            }
        }
