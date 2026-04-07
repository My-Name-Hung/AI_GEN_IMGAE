"""
LoRA Trainer Service - Fine-tuning Stable Diffusion with LoRA
Supports SD 1.5 (single text encoder) and SDXL (dual text encoders)
"""
from __future__ import annotations

import logging
import os
import threading
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, List, Optional

# Trước torch: giảm treo import trên một số môi trường (Python 3.12 / Kaggle)
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from peft import LoraConfig, get_peft_model

logger = logging.getLogger(__name__)

TRAINING_JOB_REGISTRY: dict[str, dict] = {}


def _collate_captions(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    captions = [item["caption"] for item in batch]
    return {"pixel_values": pixel_values, "captions": captions}


class LogoDataset(Dataset):
    """Custom dataset for logo/poster training from images/ + captions/ folders."""

    def __init__(self, dataset_path: str, size: int = 512):
        self.data_dir = Path(dataset_path)
        self.size = size

        self.images_dir = self.data_dir / "images"
        self.captions_dir = self.data_dir / "captions"

        self.image_files: List[Path] = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            self.image_files.extend(sorted(self.images_dir.glob(ext)))
        self.image_files = sorted(set(self.image_files))

        if not self.image_files:
            raise ValueError(f"No images found in {self.images_dir}")

        logger.info("LogoDataset: loaded %d images from %s", len(self.image_files), self.data_dir)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.image_files[idx]

        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.size, self.size), Image.Resampling.LANCZOS)
        image_np = np.array(image, dtype=np.float32) / 127.5 - 1.0
        pixel_values = torch.from_numpy(image_np).permute(2, 0, 1)

        caption_file = self.captions_dir / f"{img_path.stem}.txt"
        if caption_file.is_file():
            try:
                caption = caption_file.read_text(encoding="utf-8").strip()
            except Exception:
                caption = f"logo_style, {img_path.stem.replace('_', ' ')}"
        else:
            caption = f"logo_style, {img_path.stem.replace('_', ' ')}"

        return {"pixel_values": pixel_values, "caption": caption}


class LoRATrainer:
    """
    LoRA fine-tuner for SD 1.5 and SDXL models.

    Key features:
    - Dual-text-encoder support for SDXL (text_encoder + text_encoder_2)
    - Proper DDPMScheduler noise scheduling (not ad-hoc noise)
    - Mixed precision (AMP) on CUDA
    - Gradient accumulation
    - Per-epoch validation image generation
    - Thread-based training loop so FastAPI stays responsive
    """

    def __init__(
        self,
        model_name: str = "stabilityai/sdxl-turbo",
        output_dir: str = "./outputs",
        rank: int = 16,
        alpha: int = 16,
        learning_rate: float = 1e-4,
        num_epochs: int = 10,
        batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        resolution: int = 512,
        validation_prompt: Optional[str] = None,
        save_steps: int = 100,
        allow_cpu: bool = False,
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.rank = rank
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.resolution = resolution
        self.validation_prompt = validation_prompt
        self.save_steps = save_steps
        self.allow_cpu = allow_cpu

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.pipeline = None
        self.unet = None
        self.vae = None
        self.text_encoder = None
        self.text_encoder_2 = None
        self.noise_scheduler = None
        self.optimizer = None
        self.train_losses: List[float] = []
        self.is_sdxl = False

        self._thread: Optional[threading.Thread] = None

    def setup_model(self):
        """Load base model, freeze params, attach LoRA."""
        logger.info("Setting up %s on %s", self.model_name, self.device)

        if self.device != "cuda" and not self.allow_cpu:
            raise RuntimeError(
                "CUDA not available. Pass allow_cpu=True to LoRATrainer or "
                "add --allow_cpu to the API request."
            )

        load_kwargs = {
            "torch_dtype": self.dtype,
            "use_safetensors": True,
            "low_cpu_mem_usage": True,
        }
        hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
        if hf_token:
            load_kwargs["token"] = hf_token
        if self.device == "cuda":
            load_kwargs["variant"] = "fp16"

        from diffusers import AutoPipelineForText2Image, DDPMScheduler

        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            self.model_name,
            **load_kwargs,
        )
        self.pipeline.to(self.device)

        self.unet = self.pipeline.unet
        self.text_encoder = self.pipeline.text_encoder
        self.text_encoder_2 = getattr(self.pipeline, "text_encoder_2", None)
        self.vae = self.pipeline.vae
        self.is_sdxl = self.text_encoder_2 is not None

        if self.is_sdxl:
            logger.info("Detected SDXL model (dual text encoders)")
        else:
            logger.info("Detected SD 1.5 model (single text encoder)")

        self.text_encoder.requires_grad_(False)
        if self.text_encoder_2 is not None:
            self.text_encoder_2.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)

        lora_config = LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.05,
            bias="none",
        )
        self.unet = get_peft_model(self.unet, lora_config)
        self.unet.print_trainable_parameters()

        self.noise_scheduler = DDPMScheduler.from_config(self.pipeline.scheduler.config)
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
        )

        if self.device == "cuda":
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
            except Exception:
                logger.info("xformers not available")

    def _encode_prompt_sdxl(self, captions: List[str]) -> tuple:
        """Encode captions using SDXL's dual-encoder pipeline method."""
        b = len(captions)
        result = self.pipeline.encode_prompt(
            prompt=captions,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, _ = result

        add_text_embeds = pooled_prompt_embeds
        add_time_ids = torch.tensor(
            [[self.resolution, self.resolution, 0, 0,
              self.resolution, self.resolution]] * b,
            device=self.device,
            dtype=self.dtype,
        )
        added_cond_kwargs = {
            "text_embeds": add_text_embeds,
            "time_ids": add_time_ids,
        }
        return prompt_embeds, added_cond_kwargs

    def _encode_prompt_sdv1(self, captions: List[str]) -> tuple:
        """Encode captions using SD 1.5 single text encoder."""
        tokens = self.pipeline.tokenizer(
            captions,
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        encoder_hidden_states = self.text_encoder(tokens.input_ids)[0]
        return encoder_hidden_states, None

    def _training_step(
        self,
        pixel_values: torch.Tensor,
        captions: List[str],
    ) -> float:
        """One gradient step (with accumulation handled in _train)."""
        b = pixel_values.shape[0]

        with torch.no_grad():
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (b,),
            device=self.device,
            dtype=torch.long,
        )
        noisy_latents = self.noise_scheduler.add_noise(
            latents, noise, timesteps
        )

        with torch.no_grad():
            if self.is_sdxl:
                encoder_hidden_states, added_cond_kwargs = self._encode_prompt_sdxl(captions)
            else:
                encoder_hidden_states, added_cond_kwargs = self._encode_prompt_sdv1(captions)

        use_amp = self.device == "cuda"
        amp_ctx = torch.cuda.amp.autocast if use_amp else nullcontext
        with amp_ctx():
            if self.is_sdxl:
                model_pred = self.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample
            else:
                model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

            loss = torch.nn.functional.mse_loss(
                model_pred.float(), noise.float(), reduction="mean"
            )

        return loss

    def _generate_validation(self) -> Optional[Image.Image]:
        """Generate one validation image and save to output_dir."""
        if not self.validation_prompt:
            return None
        try:
            self.unet.eval()
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device == "cuda"):
                gen = torch.Generator(device=self.device).manual_seed(42)
                img = self.pipeline(
                    prompt=self.validation_prompt,
                    num_inference_steps=4,
                    guidance_scale=0.0,
                    generator=gen,
                ).images[0]
            self.unet.train()
            return img
        except Exception as e:
            logger.warning("Validation image failed: %s", e)
            self.unet.train()
            return None

    def start_training(
        self,
        dataset_path: str,
        job_id: str,
        callback: Optional[Callable] = None,
    ):
        """Launch training in a background thread (non-blocking for FastAPI)."""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError(f"Job {job_id} is already running")

        def train_thread():
            try:
                self._train(dataset_path, job_id, callback)
            except Exception as e:
                logger.exception("Training thread crashed")
                if callback:
                    callback("error", str(e))

        self._thread = threading.Thread(target=train_thread, daemon=True)
        self._thread.start()

    def _train(
        self,
        dataset_path: str,
        job_id: str,
        callback: Optional[Callable] = None,
    ):
        self.train_losses = []

        try:
            self.setup_model()
        except Exception as e:
            if callback:
                callback("error", f"Model setup failed: {e}")
            return

        try:
            dataset = LogoDataset(dataset_path, size=self.resolution)
        except Exception as e:
            if callback:
                callback("error", f"Dataset load failed: {e}")
            return

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(self.device == "cuda"),
            collate_fn=_collate_captions,
        )

        if callback:
            callback("training", f"Starting {self.num_epochs} epochs...")

        use_amp = self.device == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        global_step = 0
        grad_accum_counter = 0

        for epoch in range(self.num_epochs):
            self.unet.train()
            epoch_loss = 0.0

            for batch in dataloader:
                pixel_values = batch["pixel_values"].to(self.device)
                captions = batch["captions"]

                loss = self._training_step(pixel_values, captions)
                scaled_loss = loss / self.gradient_accumulation_steps
                scaler.scale(scaled_loss).backward()
                epoch_loss += loss.item()

                grad_accum_counter += 1
                if grad_accum_counter % self.gradient_accumulation_steps == 0:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                    global_step += 1
                    current_loss = loss.item()
                    self.train_losses.append(current_loss)

                    if global_step % self.save_steps == 0:
                        ckpt_dir = self.output_dir / f"checkpoint-{global_step}"
                        ckpt_dir.mkdir(parents=True, exist_ok=True)
                        self.unet.save_pretrained(ckpt_dir / "unet_lora")
                        logger.info("Saved checkpoint step %d", global_step)
                        if callback:
                            callback("checkpoint", f"Step {global_step}", str(ckpt_dir))

                    if self.device == "cuda":
                        torch.cuda.empty_cache()

            avg_loss = epoch_loss / max(len(dataloader), 1)
            logger.info("Epoch %d/%d done. Avg loss: %.4f", epoch + 1, self.num_epochs, avg_loss)
            if callback:
                callback("epoch", f"Epoch {epoch + 1}/{self.num_epochs} — loss {avg_loss:.4f}")

            if (epoch + 1) % 2 == 0:
                val_img = self._generate_validation()
                if val_img:
                    val_path = self.output_dir / f"val_epoch_{epoch + 1}.png"
                    val_img.save(val_path)
                    logger.info("Saved validation image: %s", val_path)
                    if callback:
                        callback("validation", f"Validation saved: {val_path}")

        final_dir = self.output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        self.unet.save_pretrained(final_dir / "unet_lora")
        logger.info("Training complete. Final model: %s", final_dir)

        if callback:
            callback("completed", f"Done. {global_step} steps, final loss {avg_loss:.4f}", str(final_dir))

    def get_status(self) -> dict:
        return {
            "device": self.device,
            "model": self.model_name,
            "is_sdxl": self.is_sdxl,
            "total_steps": len(self.train_losses),
            "latest_loss": self.train_losses[-1] if self.train_losses else None,
            "is_running": self._thread is not None and self._thread.is_alive(),
        }
