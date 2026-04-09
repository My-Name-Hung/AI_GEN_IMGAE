"""
LoRA Training Script - Kaggle-friendly trainer for logo/poster fine-tuning.
"""
import os

# Trước khi import torch: trên Kaggle Python 3.12, torch._dynamo có thể load rất lâu / treo cảm giác
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import logging
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from diffusers import AutoPipelineForText2Image, DDPMScheduler
from peft import LoraConfig, get_peft_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _is_p100_or_old_gpu() -> bool:
    """Kiểm tra GPU có phải P100 (sm_60) — không tương thích với PyTorch CUDA hiện tại."""
    if not torch.cuda.is_available():
        return False
    try:
        cap = torch.cuda.get_device_capability()
        sm = cap[0] * 10 + cap[1]  # ví dụ (6,0) → 60
        return sm < 70
    except Exception:
        return False


def _collate_captions(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    captions = [item["caption"] for item in batch]
    return {"pixel_values": pixel_values, "captions": captions}


class LogoTextImageDataset(Dataset):
    """Dataset for logo/poster text-to-image training."""

    def __init__(self, data_dir: str, size: int = 512):
        self.data_dir = Path(data_dir)
        self.size = size

        self.images_dir = self.data_dir / "images"
        self.captions_dir = self.data_dir / "captions"

        self.image_files = sorted(self.images_dir.glob("*.png"))
        self.image_files.extend(sorted(self.images_dir.glob("*.jpg")))
        self.image_files.extend(sorted(self.images_dir.glob("*.jpeg")))

        if not self.image_files:
            raise ValueError(f"No images found in {self.images_dir}")

        logger.info("Loaded %s images from dataset", len(self.image_files))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]

        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.size, self.size), Image.Resampling.LANCZOS)
        image = np.array(image, dtype=np.float32) / 127.5 - 1.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        caption_path = self.captions_dir / f"{img_path.stem}.txt"
        if caption_path.exists():
            caption = caption_path.read_text(encoding="utf-8").strip()
        else:
            caption = f"logo_style, {img_path.stem.replace('_', ' ')}"

        return {
            "pixel_values": image,
            "caption": caption,
        }


def maybe_enable_memory_optimizations(pipeline, device: str):
    if device != "cuda":
        return

    try:
        pipeline.enable_xformers_memory_efficient_attention()
        logger.info("Enabled xformers memory-efficient attention")
    except Exception:
        logger.info("xformers not available; continue without it")

    try:
        pipeline.enable_attention_slicing()
        logger.info("Enabled attention slicing")
    except Exception:
        logger.info("Attention slicing not available; continue")


def train_lora(
    model_name: str,
    dataset_path: str,
    output_dir: str,
    rank: int = 8,
    alpha: int = 16,
    learning_rate: float = 1e-4,
    num_epochs: int = 10,
    batch_size: int = 1,
    gradient_accumulation: int = 8,
    resolution: int = 512,
    save_steps: int = 100,
    validation_prompt: str = None,
    use_grad_checkpointing: bool = True,
    allow_cpu: bool = False,
):
    """Train LoRA for Stable Diffusion (Kaggle-safe defaults)."""
    if _is_p100_or_old_gpu():
        logger.warning(
            "Tesla P100 (sm_60) detected. PyTorch CUDA hiện tại không hỗ trợ sm_60. "
            "Chuyển sang CPU mode để training tiếp."
        )
        logger.warning(
            "Training trên CPU sẽ rất chậm (~1-2 giờ/epoch) nhưng sẽ hoàn thành đúng. "
            "Nếu cần nhanh, hãy dùng Kaggle GPU T4/P100 với PyTorch CUDA 12.x."
        )
        # Ép CPU mode
        for k in list(os.environ.keys()):
            if k.startswith("CUDA"):
                del os.environ[k]
        torch.cuda.is_available = lambda: False  # type: ignore
        device = "cpu"
        allow_cpu = True  # override để cho phép train trên CPU

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Training on %s", device)

    if device != "cuda" and not allow_cpu:
        raise RuntimeError(
            "CUDA is not available. This training script is intended for Kaggle GPU. "
            "Open Notebook Settings -> Accelerator -> GPU, then rerun. "
            "If you still want CPU mode, pass --allow_cpu (very slow, high RAM)."
        )

    dtype = torch.float32 if device == "cpu" else torch.float16

    logger.info("Loading model: %s", model_name)
    pipeline = AutoPipelineForText2Image.from_pretrained(
        model_name,
        torch_dtype=dtype,
        variant="fp16" if device == "cuda" else None,
        use_safetensors=True,
        low_cpu_mem_usage=True,
    )
    pipeline.to(device)
    maybe_enable_memory_optimizations(pipeline, device)

    unet = pipeline.unet
    text_encoder = pipeline.text_encoder
    text_encoder_2 = getattr(pipeline, "text_encoder_2", None)
    vae = pipeline.vae
    is_sdxl = text_encoder_2 is not None

    text_encoder.requires_grad_(False)
    if text_encoder_2 is not None:
        text_encoder_2.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    if use_grad_checkpointing and hasattr(unet, "enable_gradient_checkpointing"):
        unet.enable_gradient_checkpointing()
        logger.info("Enabled gradient checkpointing")

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.05,
        bias="none",
    )

    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate, weight_decay=0.01)

    dataset = LogoTextImageDataset(dataset_path, size=resolution)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device == "cuda"),
        collate_fn=_collate_captions,
    )

    noise_scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)

    global_step = 0
    train_losses = []

    use_amp = device == "cuda"
    amp_ctx = torch.amp.autocast(device_type="cuda") if use_amp else nullcontext
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(num_epochs):
        unet.train()
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(progress_bar):
            pixel_values = batch["pixel_values"].to(device)
            captions = batch["captions"]

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device,
                dtype=torch.long,
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.no_grad():
                if is_sdxl:
                    enc = pipeline.encode_prompt(
                        prompt=captions,
                        device=device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                    )
                    prompt_embeds = enc[0]
                    pooled_prompt_embeds = enc[2]
                    b = latents.shape[0]
                    add_time_ids = torch.tensor(
                        [[resolution, resolution, 0, 0, resolution, resolution]] * b,
                        device=device,
                        dtype=dtype,
                    )
                    added_cond_kwargs = {
                        "text_embeds": pooled_prompt_embeds,
                        "time_ids": add_time_ids,
                    }
                    encoder_hidden_states = prompt_embeds
                else:
                    tokens = pipeline.tokenizer(
                        captions,
                        padding="max_length",
                        max_length=pipeline.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    ).to(device)
                    encoder_hidden_states = text_encoder(tokens.input_ids)[0]
                    added_cond_kwargs = None

            from torch.cuda.amp import autocast
  		with autocast():
                if is_sdxl:
                    model_pred = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states,
                        added_cond_kwargs=added_cond_kwargs,
                    ).sample
                else:
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                loss = loss / gradient_accumulation

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * gradient_accumulation

            if (batch_idx + 1) % gradient_accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                current_loss = loss.item() * gradient_accumulation
                train_losses.append(current_loss)
                progress_bar.set_postfix({"loss": f"{current_loss:.4f}"})

                if device == "cuda":
                    torch.cuda.empty_cache()

                if global_step % save_steps == 0:
                    checkpoint_dir = output_path / f"checkpoint-{global_step}"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    unet.save_pretrained(checkpoint_dir / "unet_lora")
                    logger.info("Saved checkpoint to %s", checkpoint_dir)

        avg_loss = epoch_loss / max(len(dataloader), 1)
        logger.info("Epoch %s completed. Avg loss: %.4f", epoch + 1, avg_loss)

        if validation_prompt and (epoch + 1) % 2 == 0:
            try:
                unet.eval()
                with torch.no_grad():
                    generator = torch.Generator(device=device).manual_seed(42)
                    val_image = pipeline(
                        prompt=validation_prompt,
                        num_inference_steps=4,
                        guidance_scale=0.0,
                        generator=generator,
                    ).images[0]
                    val_image.save(output_path / f"val_epoch_{epoch + 1}.png")
                logger.info("Saved validation image for epoch %s", epoch + 1)
            except Exception as e:
                logger.warning("Validation image generation failed at epoch %s: %s", epoch + 1, e)
            finally:
                unet.train()

    final_dir = output_path / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(final_dir / "unet_lora")

    logger.info("Training complete. Model saved to %s", final_dir)

    return {
        "output_dir": str(final_dir),
        "total_steps": global_step,
        "final_loss": train_losses[-1] if train_losses else 0,
        "loss_history": train_losses,
    }


def main():
    parser = argparse.ArgumentParser(description="Train LoRA for Stable Diffusion")

    parser.add_argument("--model_name", type=str, default="stabilityai/sdxl-turbo", help="Base model name")
    parser.add_argument("--dataset", type=str, required=True, help="Path to processed dataset")
    parser.add_argument("--output", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation")
    parser.add_argument("--resolution", type=int, default=512, help="Training resolution")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--validation_prompt", type=str, default="poster_style, modern event poster, clean typography")
    parser.add_argument("--no_grad_checkpointing", action="store_true", help="Disable gradient checkpointing")
    parser.add_argument("--allow_cpu", action="store_true", help="Allow CPU training (very slow, high RAM)")

    args = parser.parse_args()

    result = train_lora(
        model_name=args.model_name,
        dataset_path=args.dataset,
        output_dir=args.output,
        rank=args.rank,
        alpha=args.alpha,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation=args.grad_accum,
        resolution=args.resolution,
        save_steps=args.save_steps,
        validation_prompt=args.validation_prompt,
        use_grad_checkpointing=not args.no_grad_checkpointing,
        allow_cpu=args.allow_cpu,
    )

    logger.info("Training result: %s", result)


if __name__ == "__main__":
    main()
