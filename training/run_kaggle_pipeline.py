"""
End-to-end Kaggle pipeline:
1) Process raw dataset (clean + auto caption + tags)
2) Train LoRA model
"""
import argparse
import logging
import os
import sys
from pathlib import Path

# Trước khi import bất kỳ module nào dùng torch (processor, train_lora)
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Make script runnable both as module and as direct file path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.data_pipeline.processor import DataPipeline
from training.lora_trainer.train_lora import train_lora

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_csv_list(value: str):
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def main():
    parser = argparse.ArgumentParser(description="Kaggle end-to-end LoRA training pipeline")

    # Raw + processed dataset
    parser.add_argument("--raw_dataset", type=str, required=True, help="Path to raw image root (e.g. ./raw)")
    parser.add_argument("--processed_dataset", type=str, default="./dataset", help="Path to processed dataset")
    parser.add_argument("--target_size", type=int, default=512, help="Processed image size")

    # Class hint folders
    parser.add_argument("--logo_folders", type=str, default="logo,logos", help="Comma-separated folder names for logo class")
    parser.add_argument("--poster_folders", type=str, default="poster,posters", help="Comma-separated folder names for poster class")

    # Train config
    parser.add_argument("--model_name", type=str, default="stabilityai/sdxl-turbo")
    parser.add_argument("--output", type=str, default="./outputs")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--validation_prompt", type=str, default="poster_style, modern event poster, clean typography")
    parser.add_argument("--no_grad_checkpointing", action="store_true")
    parser.add_argument("--allow_cpu", action="store_true")

    args = parser.parse_args()

    raw_path = Path(args.raw_dataset)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset path not found: {raw_path}")

    processed_path = Path(args.processed_dataset)
    processed_path.mkdir(parents=True, exist_ok=True)

    logo_folders = parse_csv_list(args.logo_folders)
    poster_folders = parse_csv_list(args.poster_folders)

    logger.info("Step 1/2: Processing dataset from %s", raw_path)
    pipeline = DataPipeline(
        dataset_path=str(raw_path),
        output_path=str(processed_path),
        target_size=args.target_size,
        logo_folders=logo_folders,
        poster_folders=poster_folders,
    )

    metadata = pipeline.process(recursive=True)
    validate = pipeline.validate_dataset()

    logger.info("Processed images: %s", metadata.get("total_images", 0))
    if not validate.get("valid", False):
        logger.warning("Validation issues found: %s", validate.get("issues", []))

    logger.info("Step 2/2: Training LoRA")
    result = train_lora(
        model_name=args.model_name,
        dataset_path=str(processed_path),
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

    logger.info("Done. Training result: %s", result)


if __name__ == "__main__":
    main()
