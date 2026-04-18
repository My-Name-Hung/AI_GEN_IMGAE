"""
Extract trained LoRA zip files into a standard directory structure.
Run once after training: python -m app.services.lora_extractor
"""
import logging
import shutil
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Standard path: outputs/{lora_type}/final/unet_lora/
ROOT = Path(__file__).resolve().parents[2] / "outputs"
EXTRACT_ROOT = ROOT

LORA_ZIPS = {
    "lora_logo_2d": ROOT.parent / "lora_logo_2d.zip",
    "lora_logo_3d": ROOT.parent / "lora_logo_3d.zip",
    "lora_poster": ROOT.parent / "lora_poster.zip",
}

REQUIRED_FILES = [
    "adapter_config.json",
    "adapter_model.safetensors",
]


def extract_final_lora(zip_path: Path, target_dir: Path) -> bool:
    """Extract the final/ checkpoint from a zip into target_dir."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()

        # Find the final/unet_lora/ path inside the zip
        final_prefix = None
        for name in members:
            if name.endswith("final/unet_lora/") or "final/unet_lora/" in name:
                prefix = name.split("final/unet_lora")[0]
                if final_prefix is None or len(prefix) < len(final_prefix):
                    final_prefix = prefix
                break

        if final_prefix is None:
            logger.error("No final/unet_lora found in %s", zip_path)
            return False

        final_src = f"{final_prefix}final/unet_lora/"
        src_len = len(final_src)

        written = []
        for name in members:
            if not name.startswith(final_src):
                continue
            rel = name[src_len:]
            if not rel:
                continue
            out = target_dir / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, open(out, "wb") as dst:
                dst.write(src.read())
            written.append(rel)

        for req in REQUIRED_FILES:
            if not (target_dir / req).exists():
                logger.error("Missing required file: %s", req)
                return False

    logger.info("Extracted %d files to %s", len(written), target_dir)
    return True


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    results = {}
    for lora_type, zip_path in LORA_ZIPS.items():
        if not zip_path.exists():
            logger.warning("Zip not found: %s — skipping", zip_path)
            results[lora_type] = "missing"
            continue

        target = EXTRACT_ROOT / lora_type / "final" / "unet_lora"
        if target.exists() and (target / "adapter_model.safetensors").exists():
            logger.info("Already extracted: %s", target)
            results[lora_type] = "already_exists"
            continue

        logger.info("Extracting %s from %s", lora_type, zip_path.name)
        ok = extract_final_lora(zip_path, target)
        results[lora_type] = "ok" if ok else "failed"

    print("\n=== Extraction Summary ===")
    for name, status in results.items():
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
