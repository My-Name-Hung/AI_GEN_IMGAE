"""
Import dataset từ thư mục phẳng: POSTER (n).png + POSTER (n).txt [+ POSTER (n)(1).txt]
→ cấu trúc chuẩn cho train_lora: images/, captions/, metadata.json

Không cần GPU; không gọi API; chỉ đọc file có sẵn.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)

# POSTER (12).png hoặc poster (12).jpg
_IMAGE_RE = re.compile(r"^POSTER\s*\((\d+)\)\.(png|jpg|jpeg)$", re.IGNORECASE)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _clean_image(img_path: Path, target_size: int) -> Optional[Image.Image]:
    try:
        img = Image.open(img_path)
        if img.mode not in ("RGB", "RGBA", "L"):
            return None
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        elif img.mode == "L":
            img = img.convert("RGB")
        if img.size[0] < 64 or img.size[1] < 64:
            return None
        img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        logger.error("Không đọc được ảnh %s: %s", img_path, e)
        return None


def _read_text(p: Path) -> Optional[str]:
    if not p.is_file():
        return None
    try:
        return p.read_text(encoding="utf-8").strip()
    except Exception as e:
        logger.warning("Không đọc được %s: %s", p, e)
        return None


def build_caption(
    folder: Path,
    index: int,
    style_prefix: str,
    merge_short_title: bool,
) -> Optional[str]:
    base = folder / f"POSTER ({index}).txt"
    text = _read_text(base)
    if not text:
        return None
    if merge_short_title:
        short = _read_text(folder / f"POSTER ({index})(1).txt")
        if short and short.lower() != text.lower():
            text = f"{text}. title: {short}"
    prefix = style_prefix.strip()
    if prefix and not prefix.endswith(","):
        prefix = prefix + ","
    if prefix:
        return f"{prefix} {text}".strip()
    return text


def import_paired_posters(
    input_dir: Path,
    output_dir: Path,
    target_size: int = 512,
    style_prefix: str = "poster_style,",
    merge_short_title: bool = False,
) -> Dict[str, Any]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    images_out = output_dir / "images"
    captions_out = output_dir / "captions"
    tags_out = output_dir / "tags"
    images_out.mkdir(parents=True, exist_ok=True)
    captions_out.mkdir(parents=True, exist_ok=True)
    tags_out.mkdir(parents=True, exist_ok=True)

    pairs: List[Tuple[int, Path]] = []
    for p in input_dir.iterdir():
        if not p.is_file():
            continue
        m = _IMAGE_RE.match(p.name)
        if not m:
            continue
        idx = int(m.group(1))
        pairs.append((idx, p))

    pairs.sort(key=lambda x: x[0])
    logger.info("Tìm thấy %s ảnh POSTER (*.png/jpg) trong %s", len(pairs), input_dir)

    metadata: Dict[str, Any] = {
        "version": "1.1",
        "total_images": 0,
        "image_size": target_size,
        "color_mode": "RGB",
        "images": [],
        "statistics": {
            "avg_caption_length": 0,
            "total_tags": 0,
            "unique_tags": [],
        },
        "source": "paired_poster_import",
        "imported_at": _utc_now_iso(),
    }

    skipped = 0
    for idx, img_path in pairs:
        cap = build_caption(input_dir, idx, style_prefix, merge_short_title)
        if not cap:
            logger.warning("Bỏ qua %s: không có POSTER (%s).txt hoặc rỗng", img_path.name, idx)
            skipped += 1
            continue

        img = _clean_image(img_path, target_size)
        if img is None:
            skipped += 1
            continue

        stem = f"poster_{idx:06d}"
        out_img = images_out / f"{stem}.png"
        img.save(out_img, format="PNG")

        cap_file = captions_out / f"{stem}.txt"
        cap_file.write_text(cap, encoding="utf-8")

        tags_file = tags_out / f"{stem}.json"
        tags_file.write_text(
            json.dumps({"tags": [], "style_token": "poster_style"}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        metadata["images"].append(
            {
                "id": stem,
                "filename": f"{stem}.png",
                "original_filename": img_path.name,
                "poster_index": idx,
                "style_token": "poster_style",
                "caption": cap,
                "tags": [],
                "width": target_size,
                "height": target_size,
                "format": "PNG",
            }
        )

    n = len(metadata["images"])
    metadata["total_images"] = n
    if n:
        metadata["statistics"]["avg_caption_length"] = sum(len(x["caption"]) for x in metadata["images"]) / n

    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Đã xuất %s mẫu → %s (bỏ qua %s)", n, output_dir, skipped)
    return metadata


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Chuyển POSTER (n).png + .txt sang thư mục train (images/, captions/, metadata.json)"
    )
    parser.add_argument("--input", type=str, default="dataset", help="Thư mục chứa POSTER (n).png và .txt")
    parser.add_argument(
        "--output",
        type=str,
        default="dataset_processed",
        help="Thư mục đích (nên khác --input để không trộn file gốc)",
    )
    parser.add_argument("--target_size", type=int, default=512, help="Cạnh ảnh vuông sau resize")
    parser.add_argument(
        "--style_prefix",
        type=str,
        default="poster_style,",
        help="Tiền tố thêm vào caption (giúp trigger LoRA khi prompt)",
    )
    parser.add_argument(
        "--merge_short_title",
        action="store_true",
        help="Nối thêm nội dung POSTER (n)(1).txt nếu có (thường là tiêu đề ngắn)",
    )
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.is_dir():
        logger.error("Không tìm thấy thư mục: %s", inp)
        sys.exit(1)

    import_paired_posters(
        input_dir=inp,
        output_dir=Path(args.output),
        target_size=args.target_size,
        style_prefix=args.style_prefix,
        merge_short_title=args.merge_short_title,
    )


if __name__ == "__main__":
    main()
