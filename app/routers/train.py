"""
Training Router - POST /train + POST /train/auto
LoRA fine-tuning for Stable Diffusion (SD 1.5 and SDXL)
"""
from fastapi import APIRouter, HTTPException
import logging
import uuid
from pathlib import Path

from app.models.schemas import (
    TrainRequest,
    TrainResponse,
    AutoTrainRequest,
    AutoTrainResponse,
)
from app.services.trainer import LoRATrainer, TRAINING_JOB_REGISTRY

logger = logging.getLogger(__name__)
router = APIRouter()


def get_training_overview() -> dict:
    """Summarize training jobs for health endpoint."""
    total = len(TRAINING_JOB_REGISTRY)
    by_status = {}
    running = 0
    for job_id, job in TRAINING_JOB_REGISTRY.items():
        status = job.get("status", "unknown")
        by_status[status] = by_status.get(status, 0) + 1
        if status in {"initializing", "loading", "training", "epoch", "checkpoint", "started"}:
            running += 1
    latest_id = next(reversed(TRAINING_JOB_REGISTRY), None)
    latest = TRAINING_JOB_REGISTRY.get(latest_id)
    return {
        "total_jobs": total,
        "running_jobs": running,
        "by_status": by_status,
        "latest_job_id": latest_id,
        "latest_job_status": latest.get("status") if latest else None,
    }


def _update_job(job_id: str, status: str, message: str, checkpoint: str = None,
                 processed_images: int = 0, final_path: str = None):
    if job_id in TRAINING_JOB_REGISTRY:
        TRAINING_JOB_REGISTRY[job_id]["status"] = status
        TRAINING_JOB_REGISTRY[job_id]["message"] = message
        if checkpoint:
            TRAINING_JOB_REGISTRY[job_id]["checkpoint_path"] = checkpoint
        if processed_images:
            TRAINING_JOB_REGISTRY[job_id]["processed_images"] = processed_images
        if final_path:
            TRAINING_JOB_REGISTRY[job_id]["final_path"] = final_path


# ─── POST /train ──────────────────────────────────────────────────────────────

@router.post("/train", response_model=TrainResponse)
async def train_model(request: TrainRequest):
    """
    Start LoRA fine-tuning training job.

    Training runs in a background thread. Use GET /train/status/{job_id}
    to poll progress, GET /train/jobs to list all jobs.
    """
    job_id = str(uuid.uuid4())
    logger.info("Starting training job: %s", job_id)

    dataset_path = Path(request.dataset_path)
    if not dataset_path.exists():
        raise HTTPException(status_code=400, detail=f"Dataset not found: {request.dataset_path}")

    output_dir = Path(request.output_dir) / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    TRAINING_JOB_REGISTRY[job_id] = {
        "status": "initializing",
        "request": request.model_dump(),
        "output_dir": str(output_dir),
    }

    trainer = LoRATrainer(
        model_name=request.base_model,
        output_dir=str(output_dir),
        rank=request.rank,
        alpha=request.alpha,
        learning_rate=request.learning_rate,
        num_epochs=request.num_train_epochs,
        batch_size=request.per_device_train_batch_size,
        gradient_accumulation_steps=request.gradient_accumulation_steps,
        resolution=request.resolution,
        validation_prompt=request.validation_prompt,
        save_steps=request.save_steps,
        allow_cpu=request.allow_cpu,
    )

    trainer.start_training(
        dataset_path=str(dataset_path),
        job_id=job_id,
        callback=lambda status, msg, ckpt=None: _update_job(job_id, status, msg, ckpt),
    )

    return TrainResponse(
        success=True,
        job_id=job_id,
        status="started",
        message=f"Training started. Poll status at /api/train/status/{job_id}",
        checkpoint_path=None,
    )


# ─── POST /train/auto ─────────────────────────────────────────────────────────

@router.post("/train/auto", response_model=AutoTrainResponse)
async def auto_train(request: AutoTrainRequest):
    """
    One-shot endpoint: import raw images + captions → train LoRA.

    Internally runs paired_poster_import, then LoRATrainer.
    Returns job_id immediately; poll /api/train/status/{job_id} for progress.

    Example minimal call:
        POST /api/train/auto
        {"raw_dataset_path": "dataset"}

    For full control, use POST /api/train with a pre-processed dataset.
    """
    job_id = str(uuid.uuid4())
    logger.info("Starting auto-train job: %s from %s", job_id, request.raw_dataset_path)

    raw_path = Path(request.raw_dataset_path).resolve()
    if not raw_path.exists():
        raise HTTPException(status_code=400, detail=f"Raw dataset not found: {request.raw_dataset_path}")

    processed_dir = Path(request.output_dir) / job_id / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    TRAINING_JOB_REGISTRY[job_id] = {
        "status": "importing",
        "message": "Importing dataset...",
        "output_dir": str(Path(request.output_dir) / job_id),
        "processed_images": 0,
    }

    try:
        from training.data_pipeline.paired_poster_import import import_paired_posters
        meta = import_paired_posters(
            input_dir=raw_path,
            output_dir=processed_dir,
            target_size=request.resolution,
            style_prefix=request.style_prefix,
            merge_short_title=request.merge_short_title,
        )
        processed_count = meta.get("total_images", 0)
        logger.info("Auto-train imported %d images → %s", processed_count, processed_dir)
        _update_job(job_id, "imported", f"Imported {processed_count} images", processed_images=processed_count)
    except Exception as e:
        logger.exception("Auto-train import failed")
        _update_job(job_id, "error", f"Import failed: {e}")
        raise HTTPException(status_code=500, detail=f"Dataset import failed: {e}")

    trainer = LoRATrainer(
        model_name=request.base_model,
        output_dir=str(Path(request.output_dir) / job_id),
        rank=request.rank,
        alpha=request.alpha,
        learning_rate=request.learning_rate,
        num_epochs=request.num_train_epochs,
        batch_size=request.per_device_train_batch_size,
        gradient_accumulation_steps=request.gradient_accumulation_steps,
        resolution=request.resolution,
        validation_prompt=request.validation_prompt,
        save_steps=request.save_steps,
        allow_cpu=request.allow_cpu,
    )

    trainer.start_training(
        dataset_path=str(processed_dir),
        job_id=job_id,
        callback=lambda status, msg, ckpt=None: _update_job(job_id, status, msg, ckpt),
    )

    return AutoTrainResponse(
        success=True,
        job_id=job_id,
        status="started",
        message=(
            f"Pipeline started: importing then training LoRA on {processed_count} images. "
            f"Poll progress at /api/train/status/{job_id}"
        ),
        processed_images=processed_count,
    )


# ─── GET /train/status/{job_id} ──────────────────────────────────────────────

@router.get("/train/status/{job_id}")
async def get_training_status(job_id: str):
    """Poll status of a training job."""
    if job_id not in TRAINING_JOB_REGISTRY:
        raise HTTPException(status_code=404, detail="Job not found")
    job = dict(TRAINING_JOB_REGISTRY[job_id])
    job["trainer_info"] = None
    return job


# ─── GET /train/jobs ──────────────────────────────────────────────────────────

@router.get("/train/jobs")
async def list_training_jobs():
    """List all training jobs with their statuses."""
    return {
        "total": len(TRAINING_JOB_REGISTRY),
        "jobs": [
            {
                "job_id": jid,
                "status": j.get("status"),
                "message": j.get("message"),
                "output_dir": j.get("output_dir"),
                "processed_images": j.get("processed_images", 0),
                "final_path": j.get("final_path"),
            }
            for jid, j in TRAINING_JOB_REGISTRY.items()
        ],
    }


# ─── DELETE /train/cancel/{job_id} ────────────────────────────────────────────

@router.delete("/train/cancel/{job_id}")
async def cancel_training(job_id: str):
    """Cancel a running training job (best-effort)."""
    if job_id not in TRAINING_JOB_REGISTRY:
        raise HTTPException(status_code=404, detail="Job not found")
    TRAINING_JOB_REGISTRY[job_id]["status"] = "cancelled"
    TRAINING_JOB_REGISTRY[job_id]["message"] = "Cancelled by user"
    return {"success": True, "job_id": job_id, "message": "Job cancelled"}
