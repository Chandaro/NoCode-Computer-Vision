from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from collections import defaultdict
import os, json
import numpy as np
from PIL import Image as PILImage

from database import get_session
from models import Project, Image, Annotation

router = APIRouter(prefix="/projects/{project_id}/analytics", tags=["analytics"])

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")


@router.get("")
def get_analytics(project_id: int, session: Session = Depends(get_session)):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    classes = project.classes
    images  = session.exec(select(Image).where(Image.project_id == project_id)).all()
    anns    = []
    for img in images:
        img_anns = session.exec(select(Annotation).where(Annotation.image_id == img.id)).all()
        anns.extend(img_anns)

    # ── Class distribution ─────────────────────────────────────────────────
    class_counts = defaultdict(int)
    for a in anns:
        name = classes[a.class_id] if a.class_id < len(classes) else f"cls{a.class_id}"
        class_counts[name] += 1
    class_distribution = dict(class_counts)

    # ── Shape breakdown ────────────────────────────────────────────────────
    shape_breakdown = {"bbox": 0, "polygon": 0, "point": 0}
    for a in anns:
        if a.shape_type in shape_breakdown:
            shape_breakdown[a.shape_type] += 1

    # ── Annotations per image ──────────────────────────────────────────────
    ann_per_image = defaultdict(int)
    for a in anns:
        ann_per_image[a.image_id] += 1

    ann_counts = list(ann_per_image.values())
    unannotated = len(images) - len(ann_per_image)
    ann_histogram = {
        "0": unannotated,
        "1-5": sum(1 for c in ann_counts if 1 <= c <= 5),
        "6-10": sum(1 for c in ann_counts if 6 <= c <= 10),
        "11-20": sum(1 for c in ann_counts if 11 <= c <= 20),
        "21+": sum(1 for c in ann_counts if c > 20),
    }

    # ── Image dimensions & aspect ratios ─────────────────────────────────
    sized = [img for img in images if img.width > 0 and img.height > 0]
    size_samples = [{"w": img.width, "h": img.height} for img in sized[:500]]

    aspect_buckets = {"portrait (<0.9)": 0, "square (0.9-1.1)": 0, "landscape (>1.1)": 0}
    for img in sized:
        ratio = img.width / img.height
        if ratio < 0.9:
            aspect_buckets["portrait (<0.9)"] += 1
        elif ratio <= 1.1:
            aspect_buckets["square (0.9-1.1)"] += 1
        else:
            aspect_buckets["landscape (>1.1)"] += 1

    # ── Color space breakdown ─────────────────────────────────────────────
    color_counts = defaultdict(int)
    for img in images:
        color_counts[img.color_space or "RGB"] += 1

    # ── Corrupt / valid counts ────────────────────────────────────────────
    corrupt_count = sum(1 for img in images if img.is_corrupt)

    # ── Channel mean / std (sample up to 100 images) ─────────────────────
    channel_stats = None
    sample_imgs = [img for img in images if not img.is_corrupt and img.channels == 3][:100]
    if sample_imgs:
        means, stds = [], []
        for img in sample_imgs:
            path = os.path.join(UPLOAD_DIR, img.filename)
            if not os.path.exists(path):
                continue
            try:
                arr = np.array(PILImage.open(path).convert("RGB")).astype(np.float32) / 255.0
                means.append(arr.reshape(-1, 3).mean(axis=0).tolist())
                stds.append(arr.reshape(-1, 3).std(axis=0).tolist())
            except Exception:
                pass
        if means:
            mean_arr = np.mean(means, axis=0)
            std_arr  = np.mean(stds,  axis=0)
            channel_stats = {
                "mean": {"R": round(float(mean_arr[0]), 4), "G": round(float(mean_arr[1]), 4), "B": round(float(mean_arr[2]), 4)},
                "std":  {"R": round(float(std_arr[0]),  4), "G": round(float(std_arr[1]),  4), "B": round(float(std_arr[2]),  4)},
            }

    return {
        "total_images":       len(images),
        "annotated_images":   len(ann_per_image),
        "total_annotations":  len(anns),
        "corrupt_images":     corrupt_count,
        "class_distribution": class_distribution,
        "shape_breakdown":    shape_breakdown,
        "ann_histogram":      ann_histogram,
        "size_samples":       size_samples,
        "aspect_buckets":     aspect_buckets,
        "color_space_counts": dict(color_counts),
        "channel_stats":      channel_stats,
    }
