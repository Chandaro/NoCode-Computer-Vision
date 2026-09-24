"""
Advanced detection inference.

Adds the higher-end inference techniques that the plain predict path does not
expose, so a trained model can be pushed further without any code:

  * Sliced inference (SAHI style) - split a large/high-res image into overlapping
    tiles, detect inside each tile, then merge. Big win for small or far-away
    objects that a single downscaled pass would miss.
  * Test-Time Augmentation (TTA) - run prediction over flipped/scaled copies and
    merge, for a small but real accuracy bump.
  * Advanced NMS controls - class-agnostic suppression, max detections, custom
    image size (multi-scale).
  * Merging strategies - class-aware NMS or Weighted Boxes Fusion (WBF).

Everything returns boxes in the same normalised (0-1) format the rest of the app
uses, so the existing overlay UI can draw them unchanged.
"""
from __future__ import annotations

import os
import tempfile
import shutil
import time
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, File
from sqlmodel import Session

from database import get_session
from models import Project, TrainingRun

router = APIRouter(tags=["advanced-inference"])


# ── Box helpers ───────────────────────────────────────────────────────────────
# Working space is absolute pixels of the ORIGINAL image: [x1, y1, x2, y2, conf, cls]

def _iou(a: list, b: list) -> float:
    """IoU of two [x1,y1,x2,y2,...] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms(boxes: list, iou_thr: float, agnostic: bool) -> list:
    """Greedy non-maximum suppression. boxes carry [..., conf, cls]."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    kept: list = []
    while boxes:
        best = boxes.pop(0)
        kept.append(best)
        rest = []
        for b in boxes:
            same_class = agnostic or (b[5] == best[5])
            if same_class and _iou(best, b) > iou_thr:
                continue          # suppressed
            rest.append(b)
        boxes = rest
    return kept


def _wbf(boxes: list, iou_thr: float, agnostic: bool) -> list:
    """Weighted Boxes Fusion: cluster overlapping boxes and average their
    coordinates weighted by confidence. Better than NMS when merging boxes that
    come from different passes (tiles / TTA) rather than one dense prediction."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    clusters: list[list] = []      # each cluster: list of member boxes
    for b in boxes:
        placed = False
        for cl in clusters:
            rep = cl[0]
            same_class = agnostic or (rep[5] == b[5])
            if same_class and _iou(rep, b) > iou_thr:
                cl.append(b)
                placed = True
                break
        if not placed:
            clusters.append([b])

    fused: list = []
    for cl in clusters:
        total = sum(b[4] for b in cl)
        if total <= 0:
            continue
        x1 = sum(b[0] * b[4] for b in cl) / total
        y1 = sum(b[1] * b[4] for b in cl) / total
        x2 = sum(b[2] * b[4] for b in cl) / total
        y2 = sum(b[3] * b[4] for b in cl) / total
        conf = min(1.0, total / len(cl) * (1 + 0.05 * (len(cl) - 1)))  # reward agreement
        cls = max(cl, key=lambda b: b[4])[5]
        fused.append([x1, y1, x2, y2, conf, cls])
    return fused


def _merge(boxes: list, method: str, iou: float, agnostic: bool) -> list:
    return _wbf(boxes, iou, agnostic) if method == "wbf" else _nms(boxes, iou, agnostic)


# ── Slicing ───────────────────────────────────────────────────────────────────

def _tiles(w: int, h: int, size: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Grid of (x0, y0, x1, y1) tiles covering the image with overlap.
    If the image is smaller than the slice, one tile = whole image."""
    size = max(64, int(size))
    if w <= size and h <= size:
        return [(0, 0, w, h)]
    stride = max(1, int(size * (1.0 - overlap)))

    def starts(total: int) -> list[int]:
        if total <= size:
            return [0]
        pts = list(range(0, total - size + 1, stride))
        if pts[-1] != total - size:
            pts.append(total - size)      # make sure the right/bottom edge is covered
        return pts

    out: list[tuple[int, int, int, int]] = []
    for y0 in starts(h):
        for x0 in starts(w):
            out.append((x0, y0, min(w, x0 + size), min(h, y0 + size)))
    return out


def _predict_boxes(model, source, conf, iou, imgsz, agnostic, max_det, tta, device):
    """Run one prediction and return boxes in pixel coords of `source`."""
    kwargs = dict(conf=conf, iou=iou, imgsz=imgsz, verbose=False, device=device,
                  agnostic_nms=agnostic, max_det=max_det, augment=tta)
    try:
        results = model.predict(source, **kwargs)
    except RuntimeError:
        kwargs["device"] = "cpu"
        results = model.predict(source, **kwargs)
    r = results[0]
    out = []
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        out.append([x1, y1, x2, y2, float(box.conf[0]), int(box.cls[0])])
    return out


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/projects/{project_id}/training/runs/{run_id}/detect-advanced")
async def detect_advanced(
    project_id: int,
    run_id: int,
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
    imgsz: int = Form(640),
    tta: bool = Form(False),
    agnostic_nms: bool = Form(False),
    max_det: int = Form(300),
    slice_infer: bool = Form(False),
    slice_size: int = Form(640),
    slice_overlap: float = Form(0.2),
    merge: str = Form("nms"),          # "nms" | "wbf"
    session: Session = Depends(get_session),
):
    """Detection with sliced inference / TTA / advanced NMS, merged into one set."""
    from routers.infer import _get_model, _resolve_weights

    run = session.get(TrainingRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(404, "Run not found")
    if run.status != "done":
        raise HTTPException(400, "Run has not completed")

    model_path, _ = _resolve_weights(run)
    project = session.get(Project, project_id)
    class_names = project.classes if project else []

    suffix = Path(file.filename or "img.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    t0 = time.time()
    try:
        import numpy as np
        import torch
        from PIL import Image as PILImage

        device = "0" if torch.cuda.is_available() else "cpu"
        model = _get_model(model_path)

        img = PILImage.open(tmp_path).convert("RGB")
        w, h = img.size
        arr = np.array(img)

        collected: list = []          # absolute-pixel boxes across all passes
        n_tiles = 0
        merge = merge if merge in ("nms", "wbf") else "nms"

        # Pass 1: full image (always runs - catches big objects the tiles split)
        collected += _predict_boxes(model, arr, conf, iou, imgsz,
                                    agnostic_nms, max_det, tta, device)

        # Pass 2: sliced inference
        if slice_infer:
            tiles = _tiles(w, h, slice_size, slice_overlap)
            n_tiles = len(tiles)
            for (x0, y0, x1, y1) in tiles:
                if (x1 - x0) < 16 or (y1 - y0) < 16:
                    continue
                crop = arr[y0:y1, x0:x1]
                # tiles use the same imgsz so the model sees consistent scale
                for b in _predict_boxes(model, crop, conf, iou, imgsz,
                                        agnostic_nms, max_det, False, device):
                    collected.append([b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0, b[4], b[5]])

        merged = _merge(collected, merge, iou, agnostic_nms)

        detections = []
        for x1, y1, x2, y2, c, cls in merged:
            x1 = max(0.0, min(x1, w)); y1 = max(0.0, min(y1, h))
            x2 = max(0.0, min(x2, w)); y2 = max(0.0, min(y2, h))
            detections.append({
                "x": round(x1 / w, 4), "y": round(y1 / h, 4),
                "w": round((x2 - x1) / w, 4), "h": round((y2 - y1) / h, 4),
                "conf": round(float(c), 4),
                "class_id": int(cls),
                "class_name": class_names[cls] if cls < len(class_names) else f"cls{cls}",
            })

        return {
            "detections": detections,
            "image_w": w, "image_h": h,
            "count": len(detections),
            "meta": {
                "sliced": bool(slice_infer),
                "tiles": n_tiles,
                "slice_size": slice_size,
                "slice_overlap": slice_overlap,
                "tta": bool(tta),
                "agnostic_nms": bool(agnostic_nms),
                "imgsz": imgsz,
                "merge": merge,
                "raw_boxes": len(collected),
                "time_ms": int((time.time() - t0) * 1000),
            },
        }
    finally:
        os.unlink(tmp_path)
