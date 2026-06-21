from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Form
from fastapi.responses import FileResponse
from sqlmodel import Session
from pydantic import BaseModel
from typing import Optional, List
import tempfile, os, shutil, json, threading, uuid
from pathlib import Path

from database import get_session
from models import TrainingRun, ClassificationRun, CustomTrainingRun, CustomModelConfig, Project, ExternalModel
from models import Image as ImageModel

router = APIRouter(tags=["inference"])

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")
VIDEO_OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "video_out")
os.makedirs(VIDEO_OUT_DIR, exist_ok=True)

_video_jobs: dict  = {}
_model_cache: dict = {}   # model_path → YOLO instance (loaded once, reused)


def _get_model(model_path: str):
    """Return cached YOLO model; load on first call."""
    if model_path not in _model_cache:
        from ultralytics import YOLO
        _model_cache[model_path] = YOLO(model_path)
    return _model_cache[model_path]


def _find_weights(model_path: str, run_dir: str) -> str | None:
    """Search for weights in stored path then run_dir. Returns path or None."""
    if model_path and os.path.exists(model_path):
        return model_path
    if run_dir:
        for fname in ("best.pt", "last.pt"):
            candidate = os.path.join(run_dir, "weights", fname)
            if os.path.exists(candidate):
                return candidate
    return None


def _resolve_weights(run: TrainingRun) -> tuple[str, bool]:
    """Return (model_path, is_fallback) for a detection TrainingRun.

    Priority:
      1. run.model_path              — saved best.pt from training
      2. run.run_dir/weights/best.pt — alternate location
      3. run.run_dir/weights/last.pt — last checkpoint
      4. run.model_base              — base YOLO model (ultralytics downloads)
    """
    found = _find_weights(run.model_path, run.run_dir)
    if found:
        return found, False
    base = run.model_base or "yolo11n.pt"
    return base, True


# ─── Detection inference ──────────────────────────────────────────────────────
@router.post("/projects/{project_id}/training/runs/{run_id}/infer")
async def detection_infer(
    project_id: int,
    run_id: int,
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
    session: Session = Depends(get_session),
):
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

    try:
        import torch
        from ultralytics import YOLO
        device = "0" if torch.cuda.is_available() else "cpu"
        model = _get_model(model_path)   # use cached model — avoids disk reload on every request
        try:
            results = model.predict(tmp_path, conf=conf, iou=iou, verbose=False, device=device)
        except RuntimeError:
            results = model.predict(tmp_path, conf=conf, iou=iou, verbose=False, device="cpu")
        r = results[0]
        ih, iw = r.orig_shape

        detections = []
        for box in r.boxes:
            x1n, y1n, x2n, y2n = box.xyxyn[0].tolist()
            cls_id = int(box.cls[0])
            detections.append({
                "x": round(x1n, 4), "y": round(y1n, 4),
                "w": round(x2n - x1n, 4), "h": round(y2n - y1n, 4),
                "conf": round(float(box.conf[0]), 4),
                "class_id": cls_id,
                "class_name": class_names[cls_id] if cls_id < len(class_names) else f"cls{cls_id}",
            })
        return {"detections": detections, "image_w": iw, "image_h": ih, "count": len(detections)}
    finally:
        os.unlink(tmp_path)


# ─── Image URL inference ──────────────────────────────────────────────────────
def _fetch_image_from_url(url: str) -> bytes:
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


@router.post("/projects/{project_id}/training/runs/{run_id}/infer-url")
async def detection_infer_url(
    project_id: int,
    run_id:     int,
    url:  str   = Form(...),
    conf: float = Form(0.25),
    iou:  float = Form(0.45),
    session: Session = Depends(get_session),
):
    run = session.get(TrainingRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(404, "Run not found")
    if run.status != "done":
        raise HTTPException(400, "Run has not completed")

    model_path, _ = _resolve_weights(run)

    project = session.get(Project, project_id)
    class_names = project.classes if project else []

    try:
        img_bytes = _fetch_image_from_url(url)
    except Exception as exc:
        raise HTTPException(400, f"Failed to fetch image: {exc}")

    import io, torch, numpy as np
    from PIL import Image as PILImage
    try:
        img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"URL did not return a valid image: {exc}")
    arr = np.array(img)

    from ultralytics import YOLO
    device = "0" if torch.cuda.is_available() else "cpu"
    model  = _get_model(model_path)   # use cached model
    try:
        results = model.predict(arr, conf=conf, iou=iou, verbose=False, device=device)
    except RuntimeError:
        results = model.predict(arr, conf=conf, iou=iou, verbose=False, device="cpu")

    r = results[0]
    ih, iw = r.orig_shape
    detections: list = []
    for idx, box in enumerate(r.boxes):
        x1n, y1n, x2n, y2n = box.xyxyn[0].tolist()
        cls_id = int(box.cls[0])
        mask_pts: list = []
        if r.masks is not None and idx < len(r.masks.xyn):
            mask_pts = [[round(float(p[0]), 4), round(float(p[1]), 4)]
                        for p in r.masks.xyn[idx]]
        detections.append({
            "x": round(x1n, 4), "y": round(y1n, 4),
            "w": round(x2n - x1n, 4), "h": round(y2n - y1n, 4),
            "conf": round(float(box.conf[0]), 4),
            "class_id": cls_id,
            "class_name": class_names[cls_id] if cls_id < len(class_names) else f"cls{cls_id}",
            "mask": mask_pts,
        })
    return {"detections": detections, "image_w": iw, "image_h": ih,
            "count": len(detections), "filename": url.split("/")[-1][:64] or "image"}


# ─── Detection batch test ─────────────────────────────────────────────────────
@router.post("/projects/{project_id}/training/runs/{run_id}/test-batch")
async def detection_test_batch(
    project_id: int,
    run_id: int,
    files: List[UploadFile] = File(...),
    conf: float = Form(0.25),
    iou: float  = Form(0.45),
    session: Session = Depends(get_session),
):
    """Run a trained detection model on a batch of images and return per-image
    detections plus aggregate summary statistics."""
    run = session.get(TrainingRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(404, "Run not found")
    if run.status != "done":
        raise HTTPException(400, "Run has not completed")

    model_path, _ = _resolve_weights(run)

    project     = session.get(Project, project_id)
    class_names = project.classes if project else []

    import torch
    from ultralytics import YOLO
    device = "0" if torch.cuda.is_available() else "cpu"
    model  = YOLO(model_path)

    images_out: list = []
    for upload in files:
        suffix = Path(upload.filename or "img.jpg").suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload.file, tmp)
            tmp_path = tmp.name
        try:
            try:
                results = model.predict(tmp_path, conf=conf, iou=iou, verbose=False, device=device)
            except RuntimeError:
                results = model.predict(tmp_path, conf=conf, iou=iou, verbose=False, device="cpu")
            r  = results[0]
            ih, iw = r.orig_shape
            detections: list = []
            for idx, box in enumerate(r.boxes):
                x1n, y1n, x2n, y2n = box.xyxyn[0].tolist()
                cls_id = int(box.cls[0])
                mask_pts: list = []
                if r.masks is not None and idx < len(r.masks.xyn):
                    mask_pts = [[round(float(p[0]), 4), round(float(p[1]), 4)]
                                for p in r.masks.xyn[idx]]
                detections.append({
                    "x": round(x1n, 4), "y": round(y1n, 4),
                    "w": round(x2n - x1n, 4), "h": round(y2n - y1n, 4),
                    "conf": round(float(box.conf[0]), 4),
                    "class_id": cls_id,
                    "class_name": (class_names[cls_id]
                                   if cls_id < len(class_names)
                                   else f"cls{cls_id}"),
                    "mask": mask_pts,
                })
            images_out.append({
                "filename": upload.filename or "image",
                "detections": detections,
                "count": len(detections),
                "image_w": iw,
                "image_h": ih,
            })
        finally:
            os.unlink(tmp_path)

    # ── Aggregate stats ────────────────────────────────────────────────────────
    total_det = sum(img["count"] for img in images_out)
    class_counts: dict = {}
    for img in images_out:
        for d in img["detections"]:
            name = d["class_name"]
            class_counts[name] = class_counts.get(name, 0) + 1

    return {
        "images": images_out,
        "summary": {
            "total_images":             len(images_out),
            "total_detections":         total_det,
            "avg_detections_per_image": round(total_det / max(len(images_out), 1), 2),
            "images_with_detections":   sum(1 for img in images_out if img["count"] > 0),
            "class_counts":             class_counts,
        },
    }


# ─── Classification inference ─────────────────────────────────────────────────
@router.post("/projects/{project_id}/classification/runs/{run_id}/infer")
async def classification_infer(
    project_id: int,
    run_id: int,
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    run = session.get(ClassificationRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(404, "Run not found")
    if run.status != "done":
        raise HTTPException(400, "Run has not completed")

    model_path = _find_weights(run.model_path, run.run_dir)
    if not model_path:
        raise HTTPException(404, "Model weights not found — re-train this run to restore them.")

    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    # Use the class order saved at training time (ImageFolder alphabetical order).
    # Falling back to sorted(project.classes) keeps old runs working correctly.
    saved_results = json.loads(run.results_json) if run.results_json else {}
    classes = saved_results.get("class_names") or sorted(project.classes)

    suffix = Path(file.filename or "img.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        import torch
        import torchvision.transforms as T
        import torchvision.models as M
        import torch.nn as nn
        from PIL import Image as PILImage

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_fn_map = {
            "resnet18":           M.resnet18,
            "resnet34":           M.resnet34,
            "resnet50":           M.resnet50,
            "mobilenet_v3_small": M.mobilenet_v3_small,
            "efficientnet_b0":    M.efficientnet_b0,
            "efficientnet_b1":    M.efficientnet_b1,
            "convnext_tiny":      M.convnext_tiny,
        }

        if not classes:
            raise HTTPException(400, "No classes found for this project — add dataset images first.")

        # Load state dict first so we can inspect keys and match the head architecture
        try:
            state = torch.load(model_path, map_location=device, weights_only=False)
        except Exception as e:
            raise HTTPException(500, f"Could not read weights file: {e}")

        model = model_fn_map.get(run.base_model, M.resnet18)(weights=None)
        n_classes = len(classes)

        # Detect whether the head was saved as Sequential(Dropout, Linear) or plain Linear
        # Sequential produces keys like "fc.1.weight"; plain Linear produces "fc.weight"
        has_seq_fc  = any(k.startswith("fc.1.") for k in state)
        has_seq_cls = any(".classifier." in k and ".1.weight" in k for k in state)

        if hasattr(model, "fc"):
            in_f = model.fc.in_features
            model.fc = (
                nn.Sequential(nn.Dropout(0.0), nn.Linear(in_f, n_classes))
                if has_seq_fc else nn.Linear(in_f, n_classes)
            )
        elif hasattr(model, "classifier"):
            in_feat = model.classifier[-1].in_features
            model.classifier[-1] = (
                nn.Sequential(nn.Dropout(0.0), nn.Linear(in_feat, n_classes))
                if has_seq_cls else nn.Linear(in_feat, n_classes)
            )

        try:
            model.load_state_dict(state)
        except RuntimeError as e:
            raise HTTPException(500, f"Model load error (re-train may fix this): {e}")

        model = model.to(device).eval()

        tf = T.Compose([
            T.Resize((run.imgsz, run.imgsz)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        img = PILImage.open(tmp_path).convert("RGB")
        tensor = tf(img).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0].tolist()

        preds = sorted(
            [{"class_id": i,
              "class_name": classes[i] if i < len(classes) else f"cls{i}",
              "probability": round(p, 4)}
             for i, p in enumerate(probs)],
            key=lambda x: x["probability"], reverse=True,
        )
        return {"predictions": preds, "top1": preds[0] if preds else None, "top5": preds[:5]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Inference error: {type(e).__name__}: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ─── Custom CNN inference ─────────────────────────────────────────────────────
@router.post("/projects/{project_id}/custom/runs/{run_id}/infer")
async def custom_cnn_infer(
    project_id: int,
    run_id: int,
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    run = session.get(CustomTrainingRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(404, "Run not found")
    if run.status != "done":
        raise HTTPException(400, "Run has not completed")

    model_path = _find_weights(run.model_path, run.run_dir)
    if not model_path:
        raise HTTPException(404, "Model weights not found — re-train this run to restore them.")

    project = session.get(Project, project_id)
    cfg     = session.get(CustomModelConfig, run.config_id)
    if not project or not cfg:
        raise HTTPException(404, "Project or config not found")

    saved_results = json.loads(run.results_json) if run.results_json else {}
    classes = saved_results.get("class_names") or sorted(project.classes)
    layers  = json.loads(cfg.layers_json) if hasattr(cfg, "layers_json") else []

    suffix = Path(file.filename or "img.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        import torch
        import torch.nn as nn
        import torchvision.transforms as T
        from PIL import Image as PILImage
        from routers.custom import _build_torch_model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = _build_torch_model(layers, cfg.input_h, cfg.input_w, len(classes))
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        model = model.to(device).eval()

        tf = T.Compose([
            T.Resize((cfg.input_h, cfg.input_w)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        img    = PILImage.open(tmp_path).convert("RGB")
        tensor = tf(img).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0].tolist()

        preds = sorted(
            [{"class_id": i,
              "class_name": classes[i] if i < len(classes) else f"cls{i}",
              "probability": round(p, 4)}
             for i, p in enumerate(probs)],
            key=lambda x: x["probability"], reverse=True,
        )
        return {"predictions": preds, "top1": preds[0] if preds else None, "top5": preds[:5]}
    finally:
        os.unlink(tmp_path)


# ─── Auto-annotate ────────────────────────────────────────────────────────────
_hf_cache: dict = {}   # model_dir → (processor, model, task)


def _load_hf(model_dir: str):
    """Load and cache a HuggingFace image model (detection or classification)."""
    if model_dir in _hf_cache:
        return _hf_cache[model_dir]
    import json as _json, torch
    from transformers import (AutoImageProcessor, AutoModelForObjectDetection,
                              AutoModelForImageClassification)
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        archs = " ".join(_json.load(f).get("architectures", []) or []).lower()
    proc = AutoImageProcessor.from_pretrained(model_dir)
    if "objectdetection" in archs:
        task = "detection"
        model = AutoModelForObjectDetection.from_pretrained(model_dir)
    else:
        task = "classification"
        model = AutoModelForImageClassification.from_pretrained(model_dir)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    _hf_cache[model_dir] = (proc, model, task)
    return _hf_cache[model_dir]


def _hf_auto_annotate(model_dir: str, img_path: str, conf: float,
                      project_classes: list) -> list:
    """Run a HuggingFace model on an image and return suggested annotations.

    Detection models yield real boxes; classification models yield one
    whole-image box labelled with the top-1 class.
    """
    import torch
    from PIL import Image as PILImage
    proc, model, task = _load_hf(model_dir)
    pil = PILImage.open(img_path).convert("RGB")
    inputs = proc(images=pil, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    def _match_class(label_name: str, fallback_id: int):
        for i, c in enumerate(project_classes):
            if c.lower() == str(label_name).lower():
                return i
        return fallback_id

    annotations = []
    with torch.no_grad():
        outputs = model(**inputs)

    if task == "detection":
        target_sizes = torch.tensor([pil.size[::-1]])  # (h, w)
        results = proc.post_process_object_detection(
            outputs, threshold=conf, target_sizes=target_sizes)[0]
        W, H = pil.size
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            x1n, x2n = max(0, x1) / W, min(W, x2) / W
            y1n, y2n = max(0, y1) / H, min(H, y2) / H
            label_name = model.config.id2label.get(int(label), f"cls{int(label)}")
            annotations.append({
                "class_id": _match_class(label_name, int(label)),
                "class_name": label_name,
                "shape_type": "bbox",
                "x_center": round((x1n + x2n) / 2, 6),
                "y_center": round((y1n + y2n) / 2, 6),
                "width":    round(x2n - x1n, 6),
                "height":   round(y2n - y1n, 6),
                "points":   [],
                "conf":     round(float(score), 4),
            })
    else:  # classification → one whole-image box for the top-1 class
        probs = outputs.logits.softmax(-1)[0]
        top_p, top_i = float(probs.max()), int(probs.argmax())
        if top_p >= conf:
            label_name = model.config.id2label.get(top_i, f"cls{top_i}")
            annotations.append({
                "class_id": _match_class(label_name, top_i),
                "class_name": label_name,
                "shape_type": "bbox",
                "x_center": 0.5, "y_center": 0.5, "width": 1.0, "height": 1.0,
                "points": [], "conf": round(top_p, 4),
            })
    return annotations


_sam_cache: dict = {}


def _get_sam():
    """Lazily load and cache MobileSAM (~40 MB) for click-to-segment."""
    if "m" not in _sam_cache:
        from ultralytics import SAM
        _sam_cache["m"] = SAM("mobile_sam.pt")
    return _sam_cache["m"]


def _poly_area(p) -> float:
    a = 0.0
    n = len(p)
    for i in range(n):
        x1, y1 = p[i]; x2, y2 = p[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2


class SamPrompt(BaseModel):
    points: list[list[float]] = []   # normalized [[x,y], ...]
    labels: list[int] = []           # 1 = foreground (add), 0 = background (remove)
    box: Optional[list[float]] = None  # normalized [x1, y1, x2, y2]


@router.post("/projects/{project_id}/images/{image_id}/sam-segment")
async def sam_segment(project_id: int, image_id: int, body: SamPrompt,
                      session: Session = Depends(get_session)):
    """Refinable click-to-segment: positive/negative points and/or a box are
    grouped into ONE prompt so MobileSAM returns a single mask the user can
    refine. Returns the mask polygon (normalized)."""
    img_rec = session.get(ImageModel, image_id)
    if not img_rec or img_rec.project_id != project_id:
        raise HTTPException(404, "Image not found")
    img_path = os.path.join(UPLOAD_DIR, img_rec.filename)
    if not os.path.exists(img_path):
        raise HTTPException(404, "Image file not found")

    import torch, numpy as np
    from PIL import Image as PILImage
    pil = PILImage.open(img_path).convert("RGB")
    W, H = pil.size

    kwargs: dict = {}
    if body.points:
        kwargs["points"] = [[[int(x * W), int(y * H)] for x, y in body.points]]
        labels = body.labels if len(body.labels) == len(body.points) else [1] * len(body.points)
        kwargs["labels"] = [labels]
    if body.box and len(body.box) == 4:
        x1, y1, x2, y2 = body.box
        kwargs["bboxes"] = [[int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)]]
    if not kwargs:
        raise HTTPException(400, "Provide at least one point or a box")

    model  = _get_sam()
    device = "0" if torch.cuda.is_available() else "cpu"
    try:
        res = model.predict(np.array(pil), verbose=False, device=device, **kwargs)[0]
    except (RuntimeError, Exception):
        res = model.predict(np.array(pil), verbose=False, device="cpu", **kwargs)[0]

    if res.masks is None or len(res.masks.xyn) == 0:
        raise HTTPException(404, "No object found — try another point or add a box")

    best = max(res.masks.xyn, key=_poly_area)
    if len(best) > 80:
        best = best[::len(best) // 80]
    pts = [[round(float(a), 5), round(float(b), 5)] for a, b in best]
    return {"points": pts, "count": len(pts)}


@router.post("/projects/{project_id}/images/{image_id}/auto-annotate")
async def auto_annotate(
    project_id: int,
    image_id: int,
    run_id: Optional[int] = None,
    external_model_id: Optional[int] = None,
    conf: float = 0.25,
    session: Session = Depends(get_session),
):
    """Run a detection model on a stored image; returns suggested annotations (not saved).
    Supply either run_id (trained run) or external_model_id (imported .pt file)."""
    if run_id is None and external_model_id is None:
        raise HTTPException(400, "Provide run_id or external_model_id")

    img_rec = session.get(ImageModel, image_id)
    if not img_rec or img_rec.project_id != project_id:
        raise HTTPException(404, "Image not found")

    # Resolve model path
    if run_id is not None:
        run = session.get(TrainingRun, run_id)
        if not run or run.project_id != project_id:
            raise HTTPException(404, "Run not found")
        if run.status != "done":
            raise HTTPException(400, "Run not complete")
        model_path = run.model_path
    else:
        ext_model = session.get(ExternalModel, external_model_id)
        if not ext_model:
            raise HTTPException(404, "External model not found")
        model_path = ext_model.model_path

    if not model_path or not os.path.exists(model_path):
        raise HTTPException(404, "Model file not found on disk")

    img_path = os.path.join(UPLOAD_DIR, img_rec.filename)
    if not os.path.exists(img_path):
        raise HTTPException(404, "Image file not found")

    # Project classes — used to map predicted label names back to class indices.
    proj = session.get(Project, project_id)
    project_classes: list = proj.classes if proj else []

    # HuggingFace model (stored as a directory) → transformers inference.
    if os.path.isdir(model_path):
        try:
            annotations = _hf_auto_annotate(model_path, img_path, conf, project_classes)
        except Exception as exc:
            raise HTTPException(500, f"HuggingFace inference failed: {exc}")
        return {"annotations": annotations, "count": len(annotations)}

    import torch
    from ultralytics import YOLO
    device = "0" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)
    try:
        results = model.predict(img_path, conf=conf, verbose=False, device=device)
    except RuntimeError:
        results = model.predict(img_path, conf=conf, verbose=False, device="cpu")
    r = results[0]

    # Map YOLO class ids back to project classes when using a trained run
    project_classes: list = []
    if run_id is not None:
        proj = session.get(Project, project_id)
        project_classes = proj.classes if proj else []

    annotations = []
    for box in r.boxes:
        x1n, y1n, x2n, y2n = box.xyxyn[0].tolist()
        cls_id = int(box.cls[0])
        annotations.append({
            "class_id": cls_id,
            "class_name": project_classes[cls_id] if cls_id < len(project_classes) else (r.names or {}).get(cls_id, f"cls{cls_id}"),
            "shape_type": "bbox",
            "x_center": round((x1n + x2n) / 2, 6),
            "y_center": round((y1n + y2n) / 2, 6),
            "width":    round(x2n - x1n, 6),
            "height":   round(y2n - y1n, 6),
            "points":   [],
            "conf":     round(float(box.conf[0]), 4),
        })
    return {"annotations": annotations, "count": len(annotations)}


# ─── Live webcam frame inference ──────────────────────────────────────────────
@router.post("/projects/{project_id}/training/runs/{run_id}/webcam-frame")
async def webcam_frame(
    project_id: int,
    run_id: int,
    frame: UploadFile = File(...),
    conf: float = Form(0.25),
    iou:  float = Form(0.45),
    session: Session = Depends(get_session),
):
    """Single-frame webcam inference. Decodes in memory (no temp file) and
    returns JSON detections only — no image round-trip for minimal latency."""
    run = session.get(TrainingRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(404, "Run not found")
    if run.status != "done":
        raise HTTPException(400, "Run not complete")
    if not run.model_path or not os.path.exists(run.model_path):
        raise HTTPException(404, "Model not found")

    project     = session.get(Project, project_id)
    class_names = project.classes if project else []

    import io as _io
    import numpy as _np
    from PIL import Image as _PILImage
    import torch

    raw   = await frame.read()
    img   = _PILImage.open(_io.BytesIO(raw)).convert("RGB")
    frame_np = _np.array(img)

    device = "0" if torch.cuda.is_available() else "cpu"
    model  = _get_model(run.model_path)

    try:
        results = model.predict(frame_np, conf=conf, iou=iou, verbose=False, device=device)
    except RuntimeError:
        results = model.predict(frame_np, conf=conf, iou=iou, verbose=False, device="cpu")

    r = results[0]
    detections = []
    for idx, box in enumerate(r.boxes):
        x1n, y1n, x2n, y2n = box.xyxyn[0].tolist()
        cls_id = int(box.cls[0])
        mask_pts: list = []
        if r.masks is not None and idx < len(r.masks.xyn):
            mask_pts = [[round(float(p[0]), 4), round(float(p[1]), 4)]
                        for p in r.masks.xyn[idx]]
        detections.append({
            "x": round(x1n, 4), "y": round(y1n, 4),
            "w": round(x2n - x1n, 4), "h": round(y2n - y1n, 4),
            "conf": round(float(box.conf[0]), 4),
            "class_id": cls_id,
            "class_name": class_names[cls_id] if cls_id < len(class_names) else f"cls{cls_id}",
            "mask": mask_pts,
        })
    return {"detections": detections, "count": len(detections)}


# ─── Video inference ──────────────────────────────────────────────────────────
def _download_from_url(url: str) -> tuple:
    """Download a video from any yt-dlp supported URL. Returns (video_path, tmp_dir)."""
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("yt-dlp is not installed. Run: pip install yt-dlp")

    dl_dir = os.path.join(VIDEO_OUT_DIR, f"dl_{uuid.uuid4().hex}")
    os.makedirs(dl_dir, exist_ok=True)

    ydl_opts = {
        # Prefer a single pre-merged file so ffmpeg is not required.
        # Falls back to best single-file if the mp4 variant isn't available.
        "format": "best[height<=720][ext=mp4]/best[height<=720]/best[ext=mp4]/best",
        "outtmpl": os.path.join(dl_dir, "video.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "abort_on_error": False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    files = os.listdir(dl_dir)
    if not files:
        raise RuntimeError("Download produced no file — check the URL and try again")
    return os.path.join(dl_dir, files[0]), dl_dir


def _run_video_from_url(job_id: str, url: str, model_path: str,
                        class_names: list, conf: float, iou: float, tracker: bool = False):
    _video_jobs[job_id]["stage"] = "downloading"
    dl_dir = None
    try:
        video_path, dl_dir = _download_from_url(url)
    except Exception as exc:
        _video_jobs[job_id].update({"status": "failed", "error": str(exc)})
        return
    _video_jobs[job_id]["stage"] = "processing"
    _process_video(job_id, video_path, model_path, class_names, conf, iou, tracker)
    if dl_dir and os.path.exists(dl_dir):
        shutil.rmtree(dl_dir, ignore_errors=True)


def _process_video(job_id: str, video_path: str, model_path: str,
                   class_names: list, conf: float, iou: float, tracker: bool = False):
    try:
        import cv2
        import torch

        device = "0" if torch.cuda.is_available() else "cpu"
        model  = _get_model(model_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            _video_jobs[job_id]["status"] = "failed"
            _video_jobs[job_id]["error"]  = "Cannot open video"
            return

        fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        _video_jobs[job_id]["total_frames"] = total

        out_path = os.path.join(VIDEO_OUT_DIR, f"{job_id}.mp4")
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                if tracker:
                    results = model.track(frame, conf=conf, iou=iou,
                                          verbose=False, device=device, persist=True)
                else:
                    results = model.predict(frame, conf=conf, iou=iou,
                                            verbose=False, device=device)
            except RuntimeError:
                if tracker:
                    results = model.track(frame, conf=conf, iou=iou,
                                          verbose=False, device="cpu", persist=True)
                else:
                    results = model.predict(frame, conf=conf, iou=iou,
                                            verbose=False, device="cpu")
            annotated = results[0].plot()
            writer.write(annotated)
            frame_idx += 1
            _video_jobs[job_id]["processed"] = frame_idx

        cap.release()
        writer.release()
        _video_jobs[job_id]["status"]   = "done"
        _video_jobs[job_id]["out_path"] = out_path
    except Exception as exc:
        _video_jobs[job_id]["status"] = "failed"
        _video_jobs[job_id]["error"]  = str(exc)
    finally:
        if os.path.exists(video_path):
            os.unlink(video_path)


@router.post("/projects/{project_id}/training/runs/{run_id}/video-infer")
async def start_video_infer(
    project_id: int,
    run_id: int,
    file: UploadFile = File(...),
    conf:    float = Form(0.25),
    iou:     float = Form(0.45),
    tracker: bool  = Form(False),
    session: Session = Depends(get_session),
):
    run = session.get(TrainingRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(404, "Run not found")
    if run.status != "done":
        raise HTTPException(400, "Run has not completed")

    model_path, is_fallback = _resolve_weights(run)

    try:
        import cv2  # noqa: F401
    except ImportError:
        raise HTTPException(500, "OpenCV (cv2) is not installed on this server. Run: pip install opencv-python")

    project = session.get(Project, project_id)
    class_names = project.classes if project else []

    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    tmp_path = os.path.join(VIDEO_OUT_DIR, f"input_{uuid.uuid4().hex}{suffix}")
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    job_id = uuid.uuid4().hex
    _video_jobs[job_id] = {"status": "running", "stage": "processing", "processed": 0,
                            "total_frames": 0, "out_path": None, "error": None,
                            "fallback_model": model_path if is_fallback else None}

    threading.Thread(
        target=_process_video,
        args=(job_id, tmp_path, model_path, class_names, conf, iou, tracker),
        daemon=True,
    ).start()

    return {"job_id": job_id, "fallback_model": model_path if is_fallback else None}


@router.post("/projects/{project_id}/training/runs/{run_id}/video-infer-url")
async def start_video_infer_url(
    project_id: int,
    run_id: int,
    url:     str   = Form(...),
    conf:    float = Form(0.25),
    iou:     float = Form(0.45),
    tracker: bool  = Form(False),
    session: Session = Depends(get_session),
):
    run = session.get(TrainingRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(404, "Run not found")
    if run.status != "done":
        raise HTTPException(400, "Run has not completed")

    model_path, is_fallback = _resolve_weights(run)

    try:
        import cv2  # noqa: F401
    except ImportError:
        raise HTTPException(500, "OpenCV (cv2) is not installed on this server. Run: pip install opencv-python")

    project     = session.get(Project, project_id)
    class_names = project.classes if project else []

    job_id = uuid.uuid4().hex
    _video_jobs[job_id] = {"status": "running", "stage": "downloading", "processed": 0,
                            "total_frames": 0, "out_path": None, "error": None,
                            "fallback_model": model_path if is_fallback else None}

    threading.Thread(
        target=_run_video_from_url,
        args=(job_id, url, model_path, class_names, conf, iou, tracker),
        daemon=True,
    ).start()

    return {"job_id": job_id, "fallback_model": model_path if is_fallback else None}


@router.get("/projects/{project_id}/training/runs/{run_id}/video-infer/{job_id}/status")
def video_infer_status(project_id: int, run_id: int, job_id: str):
    job = _video_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "status":       job["status"],
        "stage":        job.get("stage", "processing"),
        "processed":    job["processed"],
        "total_frames": job["total_frames"],
        "error":        job.get("error"),
    }


@router.get("/projects/{project_id}/training/runs/{run_id}/video-infer/{job_id}/download")
def video_infer_download(project_id: int, run_id: int, job_id: str):
    job = _video_jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "Result not ready")
    out_path = job["out_path"]
    if not out_path or not os.path.exists(out_path):
        raise HTTPException(404, "Output file not found")
    return FileResponse(out_path, filename=f"annotated_{job_id[:8]}.mp4",
                        media_type="video/mp4")
