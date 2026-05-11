from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Form
from fastapi.responses import FileResponse
from sqlmodel import Session
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
    if not run.model_path or not os.path.exists(run.model_path):
        raise HTTPException(404, "Model weights not found")

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
        model = YOLO(run.model_path)
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
    if not run.model_path or not os.path.exists(run.model_path):
        raise HTTPException(404, "Model weights not found")

    project = session.get(Project, project_id)
    class_names = project.classes if project else []

    try:
        img_bytes = _fetch_image_from_url(url)
    except Exception as exc:
        raise HTTPException(400, f"Failed to fetch image: {exc}")

    import io, torch, numpy as np
    from PIL import Image as PILImage
    img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
    arr = np.array(img)

    from ultralytics import YOLO
    device = "0" if torch.cuda.is_available() else "cpu"
    model  = YOLO(run.model_path)
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
    if not run.model_path or not os.path.exists(run.model_path):
        raise HTTPException(404, "Model weights not found")

    project     = session.get(Project, project_id)
    class_names = project.classes if project else []

    import torch
    from ultralytics import YOLO
    device = "0" if torch.cuda.is_available() else "cpu"
    model  = YOLO(run.model_path)

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
    if not run.model_path or not os.path.exists(run.model_path):
        raise HTTPException(404, "Model weights not found")

    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    classes = project.classes

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
        model = model_fn_map.get(run.base_model, M.resnet18)(weights=None)
        n_classes = len(classes)
        if hasattr(model, "fc"):
            model.fc = nn.Linear(model.fc.in_features, n_classes)
        elif hasattr(model, "classifier"):
            in_feat = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_feat, n_classes)

        model.load_state_dict(torch.load(run.model_path, map_location=device, weights_only=False))
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
    finally:
        os.unlink(tmp_path)


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
    if not run.model_path or not os.path.exists(run.model_path):
        raise HTTPException(404, "Model weights not found")

    project = session.get(Project, project_id)
    cfg     = session.get(CustomModelConfig, run.config_id)
    if not project or not cfg:
        raise HTTPException(404, "Project or config not found")

    classes = project.classes
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
        model.load_state_dict(torch.load(run.model_path, map_location=device, weights_only=False))
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
        "format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]/best",
        "outtmpl": os.path.join(dl_dir, "video.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
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
    if not run.model_path or not os.path.exists(run.model_path):
        raise HTTPException(404, "Model weights not found")

    try:
        import cv2  # noqa: F401
    except ImportError:
        raise HTTPException(500, "OpenCV (cv2) is not installed on this server")

    project = session.get(Project, project_id)
    class_names = project.classes if project else []

    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    tmp_path = os.path.join(VIDEO_OUT_DIR, f"input_{uuid.uuid4().hex}{suffix}")
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    job_id = uuid.uuid4().hex
    _video_jobs[job_id] = {"status": "running", "stage": "processing", "processed": 0,
                            "total_frames": 0, "out_path": None, "error": None}

    threading.Thread(
        target=_process_video,
        args=(job_id, tmp_path, run.model_path, class_names, conf, iou, tracker),
        daemon=True,
    ).start()

    return {"job_id": job_id}


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
    if not run.model_path or not os.path.exists(run.model_path):
        raise HTTPException(404, "Model weights not found")

    project     = session.get(Project, project_id)
    class_names = project.classes if project else []

    job_id = uuid.uuid4().hex
    _video_jobs[job_id] = {"status": "running", "stage": "downloading", "processed": 0,
                            "total_frames": 0, "out_path": None, "error": None}

    threading.Thread(
        target=_run_video_from_url,
        args=(job_id, url, run.model_path, class_names, conf, iou, tracker),
        daemon=True,
    ).start()

    return {"job_id": job_id}


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
