from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
import os, shutil, threading, uuid, io
from pathlib import Path
from routers.infer import _download_from_url, VIDEO_OUT_DIR

router = APIRouter(tags=["segmentation"])

CUSTOM_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "custom_seg_models")
os.makedirs(CUSTOM_MODEL_DIR, exist_ok=True)

_seg_jobs:  dict = {}
_seg_cache: dict = {}   # model_name → YOLO instance


def _is_fastsam(model_name: str) -> bool:
    """FastSAM 'Segment Everything' model (class-agnostic, no training needed)."""
    return "fastsam" in os.path.basename(str(model_name)).lower()


def _get_seg_model(model_name: str):
    if model_name not in _seg_cache:
        if _is_fastsam(model_name):
            from ultralytics import FastSAM
            _seg_cache[model_name] = FastSAM(model_name)
        else:
            from ultralytics import YOLO
            _seg_cache[model_name] = YOLO(model_name)
    return _seg_cache[model_name]


def _seg_predict(model, model_name, arr, conf, device):
    """Run prediction with model-appropriate args. FastSAM gets full-resolution
    (retina) masks for crisp 'segment everything' output."""
    extra = {"retina_masks": True} if _is_fastsam(model_name) else {}
    try:
        return model.predict(arr, conf=conf, verbose=False, device=device, **extra)
    except RuntimeError:
        return model.predict(arr, conf=conf, verbose=False, device="cpu", **extra)


def _extract_segments(results):
    """Extract detections + mask polygons from YOLO seg results."""
    r = results[0]
    ih, iw = r.orig_shape
    detections: list = []
    for idx, box in enumerate(r.boxes):
        x1n, y1n, x2n, y2n = box.xyxyn[0].tolist()
        cls_id   = int(box.cls[0])
        cls_name = r.names.get(cls_id, f"cls{cls_id}")
        mask_pts: list = []
        if r.masks is not None and idx < len(r.masks.xyn):
            mask_pts = [[round(float(p[0]), 4), round(float(p[1]), 4)]
                        for p in r.masks.xyn[idx]]
        detections.append({
            "x": round(x1n, 4), "y": round(y1n, 4),
            "w": round(x2n - x1n, 4), "h": round(y2n - y1n, 4),
            "conf": round(float(box.conf[0]), 4),
            "class_id": cls_id,
            "class_name": cls_name,
            "mask": mask_pts,
        })
    return detections, iw, ih


# ─── Upload custom model ──────────────────────────────────────────────────────
@router.post("/segment/upload-custom-model")
async def upload_custom_seg_model(file: UploadFile = File(...)):
    """Save a user-supplied .pt weights file and return its server path."""
    if not (file.filename or "").lower().endswith(".pt"):
        raise HTTPException(400, "Only .pt files are accepted")
    safe_name = f"{uuid.uuid4().hex}_{Path(file.filename).name}"
    dest = os.path.join(CUSTOM_MODEL_DIR, safe_name)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"model_path": dest, "filename": Path(file.filename).name}


# ─── Image inference ──────────────────────────────────────────────────────────
@router.post("/segment/infer")
async def segment_image_infer(
    file:       UploadFile = File(...),
    model_name: str   = Form("yolo11n-seg.pt"),
    conf:       float = Form(0.25),
):
    import torch, numpy as np
    from PIL import Image as PILImage

    raw = await file.read()
    img = PILImage.open(io.BytesIO(raw)).convert("RGB")
    arr = np.array(img)

    device = "0" if torch.cuda.is_available() else "cpu"
    model  = _get_seg_model(model_name)
    results = _seg_predict(model, model_name, arr, conf, device)

    detections, iw, ih = _extract_segments(results)
    return {"detections": detections, "count": len(detections),
            "image_w": iw, "image_h": ih}


# ─── Image URL inference ──────────────────────────────────────────────────────
@router.post("/segment/infer-url")
async def segment_image_infer_url(
    url:        str   = Form(...),
    model_name: str   = Form("yolo11n-seg.pt"),
    conf:       float = Form(0.25),
):
    import torch, numpy as np
    from PIL import Image as PILImage
    import urllib.request

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            img_bytes = resp.read()
    except Exception as exc:
        raise HTTPException(400, f"Failed to fetch image: {exc}")

    try:
        img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"URL did not return a valid image: {exc}")
    arr = np.array(img)

    device = "0" if torch.cuda.is_available() else "cpu"
    model  = _get_seg_model(model_name)
    results = _seg_predict(model, model_name, arr, conf, device)

    detections, iw, ih = _extract_segments(results)
    return {"detections": detections, "count": len(detections),
            "image_w": iw, "image_h": ih}


# ─── Webcam frame ─────────────────────────────────────────────────────────────
@router.post("/segment/webcam-frame")
async def segment_webcam_frame(
    frame:      UploadFile = File(...),
    model_name: str   = Form("yolo11n-seg.pt"),
    conf:       float = Form(0.25),
):
    import torch, numpy as np
    from PIL import Image as PILImage

    raw = await frame.read()
    img = PILImage.open(io.BytesIO(raw)).convert("RGB")
    arr = np.array(img)

    device = "0" if torch.cuda.is_available() else "cpu"
    model  = _get_seg_model(model_name)
    try:
        results = model.predict(arr, conf=conf, verbose=False, device=device)
    except RuntimeError:
        results = model.predict(arr, conf=conf, verbose=False, device="cpu")

    detections, _, _ = _extract_segments(results)
    return {"detections": detections, "count": len(detections)}


# ─── Video processing ─────────────────────────────────────────────────────────
def _process_seg_video(job_id: str, video_path: str, model_name: str, conf: float):
    try:
        import cv2, torch
        device = "0" if torch.cuda.is_available() else "cpu"
        model  = _get_seg_model(model_name)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            _seg_jobs[job_id].update({"status": "failed", "error": "Cannot open video"})
            return

        fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        _seg_jobs[job_id]["total_frames"] = total

        out_path = os.path.join(VIDEO_OUT_DIR, f"seg_{job_id}.mp4")
        writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                   fps, (width, height))
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                results = model.track(frame, conf=conf, verbose=False,
                                      device=device, persist=True)
            except RuntimeError:
                results = model.track(frame, conf=conf, verbose=False,
                                      device="cpu", persist=True)
            writer.write(results[0].plot())
            frame_idx += 1
            _seg_jobs[job_id]["processed"] = frame_idx

        cap.release()
        writer.release()
        _seg_jobs[job_id].update({"status": "done", "out_path": out_path})
    except Exception as exc:
        _seg_jobs[job_id].update({"status": "failed", "error": str(exc)})
    finally:
        if os.path.exists(video_path):
            os.unlink(video_path)


def _run_seg_from_url(job_id: str, url: str, model_name: str, conf: float):
    _seg_jobs[job_id]["stage"] = "downloading"
    dl_dir = None
    try:
        video_path, dl_dir = _download_from_url(url)
    except Exception as exc:
        _seg_jobs[job_id].update({"status": "failed", "error": str(exc)})
        return
    _seg_jobs[job_id]["stage"] = "processing"
    _process_seg_video(job_id, video_path, model_name, conf)
    if dl_dir and os.path.exists(dl_dir):
        shutil.rmtree(dl_dir, ignore_errors=True)


# ─── Video endpoints ──────────────────────────────────────────────────────────
@router.post("/segment/video-infer")
async def start_seg_video(
    file:       UploadFile = File(...),
    model_name: str   = Form("yolo11n-seg.pt"),
    conf:       float = Form(0.25),
):
    try:
        import cv2  # noqa
    except ImportError:
        raise HTTPException(500, "OpenCV (cv2) is not installed")

    suffix   = Path(file.filename or "video.mp4").suffix or ".mp4"
    tmp_path = os.path.join(VIDEO_OUT_DIR, f"seg_in_{uuid.uuid4().hex}{suffix}")
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    job_id = uuid.uuid4().hex
    _seg_jobs[job_id] = {"status": "running", "stage": "processing", "processed": 0,
                         "total_frames": 0, "out_path": None, "error": None}

    threading.Thread(target=_process_seg_video,
                     args=(job_id, tmp_path, model_name, conf),
                     daemon=True).start()
    return {"job_id": job_id}


@router.post("/segment/video-infer-url")
async def start_seg_video_url(
    url:        str   = Form(...),
    model_name: str   = Form("yolo11n-seg.pt"),
    conf:       float = Form(0.25),
):
    try:
        import cv2  # noqa
    except ImportError:
        raise HTTPException(500, "OpenCV (cv2) is not installed")

    job_id = uuid.uuid4().hex
    _seg_jobs[job_id] = {"status": "running", "stage": "downloading", "processed": 0,
                         "total_frames": 0, "out_path": None, "error": None}

    threading.Thread(target=_run_seg_from_url,
                     args=(job_id, url, model_name, conf),
                     daemon=True).start()
    return {"job_id": job_id}


@router.get("/segment/video-infer/{job_id}/status")
def seg_video_status(job_id: str):
    job = _seg_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {"status": job["status"], "stage": job.get("stage", "processing"),
            "processed": job["processed"], "total_frames": job["total_frames"],
            "error": job.get("error")}


@router.get("/segment/video-infer/{job_id}/download")
def seg_video_download(job_id: str):
    job = _seg_jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "Result not ready")
    out_path = job["out_path"]
    if not out_path or not os.path.exists(out_path):
        raise HTTPException(404, "Output file not found")
    return FileResponse(out_path, filename=f"seg_{job_id[:8]}.mp4",
                        media_type="video/mp4")
