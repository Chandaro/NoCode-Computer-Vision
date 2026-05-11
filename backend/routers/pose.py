from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
import os, shutil, threading, uuid, io
from pathlib import Path
from routers.infer import _download_from_url

router = APIRouter(tags=["pose"])

VIDEO_OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "video_out")
os.makedirs(VIDEO_OUT_DIR, exist_ok=True)

_pose_jobs:   dict = {}
_pose_cache:  dict = {}   # model_name → YOLO instance

# COCO 17-keypoint skeleton connections (0-indexed)
SKELETON = [
    [0, 1], [0, 2], [1, 3], [2, 4],          # face
    [5, 7], [7, 9], [6, 8], [8, 10],          # arms
    [5, 6], [5, 11], [6, 12], [11, 12],        # torso
    [11, 13], [13, 15], [12, 14], [14, 16],    # legs
]


def _get_pose_model(model_name: str):
    if model_name not in _pose_cache:
        from ultralytics import YOLO
        _pose_cache[model_name] = YOLO(model_name)
    return _pose_cache[model_name]


def _predict_keypoints(results):
    """Extract person boxes + keypoints from YOLO pose results."""
    r = results[0]
    ih, iw = r.orig_shape
    persons = []
    for i, box in enumerate(r.boxes):
        x1n, y1n, x2n, y2n = box.xyxyn[0].tolist()
        kps: list = []
        kp_conf: list = []
        if r.keypoints is not None and i < len(r.keypoints.xyn):
            for pt in r.keypoints.xyn[i]:
                kps.append([round(float(pt[0]), 4), round(float(pt[1]), 4)])
        if (r.keypoints is not None
                and hasattr(r.keypoints, "conf")
                and r.keypoints.conf is not None
                and i < len(r.keypoints.conf)):
            kp_conf = [round(float(c), 3) for c in r.keypoints.conf[i]]
        persons.append({
            "x": round(x1n, 4), "y": round(y1n, 4),
            "w": round(x2n - x1n, 4), "h": round(y2n - y1n, 4),
            "conf": round(float(box.conf[0]), 4),
            "keypoints": kps,
            "kp_conf": kp_conf,
        })
    return persons, iw, ih


# ─── Image inference ──────────────────────────────────────────────────────────
@router.post("/pose/infer-url")
async def pose_image_infer_url(
    url:        str   = Form(...),
    model_name: str   = Form("yolo11n-pose.pt"),
    conf:       float = Form(0.25),
):
    import torch, io, numpy as np
    from PIL import Image as PILImage
    import urllib.request

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            img_bytes = resp.read()
    except Exception as exc:
        from fastapi import HTTPException
        raise HTTPException(400, f"Failed to fetch image: {exc}")

    try:
        img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as exc:
        from fastapi import HTTPException as _HTTPException
        raise _HTTPException(400, f"URL did not return a valid image: {exc}")
    arr = np.array(img)

    device = "0" if torch.cuda.is_available() else "cpu"
    model  = _get_pose_model(model_name)
    try:
        results = model.predict(arr, conf=conf, verbose=False, device=device)
    except RuntimeError:
        results = model.predict(arr, conf=conf, verbose=False, device="cpu")

    persons, iw, ih = _predict_keypoints(results)
    return {"persons": persons, "count": len(persons),
            "image_w": iw, "image_h": ih, "skeleton": SKELETON}


@router.post("/pose/infer")
async def pose_image_infer(
    file:       UploadFile = File(...),
    model_name: str   = Form("yolo11n-pose.pt"),
    conf:       float = Form(0.25),
):
    import torch, numpy as np
    from PIL import Image as PILImage

    raw  = await file.read()
    img  = PILImage.open(io.BytesIO(raw)).convert("RGB")
    arr  = np.array(img)

    device = "0" if torch.cuda.is_available() else "cpu"
    model  = _get_pose_model(model_name)

    try:
        results = model.predict(arr, conf=conf, verbose=False, device=device)
    except RuntimeError:
        results = model.predict(arr, conf=conf, verbose=False, device="cpu")

    persons, iw, ih = _predict_keypoints(results)
    return {"persons": persons, "count": len(persons),
            "image_w": iw, "image_h": ih, "skeleton": SKELETON}


# ─── Webcam frame ─────────────────────────────────────────────────────────────
@router.post("/pose/webcam-frame")
async def pose_webcam_frame(
    frame:      UploadFile = File(...),
    model_name: str   = Form("yolo11n-pose.pt"),
    conf:       float = Form(0.25),
):
    import torch, numpy as np
    from PIL import Image as PILImage

    raw  = await frame.read()
    img  = PILImage.open(io.BytesIO(raw)).convert("RGB")
    arr  = np.array(img)

    device = "0" if torch.cuda.is_available() else "cpu"
    model  = _get_pose_model(model_name)

    try:
        results = model.predict(arr, conf=conf, verbose=False, device=device)
    except RuntimeError:
        results = model.predict(arr, conf=conf, verbose=False, device="cpu")

    persons, _, _ = _predict_keypoints(results)
    return {"persons": persons, "count": len(persons), "skeleton": SKELETON}


# ─── Video inference ──────────────────────────────────────────────────────────
def _run_pose_from_url(job_id: str, url: str, model_name: str, conf: float):
    _pose_jobs[job_id]["stage"] = "downloading"
    dl_dir = None
    try:
        video_path, dl_dir = _download_from_url(url)
    except Exception as exc:
        _pose_jobs[job_id].update({"status": "failed", "error": str(exc)})
        return
    _pose_jobs[job_id]["stage"] = "processing"
    _process_pose_video(job_id, video_path, model_name, conf)
    if dl_dir and os.path.exists(dl_dir):
        shutil.rmtree(dl_dir, ignore_errors=True)


def _process_pose_video(job_id: str, video_path: str, model_name: str, conf: float):
    try:
        import cv2, torch
        device = "0" if torch.cuda.is_available() else "cpu"
        model  = _get_pose_model(model_name)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            _pose_jobs[job_id].update({"status": "failed", "error": "Cannot open video"})
            return

        fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        _pose_jobs[job_id]["total_frames"] = total

        out_path = os.path.join(VIDEO_OUT_DIR, f"pose_{job_id}.mp4")
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
            _pose_jobs[job_id]["processed"] = frame_idx

        cap.release()
        writer.release()
        _pose_jobs[job_id].update({"status": "done", "out_path": out_path})
    except Exception as exc:
        _pose_jobs[job_id].update({"status": "failed", "error": str(exc)})
    finally:
        if os.path.exists(video_path):
            os.unlink(video_path)


@router.post("/pose/video-infer")
async def start_pose_video(
    file:       UploadFile = File(...),
    model_name: str   = Form("yolo11n-pose.pt"),
    conf:       float = Form(0.25),
):
    try:
        import cv2  # noqa
    except ImportError:
        raise HTTPException(500, "OpenCV (cv2) is not installed")

    suffix   = Path(file.filename or "video.mp4").suffix or ".mp4"
    tmp_path = os.path.join(VIDEO_OUT_DIR, f"pose_in_{uuid.uuid4().hex}{suffix}")
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    job_id = uuid.uuid4().hex
    _pose_jobs[job_id] = {"status": "running", "stage": "processing", "processed": 0,
                          "total_frames": 0, "out_path": None, "error": None}

    threading.Thread(target=_process_pose_video,
                     args=(job_id, tmp_path, model_name, conf),
                     daemon=True).start()
    return {"job_id": job_id}


@router.post("/pose/video-infer-url")
async def start_pose_video_url(
    url:        str   = Form(...),
    model_name: str   = Form("yolo11n-pose.pt"),
    conf:       float = Form(0.25),
):
    try:
        import cv2  # noqa
    except ImportError:
        raise HTTPException(500, "OpenCV (cv2) is not installed")

    job_id = uuid.uuid4().hex
    _pose_jobs[job_id] = {"status": "running", "stage": "downloading", "processed": 0,
                          "total_frames": 0, "out_path": None, "error": None}

    threading.Thread(target=_run_pose_from_url,
                     args=(job_id, url, model_name, conf),
                     daemon=True).start()
    return {"job_id": job_id}


@router.get("/pose/video-infer/{job_id}/status")
def pose_video_status(job_id: str):
    job = _pose_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {"status": job["status"], "stage": job.get("stage", "processing"),
            "processed": job["processed"],
            "total_frames": job["total_frames"], "error": job.get("error")}


@router.get("/pose/video-infer/{job_id}/download")
def pose_video_download(job_id: str):
    job = _pose_jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "Result not ready")
    out_path = job["out_path"]
    if not out_path or not os.path.exists(out_path):
        raise HTTPException(404, "Output file not found")
    return FileResponse(out_path, filename=f"pose_{job_id[:8]}.mp4",
                        media_type="video/mp4")
