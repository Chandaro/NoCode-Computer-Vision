from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
import os, shutil, threading, uuid, io, importlib.util
from pathlib import Path
from routers.infer import _download_from_url, VIDEO_OUT_DIR
from routers.segmentation import _extract_segments

# "Find Anything": open-vocabulary detection + segmentation with YOLOE-26.
# The user types what to look for ("helmet, red car") instead of training a model.
router = APIRouter(tags=["find"])

DEFAULT_MODEL = "yoloe-26s-seg.pt"
MODELS = ["yoloe-26n-seg.pt", "yoloe-26s-seg.pt", "yoloe-26m-seg.pt", "yoloe-26l-seg.pt"]
TEXT_ENCODER = "mobileclip2_b.ts"   # downloaded by ultralytics on the first set_classes()
MAX_PROMPTS  = 20

_find_jobs:  dict = {}
_find_cache: dict = {}   # model_name → YOLOE instance (shared by image / webcam requests)
# set_classes() mutates the model, so prompt-switch + predict must not interleave
# between concurrent requests (e.g. webcam frames arriving while an image runs).
_find_lock = threading.Lock()


def _parse_prompts(prompts: str) -> list[str]:
    names: list[str] = []
    for p in prompts.replace("\n", ",").split(","):
        p = p.strip()
        if p and p.lower() not in (n.lower() for n in names):
            names.append(p)
    if not names:
        raise HTTPException(400, "Type at least one thing to find, e.g. \"person, bicycle\".")
    if len(names) > MAX_PROMPTS:
        raise HTTPException(400, f"Too many things at once. Use up to {MAX_PROMPTS}.")
    return names


def _check_model_name(model_name: str):
    if model_name not in MODELS:
        raise HTTPException(400, f"Unknown model '{model_name}'. Choose one of: {', '.join(MODELS)}")


def _load_model(model_name: str):
    try:
        from ultralytics import YOLOE
        return YOLOE(model_name)
    except Exception as exc:
        raise HTTPException(503, f"Could not load {model_name}. The first use downloads it, "
                                 f"so check your internet connection. ({exc})")


def _set_prompts(model, names: list[str]):
    try:
        model.set_classes(names)
    except Exception as exc:
        raise HTTPException(503, "Could not load the text model that reads your prompts. "
                                 "The first use downloads about 250 MB, so check your internet "
                                 f"connection and try again. ({exc})")


def _get_find_model(model_name: str):
    if model_name not in _find_cache:
        _find_cache[model_name] = _load_model(model_name)
    return _find_cache[model_name]


def _predict(model, arr, conf, device):
    try:
        return model.predict(arr, conf=conf, verbose=False, device=device)
    except RuntimeError:   # e.g. CUDA out of memory → retry on CPU
        return model.predict(arr, conf=conf, verbose=False, device="cpu")


def _find_in_array(arr, names: list[str], model_name: str, conf: float):
    import torch
    device = "0" if torch.cuda.is_available() else "cpu"
    with _find_lock:
        model = _get_find_model(model_name)
        _set_prompts(model, names)
        results = _predict(model, arr, conf, device)
    return _extract_segments(results)


def _summary(detections: list, names: list[str]) -> dict:
    counts = {n: 0 for n in names}
    for d in detections:
        counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1
    return counts


# ─── Readiness (lets the UI warn about the one-time download) ─────────────────
@router.get("/find/status")
def find_status():
    return {
        "models": MODELS,
        "default_model": DEFAULT_MODEL,
        "downloaded": [m for m in MODELS if os.path.isfile(m)],
        "text_encoder_ready": os.path.isfile(TEXT_ENCODER)
                              and importlib.util.find_spec("clip") is not None,
        "loaded": list(_find_cache.keys()),
    }


# ─── Image inference ──────────────────────────────────────────────────────────
@router.post("/find/infer")
async def find_image(
    file:       UploadFile = File(...),
    prompts:    str   = Form(...),
    model_name: str   = Form(DEFAULT_MODEL),
    conf:       float = Form(0.25),
):
    import numpy as np
    from PIL import Image as PILImage

    _check_model_name(model_name)
    names = _parse_prompts(prompts)
    try:
        img = PILImage.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(400, "That file is not an image the app can read.")
    detections, iw, ih = _find_in_array(np.array(img), names, model_name, conf)
    return {"detections": detections, "count": len(detections),
            "counts": _summary(detections, names), "prompts": names,
            "image_w": iw, "image_h": ih}


@router.post("/find/infer-url")
async def find_image_url(
    url:        str   = Form(...),
    prompts:    str   = Form(...),
    model_name: str   = Form(DEFAULT_MODEL),
    conf:       float = Form(0.25),
):
    import numpy as np
    from PIL import Image as PILImage
    import urllib.request

    _check_model_name(model_name)
    names = _parse_prompts(prompts)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            img = PILImage.open(io.BytesIO(resp.read())).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"Could not get an image from that link: {exc}")
    detections, iw, ih = _find_in_array(np.array(img), names, model_name, conf)
    return {"detections": detections, "count": len(detections),
            "counts": _summary(detections, names), "prompts": names,
            "image_w": iw, "image_h": ih}


@router.post("/find/webcam-frame")
async def find_webcam_frame(
    frame:      UploadFile = File(...),
    prompts:    str   = Form(...),
    model_name: str   = Form(DEFAULT_MODEL),
    conf:       float = Form(0.25),
):
    import numpy as np
    from PIL import Image as PILImage

    _check_model_name(model_name)
    names = _parse_prompts(prompts)
    img = PILImage.open(io.BytesIO(await frame.read())).convert("RGB")
    detections, _, _ = _find_in_array(np.array(img), names, model_name, conf)
    return {"detections": detections, "count": len(detections),
            "counts": _summary(detections, names)}


# ─── Video processing ─────────────────────────────────────────────────────────
def _process_find_video(job_id: str, video_path: str, names: list[str],
                        model_name: str, conf: float):
    try:
        import cv2, torch
        device = "0" if torch.cuda.is_available() else "cpu"
        # A video holds its prompts for minutes, so it gets its own model instance
        # instead of locking the shared one away from image / webcam requests.
        from ultralytics import YOLOE
        model = YOLOE(model_name)
        model.set_classes(names)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            _find_jobs[job_id].update({"status": "failed", "error": "Cannot open video"})
            return

        fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _find_jobs[job_id]["total_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        out_path = os.path.join(VIDEO_OUT_DIR, f"find_{job_id}.mp4")
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
            _find_jobs[job_id]["processed"] = frame_idx

        cap.release()
        writer.release()
        _find_jobs[job_id].update({"status": "done", "out_path": out_path})
    except Exception as exc:
        _find_jobs[job_id].update({"status": "failed", "error": str(exc)})
    finally:
        if os.path.exists(video_path):
            os.unlink(video_path)


def _run_find_from_url(job_id: str, url: str, names: list[str], model_name: str, conf: float):
    _find_jobs[job_id]["stage"] = "downloading"
    dl_dir = None
    try:
        video_path, dl_dir = _download_from_url(url)
    except Exception as exc:
        _find_jobs[job_id].update({"status": "failed", "error": str(exc)})
        return
    _find_jobs[job_id]["stage"] = "processing"
    _process_find_video(job_id, video_path, names, model_name, conf)
    if dl_dir and os.path.exists(dl_dir):
        shutil.rmtree(dl_dir, ignore_errors=True)


def _new_job(stage: str) -> str:
    job_id = uuid.uuid4().hex
    _find_jobs[job_id] = {"status": "running", "stage": stage, "processed": 0,
                          "total_frames": 0, "out_path": None, "error": None}
    return job_id


@router.post("/find/video-infer")
async def start_find_video(
    file:       UploadFile = File(...),
    prompts:    str   = Form(...),
    model_name: str   = Form(DEFAULT_MODEL),
    conf:       float = Form(0.25),
):
    _check_model_name(model_name)
    names    = _parse_prompts(prompts)
    suffix   = Path(file.filename or "video.mp4").suffix or ".mp4"
    tmp_path = os.path.join(VIDEO_OUT_DIR, f"find_in_{uuid.uuid4().hex}{suffix}")
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    job_id = _new_job("processing")
    threading.Thread(target=_process_find_video,
                     args=(job_id, tmp_path, names, model_name, conf),
                     daemon=True).start()
    return {"job_id": job_id}


@router.post("/find/video-infer-url")
async def start_find_video_url(
    url:        str   = Form(...),
    prompts:    str   = Form(...),
    model_name: str   = Form(DEFAULT_MODEL),
    conf:       float = Form(0.25),
):
    _check_model_name(model_name)
    names  = _parse_prompts(prompts)
    job_id = _new_job("downloading")
    threading.Thread(target=_run_find_from_url,
                     args=(job_id, url, names, model_name, conf),
                     daemon=True).start()
    return {"job_id": job_id}


@router.get("/find/video-infer/{job_id}/status")
def find_video_status(job_id: str):
    job = _find_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {"status": job["status"], "stage": job.get("stage", "processing"),
            "processed": job["processed"], "total_frames": job["total_frames"],
            "error": job.get("error")}


@router.get("/find/video-infer/{job_id}/download")
def find_video_download(job_id: str):
    job = _find_jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "Result not ready")
    out_path = job["out_path"]
    if not out_path or not os.path.exists(out_path):
        raise HTTPException(404, "Output file not found")
    return FileResponse(out_path, filename=f"find_{job_id[:8]}.mp4", media_type="video/mp4")
