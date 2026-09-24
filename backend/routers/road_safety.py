"""
Road safety vision.

A focused mode for dashcam / roadside footage: detect and TRACK the road users
in the scene (people, bicycles, motorcycles, cars, buses, trucks) and put a
clear indicator on each one. When helmet information is available it colour
codes riders so you can see at a glance who is protected and who is not.

Helmet status comes from, in priority order:
  1. The selected model itself, if its classes look like helmet / no-helmet.
     (Train one in this app: upload helmet + no-helmet crops and train a
      detection model - it plugs straight in.)
  2. Zero-shot CLIP, if transformers can load a CLIP model. Works with no
     training at all, but needs the model available (first use downloads it).
  3. Otherwise "unknown" - tracking and indicators still work.

Nothing here needs a special dataset to be useful: with the built-in COCO
fallback you already get person / bicycle / motorcycle / vehicle tracking.
"""
from __future__ import annotations

import os
import re
import tempfile
import shutil
import threading
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from sqlmodel import Session

from database import get_session
from models import Project, TrainingRun

router = APIRouter(tags=["road-safety"])

VIDEO_OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "video_out")
os.makedirs(VIDEO_OUT_DIR, exist_ok=True)

_video_jobs: dict = {}

# Road-user classes we care about (names as they appear in COCO).
ROAD_USERS = ["person", "bicycle", "motorcycle", "car", "bus", "truck"]
# Two-wheelers worth a helmet check.
TWO_WHEELERS = ["bicycle", "motorcycle"]

_HELMET_YES = re.compile(r"helmet|hard.?hat", re.I)
_HELMET_NO = re.compile(r"no.?helmet|without.?helmet|bare.?head|no.?hat", re.I)


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_model_and_classes(project_id: int, run_id, session: Session):
    """Return (model, class_names, source_note).

    Uses the given training run if it has weights, otherwise the project, and
    finally falls back to the built-in COCO model so the page works instantly.
    """
    from routers.infer import _get_model, _resolve_weights

    if run_id:
        run = session.get(TrainingRun, run_id)
        if run and run.project_id == project_id and run.status == "done":
            path, is_fallback = _resolve_weights(run)
            project = session.get(Project, project_id)
            names = (project.classes if project else []) or []
            # a real trained model has its own weights; if it fell back to the
            # base COCO model we should use COCO's names, not the project's
            if is_fallback:
                m = _get_model("yolov8n.pt")
                return m, list(m.names.values()), f"COCO fallback ({path})"
            m = _get_model(path)
            return m, names, os.path.basename(path)

    m = _get_model("yolov8n.pt")
    return m, list(m.names.values()), "COCO yolov8n (no trained model selected)"


def _name_for(cls_id: int, class_names: list) -> str:
    if 0 <= cls_id < len(class_names):
        return class_names[cls_id]
    return f"cls{cls_id}"


# ── Helmet helpers ────────────────────────────────────────────────────────────

def _overlap_ratio(a: list, b: list) -> float:
    """Fraction of box `a` covered by box `b` (0-1)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    area = (a[2] - a[0]) * (a[3] - a[1])
    return (iw * ih) / area if area > 0 else 0.0


def _head_region(person_box: list) -> list:
    """Approximate the head: the top ~35% of a person box, slightly inset."""
    x1, y1, x2, y2 = person_box
    h = y2 - y1
    w = x2 - x1
    return [x1 + 0.2 * w, y1, x2 - 0.2 * w, y1 + 0.35 * h]


def _helmet_from_model(person_box, dets, class_names):
    """Look for a helmet / no-helmet detection over the person's head."""
    head = _head_region(person_box)
    best, best_ratio = None, 0.0
    for d in dets:
        name = _name_for(d["cls"], class_names)
        # Check the negative pattern FIRST: "no_helmet" also contains "helmet",
        # so matching positives first would wrongly label it as protected.
        if _HELMET_NO.search(name):
            r = _overlap_ratio(head, [d["x1"], d["y1"], d["x2"], d["y2"]])
            if r > best_ratio:
                best, best_ratio = "no", r
        elif _HELMET_YES.search(name):
            r = _overlap_ratio(head, [d["x1"], d["y1"], d["x2"], d["y2"]])
            if r > best_ratio:
                best, best_ratio = "yes", r
    return best if best_ratio > 0.15 else None


# Lazy CLIP cache so the model loads at most once per server run.
_clip = {"ready": None, "model": None, "proc": None}


def _clip_importable() -> bool:
    """Cheap check: is transformers available? Does NOT load/download CLIP."""
    try:
        import importlib.util
        return importlib.util.find_spec("transformers") is not None
    except Exception:
        return False


def _clip_ready() -> bool:
    """True once CLIP is actually loaded. Loads (and may download) on first call,
    so only call this when the user has explicitly asked for a zero-shot check."""
    if _clip["ready"] is not None:
        return _clip["ready"]
    try:
        import transformers  # noqa: F401
        from transformers import CLIPModel, CLIPProcessor
        name = os.environ.get("ROAD_SAFETY_CLIP", "openai/clip-vit-base-patch32")
        _clip["model"] = CLIPModel.from_pretrained(name)
        _clip["proc"] = CLIPProcessor.from_pretrained(name)
        _clip["model"].eval()
        _clip["ready"] = True
    except Exception:
        _clip["ready"] = False
    return _clip["ready"]


def _helmet_from_clip(crop_img):
    """Zero-shot helmet check on a head crop. None if CLIP is unavailable."""
    if crop_img is None or not _clip_ready():
        return None
    try:
        import torch
        prompts = ["a photo of a person wearing a bike helmet",
                   "a photo of a person with a bare head, no helmet"]
        inputs = _clip["proc"](text=prompts, images=crop_img,
                               return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = _clip["model"](**inputs).logits_per_image
        probs = logits.softmax(dim=1)[0].tolist()
        return "yes" if probs[0] >= probs[1] else "no"
    except Exception:
        return None


# ── Core analysis ─────────────────────────────────────────────────────────────

def _analyse_frame(model, arr, class_names, conf, iou, imgsz, use_clip, track):
    """Detect (and optionally track) road users, pair riders, score helmets."""
    from PIL import Image as PILImage

    kwargs = dict(conf=conf, iou=iou, imgsz=imgsz, verbose=False, device="cpu")
    try:
        import torch
        if torch.cuda.is_available():
            kwargs["device"] = "0"
    except Exception:
        pass

    if track:
        results = model.track(arr, persist=False, tracker="bytetrack.yaml", **kwargs)
    else:
        results = model.predict(arr, **kwargs)
    r = results[0]
    ih, iw = r.orig_shape

    dets = []
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        tid = None
        if getattr(box, "id", None) is not None:
            try:
                tid = int(box.id[0])
            except Exception:
                tid = None
        dets.append({
            "cls": int(box.cls[0]), "conf": float(box.conf[0]),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2, "track_id": tid,
            "name": _name_for(int(box.cls[0]), class_names),
        })

    # keep only road users
    road = [d for d in dets if d["name"] in ROAD_USERS]
    persons = [d for d in road if d["name"] == "person"]
    wheels = [d for d in road if d["name"] in TWO_WHEELERS]

    # Pair riders: a person sitting on a two-wheeler = their boxes overlap a lot
    rider_ids = set()
    for w in wheels:
        wb = [w["x1"], w["y1"], w["x2"], w["y2"]]
        for p in persons:
            pb = [p["x1"], p["y1"], p["x2"], p["y2"]]
            if _overlap_ratio(pb, wb) > 0.45 or _overlap_ratio(wb, pb) > 0.6:
                rider_ids.add(id(p))
                w["has_rider"] = True

    img = PILImage.fromarray(arr) if use_clip else None
    objects = []
    for d in road:
        is_rider = id(d) in rider_ids
        if d["name"] == "person":
            role = "rider" if is_rider else "pedestrian"
        elif d["name"] in TWO_WHEELERS:
            role = "rider"
        else:
            role = "vehicle"

        helmet = None
        if role == "rider":
            helmet = _helmet_from_model([d["x1"], d["y1"], d["x2"], d["y2"]],
                                        dets, class_names)
            if helmet is None and use_clip and img is not None:
                pad = 8
                hx1, hy1, hx2, hy2 = _head_region([d["x1"], d["y1"], d["x2"], d["y2"]])
                crop = img.crop((max(0, int(hx1) - pad), max(0, int(hy1) - pad),
                                 min(iw, int(hx2) + pad), min(ih, int(hy2) + pad)))
                if crop.width > 8 and crop.height > 8:
                    helmet = _helmet_from_clip(crop)
        helmet = helmet or "unknown"

        # indicator colour meaning: ok / warn / unknown / info
        if role == "rider":
            indicator = {"yes": "ok", "no": "warn", "unknown": "unknown"}[helmet]
        else:
            indicator = "info"

        objects.append({
            "track_id": d["track_id"],
            "class_name": d["name"],
            "conf": round(d["conf"], 3),
            "x": round(d["x1"] / iw, 4), "y": round(d["y1"] / ih, 4),
            "w": round((d["x2"] - d["x1"]) / iw, 4),
            "h": round((d["y2"] - d["y1"]) / ih, 4),
            "role": role,
            "helmet": helmet if role == "rider" else None,
            "indicator": indicator,
        })

    return objects, {"w": iw, "h": ih}


def _summarise(objects: list) -> dict:
    riders = [o for o in objects if o["role"] == "rider"]
    helmets = sum(1 for o in riders if o["helmet"] == "yes")
    no_helmet = sum(1 for o in riders if o["helmet"] == "no")
    unknown = sum(1 for o in riders if o["helmet"] == "unknown")
    known = helmets + no_helmet

    # Safety score: start at 100, lose points for unprotected riders. Only
    # considers riders whose helmet status we actually know.
    if known > 0:
        score = round(100 * helmets / known)
    else:
        score = None

    alerts = []
    if no_helmet:
        alerts.append(f"{no_helmet} rider(s) without a helmet")
    if unknown and known == 0:
        alerts.append("Train a helmet model (or enable zero-shot) to score riders")

    counts = {}
    for o in objects:
        counts[o["class_name"]] = counts.get(o["class_name"], 0) + 1

    return {
        "total": len(objects),
        "riders": len(riders),
        "helmets": helmets,
        "no_helmet": no_helmet,
        "unknown": unknown,
        "helmet_rate": round(helmets / known, 3) if known else None,
        "safety_score": score,
        "counts": counts,
        "alerts": alerts,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/projects/{project_id}/road-safety/status")
def road_safety_status(project_id: int, run_id: int | None = None,
                       session: Session = Depends(get_session)):
    """Report which helmet source would be used, so the UI can explain itself."""
    model, names, note = _load_model_and_classes(project_id, run_id, session)
    model_helmet = any(_HELMET_YES.search(n) or _HELMET_NO.search(n) for n in names)
    return {
        "model_note": note,
        "classes": names[:80],
        "helmet_from_model": model_helmet,
        "clip_available": _clip_importable(),
        "clip_loaded": _clip["ready"] is True,
        "road_classes": ROAD_USERS,
    }


@router.post("/projects/{project_id}/road-safety/analyze")
async def road_safety_analyze(
    project_id: int,
    file: UploadFile = File(...),
    run_id: int | None = Form(None),
    conf: float = Form(0.3),
    iou: float = Form(0.45),
    imgsz: int = Form(640),
    use_clip: bool = Form(False),
    track: bool = Form(True),
    session: Session = Depends(get_session),
):
    """Analyse a single road image and return tracked road users + indicators."""
    from routers.infer import _get_model  # noqa: F401  (kept for symmetry)

    model, class_names, note = _load_model_and_classes(project_id, run_id, session)

    suffix = Path(file.filename or "img.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    t0 = time.time()
    try:
        import numpy as np
        from PIL import Image as PILImage
        arr = np.array(PILImage.open(tmp_path).convert("RGB"))
        objects, dims = _analyse_frame(model, arr, class_names, conf, iou,
                                       imgsz, use_clip, track)
        summary = _summarise(objects)
        return {
            "objects": objects,
            "summary": summary,
            "image_w": dims["w"], "image_h": dims["h"],
            "meta": {
                "model": note,
                "tracked": bool(track),
                "clip": bool(use_clip and _clip_ready()),
                "time_ms": int((time.time() - t0) * 1000),
            },
        }
    finally:
        os.unlink(tmp_path)


# ── Video ─────────────────────────────────────────────────────────────────────

def _process_road_video(job_id, in_path, model, class_names, conf, iou, imgsz,
                        use_clip, tracker_on, class_filter):
    """Process a video, drawing road-safety indicators on each frame."""
    job = _video_jobs[job_id]
    try:
        import cv2
        import numpy as np
        from PIL import Image as PILImage

        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open video")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        job["total_frames"] = total

        out_path = os.path.join(VIDEO_OUT_DIR, f"road_{job_id}.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (w, h))

        counts = {}          # class -> max simultaneous
        seen_tracks = set()
        frame_alerts = 0
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            objects, _ = _analyse_frame(model, rgb, class_names, conf, iou,
                                        imgsz, use_clip, tracker_on)
            s = _summarise(objects)
            if s["no_helmet"]:
                frame_alerts += 1
            for k, v in s["counts"].items():
                counts[k] = max(counts.get(k, 0), v)
            for o in objects:
                if o["track_id"] is not None:
                    seen_tracks.add((o["class_name"], o["track_id"]))

            # draw indicators
            for o in objects:
                x1 = int(o["x"] * w); y1 = int(o["y"] * h)
                x2 = int((o["x"] + o["w"]) * w); y2 = int((o["y"] + o["h"]) * h)
                colour = {"ok": (80, 220, 100), "warn": (60, 60, 240),
                          "unknown": (60, 190, 240), "info": (240, 170, 60)}[o["indicator"]]
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                label = o["class_name"]
                if o["track_id"] is not None:
                    label += f" #{o['track_id']}"
                if o["role"] == "rider":
                    label += f" | {o['helmet']}"
                cv2.putText(frame, label, (x1, max(12, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)

            banner = f"road users {s['total']}  riders {s['riders']}  no-helmet {s['no_helmet']}"
            if s["safety_score"] is not None:
                banner += f"  score {s['safety_score']}"
            cv2.rectangle(frame, (0, 0), (w, 26), (0, 0, 0), -1)
            cv2.putText(frame, banner, (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1, cv2.LINE_AA)

            writer.write(frame)
            i += 1
            job["processed"] = i
            if i % 5 == 0:
                job["stage"] = "processing"

        cap.release()
        writer.release()
        job["out_path"] = out_path
        job["status"] = "done"
        job["summary"] = {
            "frames": i,
            "unique_tracks": len(seen_tracks),
            "max_counts": counts,
            "frames_with_unprotected_rider": frame_alerts,
        }
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)
    finally:
        if os.path.exists(in_path):
            try:
                os.unlink(in_path)
            except OSError:
                pass


@router.post("/projects/{project_id}/road-safety/analyze-video")
async def road_safety_analyze_video(
    project_id: int,
    file: UploadFile = File(...),
    run_id: int | None = Form(None),
    conf: float = Form(0.3),
    iou: float = Form(0.45),
    imgsz: int = Form(640),
    use_clip: bool = Form(False),
    track: bool = Form(True),
    session: Session = Depends(get_session),
):
    """Analyse a road video: tracking + helmet indicators, returns a job id."""
    try:
        import cv2  # noqa: F401
    except ImportError:
        raise HTTPException(500, "OpenCV (cv2) is not installed on this server.")

    model, class_names, note = _load_model_and_classes(project_id, run_id, session)

    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    in_path = os.path.join(VIDEO_OUT_DIR, f"road_in_{uuid.uuid4().hex}{suffix}")
    with open(in_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    job_id = uuid.uuid4().hex
    _video_jobs[job_id] = {"status": "running", "stage": "processing", "processed": 0,
                           "total_frames": 0, "out_path": None, "error": None,
                           "model": note}
    threading.Thread(
        target=_process_road_video,
        args=(job_id, in_path, model, class_names, conf, iou, imgsz, use_clip, track, None),
        daemon=True,
    ).start()
    return {"job_id": job_id, "model": note}


@router.get("/projects/{project_id}/road-safety/video/{job_id}/status")
def road_video_status(project_id: int, job_id: str):
    job = _video_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "status": job["status"],
        "stage": job.get("stage", "processing"),
        "processed": job["processed"],
        "total_frames": job["total_frames"],
        "error": job.get("error"),
        "summary": job.get("summary"),
    }


@router.get("/projects/{project_id}/road-safety/video/{job_id}/download")
def road_video_download(project_id: int, job_id: str):
    job = _video_jobs.get(job_id)
    if not job or not job.get("out_path") or not os.path.exists(job["out_path"]):
        raise HTTPException(404, "Result not ready")
    return FileResponse(job["out_path"], media_type="video/mp4",
                        filename=f"road_safety_{job_id}.mp4")
