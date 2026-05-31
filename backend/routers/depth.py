from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, StreamingResponse
import os, io, uuid, shutil, threading, base64, time
from pathlib import Path

router = APIRouter(tags=["depth"])

VIDEO_OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "video_out")
os.makedirs(VIDEO_OUT_DIR, exist_ok=True)

_depth_cache:   dict = {}   # model_id  → HF pipeline
_depth_jobs:    dict = {}   # job_id    → status dict
_depth_results: dict = {}   # result_id → {depth_norm, orig_rgb, w, h, ts}

# Models that produce metric depth (values in metres)
METRIC_MODELS = {
    "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
}

CV2_COLORMAPS = {
    "inferno": 8,   # cv2.COLORMAP_INFERNO
    "magma":   16,  # cv2.COLORMAP_MAGMA
    "plasma":  15,  # cv2.COLORMAP_PLASMA
    "viridis": 16,  # cv2.COLORMAP_VIRIDIS  (same int on most builds; fine)
    "turbo":   20,  # cv2.COLORMAP_TURBO
    "jet":     2,   # cv2.COLORMAP_JET
    "hot":     11,  # cv2.COLORMAP_HOT
}


# ─── Model cache ──────────────────────────────────────────────────────────────
def _get_depth_pipe(model_id: str):
    """Lazy-load and cache a HuggingFace depth-estimation pipeline."""
    if model_id in _depth_cache:
        return _depth_cache[model_id]
    try:
        import torch
        from transformers import pipeline as hf_pipeline
        device = 0 if torch.cuda.is_available() else -1
        pipe = hf_pipeline(
            task="depth-estimation",
            model=model_id,
            device=device,
        )
        _depth_cache[model_id] = pipe
        return pipe
    except ImportError:
        raise HTTPException(500,
            "transformers is not installed. Run: pip install transformers>=4.41.0 timm")
    except Exception as exc:
        raise HTTPException(500,
            f"Failed to load depth model '{model_id}': {exc}")


# ─── Core helpers ─────────────────────────────────────────────────────────────
def _run_depth(pipe, image_pil):
    """Run depth estimation pipeline, return float32 numpy depth array."""
    import numpy as np
    result = pipe(image_pil)
    return result["predicted_depth"].squeeze().cpu().numpy().astype("float32")


def _cap_image(img, max_size: int = 1024):
    """Resize image so its longest side ≤ max_size (preserves aspect ratio)."""
    from PIL import Image as PILImage
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
    return img


def _normalize_depth(depth_np, g_min=None, g_max=None):
    """
    Normalise float32 depth → uint8 [0, 255].
    All supported models output *inverse* depth (closer = larger value),
    so after normalisation bright = close with NO inversion needed.

    g_min / g_max: optional global clip range for temporal video consistency.
    Returns (uint8_array, uniform_warning, raw_min, raw_max).
    """
    import numpy as np
    d_min = float(depth_np.min()) if g_min is None else g_min
    d_max = float(depth_np.max()) if g_max is None else g_max
    span  = d_max - d_min
    if span < 1e-6:                       # uniform image guard
        return np.full_like(depth_np, 128, dtype=np.uint8), True, d_min, d_max
    normalized = (depth_np - d_min) / span
    normalized = normalized.clip(0.0, 1.0)
    return (normalized * 255).astype(np.uint8), False, d_min, d_max


def _colorize(depth_uint8, colormap: str):
    """Apply OpenCV colormap to uint8 depth, return BGR ndarray."""
    import cv2
    cv_cmap = CV2_COLORMAPS.get(colormap, 8)   # default = inferno
    return cv2.applyColorMap(depth_uint8, cv_cmap)


def _to_b64(bgr_or_gray) -> str:
    """Encode a cv2 BGR or grayscale ndarray as base64 PNG string."""
    import cv2
    _, buf = cv2.imencode(".png", bgr_or_gray)
    return base64.b64encode(buf.tobytes()).decode()


def _sidebyside(orig_pil, colorized_bgr):
    """Horizontally concatenate original photo + depth map."""
    import cv2, numpy as np
    from PIL import Image as PILImage
    orig_bgr = cv2.cvtColor(np.array(orig_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w     = orig_bgr.shape[:2]
    depth_rs = cv2.resize(colorized_bgr, (w, h))
    return np.hstack([orig_bgr, depth_rs])


def _build_response(orig_pil, depth_np, colormap: str, model_id: str):
    """Produce the complete JSON response for single-image endpoints."""
    from PIL import Image as PILImage
    import numpy as np

    # Resize depth to original image dimensions
    w, h = orig_pil.size
    depth_rs = np.array(
        PILImage.fromarray(depth_np).resize((w, h), PILImage.BILINEAR)
    )

    depth_u8, uniform, d_min, d_max = _normalize_depth(depth_rs)
    colorized = _colorize(depth_u8, colormap)
    sbs       = _sidebyside(orig_pil, colorized)
    is_metric = model_id in METRIC_MODELS

    # Normalise depth to [0,1] float32 for point cloud generation
    span = max(float(d_max - d_min), 1e-6)
    depth_norm = ((depth_rs - d_min) / span).astype("float32")

    # Store result for point cloud endpoint (keep for 1 hour)
    result_id = uuid.uuid4().hex
    orig_rgb  = np.array(orig_pil.convert("RGB"), dtype=np.uint8)
    # Purge results older than 1 hour to avoid memory creep
    now = time.time()
    expired = [k for k, v in _depth_results.items() if now - v["ts"] > 3600]
    for k in expired:
        del _depth_results[k]
    _depth_results[result_id] = {
        "depth_norm": depth_norm,   # float32 [0..1], bright = close
        "orig_rgb":   orig_rgb,     # uint8 RGB
        "w": w, "h": h, "ts": now,
    }

    return {
        "depth_colorized_b64": _to_b64(colorized),
        "depth_gray_b64":      _to_b64(depth_u8),
        "sidebyside_b64":      _to_b64(sbs),
        "depth_stats": {
            "min_raw":         round(float(d_min), 4),
            "max_raw":         round(float(d_max), 4),
            "mean_raw":        round(float(depth_rs.mean()), 4),
            "is_metric":       is_metric,
            "unit":            "meters" if is_metric else "relative",
            "uniform_warning": uniform,
        },
        "model_used": model_id,
        "colormap":   colormap,
        "image_w":    w,
        "image_h":    h,
        "result_id":  result_id,   # use this to fetch point cloud
    }


# ─── Image inference (file upload) ───────────────────────────────────────────
@router.post("/depth/infer")
async def depth_infer(
    file:     UploadFile = File(...),
    model_id: str  = Form("depth-anything/Depth-Anything-V2-Small-hf"),
    colormap: str  = Form("inferno"),
):
    from PIL import Image as PILImage
    raw = await file.read()
    try:
        img = PILImage.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"Cannot read image: {exc}")
    img     = _cap_image(img, 1024)
    pipe    = _get_depth_pipe(model_id)
    depth   = _run_depth(pipe, img)
    return _build_response(img, depth, colormap, model_id)


# ─── Image inference (URL) ────────────────────────────────────────────────────
@router.post("/depth/infer-url")
async def depth_infer_url(
    url:      str  = Form(...),
    model_id: str  = Form("depth-anything/Depth-Anything-V2-Small-hf"),
    colormap: str  = Form("inferno"),
):
    import urllib.request
    from PIL import Image as PILImage
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
    except Exception as exc:
        raise HTTPException(400, f"Failed to fetch image: {exc}")
    try:
        img = PILImage.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"URL did not return a valid image: {exc}")
    img   = _cap_image(img, 1024)
    pipe  = _get_depth_pipe(model_id)
    depth = _run_depth(pipe, img)
    return _build_response(img, depth, colormap, model_id)


# ─── Webcam frame ─────────────────────────────────────────────────────────────
@router.post("/depth/webcam-frame")
async def depth_webcam_frame(
    frame:    UploadFile = File(...),
    model_id: str  = Form("depth-anything/Depth-Anything-V2-Small-hf"),
    colormap: str  = Form("inferno"),
):
    import numpy as np
    from PIL import Image as PILImage
    raw = await frame.read()
    try:
        img = PILImage.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"Bad frame: {exc}")
    img  = _cap_image(img, 640)    # keep webcam fast
    pipe = _get_depth_pipe(model_id)
    depth = _run_depth(pipe, img)

    w, h = img.size
    depth_rs = np.array(PILImage.fromarray(depth).resize((w, h), PILImage.BILINEAR))
    depth_u8, uniform, _, _ = _normalize_depth(depth_rs)
    colorized = _colorize(depth_u8, colormap)
    return {
        "depth_b64":       _to_b64(colorized),
        "uniform_warning": uniform,
        "is_metric":       model_id in METRIC_MODELS,
    }


# ─── Video processing ─────────────────────────────────────────────────────────
def _process_depth_video(job_id: str, video_path: str, model_id: str, colormap: str):
    """
    Two-pass video depth estimation:
      Pass 1 — run inference on every frame, collect depth arrays + global min/max.
      Pass 2 — normalise all frames with the global range → temporally stable colours.
    """
    try:
        import cv2, numpy as np
        from PIL import Image as PILImage

        pipe = _get_depth_pipe(model_id)
        cap  = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            _depth_jobs[job_id].update({"status": "failed", "error": "Cannot open video"})
            return

        fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        _depth_jobs[job_id]["total_frames"] = total

        # Scale down for inference to keep memory/speed manageable
        proc_w = min(width, 640)
        proc_h = int(height * (proc_w / width))

        # ── Pass 1: collect all depth arrays and global range ─────────────────
        _depth_jobs[job_id]["stage"] = "analyzing"
        all_depths: list = []
        g_min, g_max = float("inf"), float("-inf")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            small = cv2.resize(frame, (proc_w, proc_h))
            pil   = PILImage.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            d     = _run_depth(pipe, pil)
            all_depths.append(d)
            g_min = min(g_min, float(d.min()))
            g_max = max(g_max, float(d.max()))
            _depth_jobs[job_id]["processed"] = len(all_depths)
        cap.release()

        if not all_depths:
            _depth_jobs[job_id].update({"status": "failed", "error": "No frames decoded"})
            return

        # ── Pass 2: render each frame with global normalisation ───────────────
        _depth_jobs[job_id]["stage"] = "rendering"
        out_path = os.path.join(VIDEO_OUT_DIR, f"depth_{job_id}.mp4")
        writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                   fps, (width, height))
        span     = g_max - g_min if (g_max - g_min) > 1e-6 else 1.0
        cv_cmap  = CV2_COLORMAPS.get(colormap, 8)

        for i, d in enumerate(all_depths):
            d_u8      = ((d - g_min) / span * 255).clip(0, 255).astype(np.uint8)
            colorized = cv2.applyColorMap(d_u8, cv_cmap)
            writer.write(cv2.resize(colorized, (width, height)))
            _depth_jobs[job_id]["processed"] = total + i   # second pass progress

        writer.release()
        del all_depths   # free memory

        _depth_jobs[job_id].update({"status": "done", "out_path": out_path})
    except Exception as exc:
        _depth_jobs[job_id].update({"status": "failed", "error": str(exc)})
    finally:
        try:
            os.unlink(video_path)
        except Exception:
            pass


def _run_depth_video_url(job_id: str, url: str, model_id: str, colormap: str):
    from routers.infer import _download_from_url
    _depth_jobs[job_id]["stage"] = "downloading"
    dl_dir = None
    try:
        video_path, dl_dir = _download_from_url(url)
    except Exception as exc:
        _depth_jobs[job_id].update({"status": "failed", "error": str(exc)})
        return
    _depth_jobs[job_id]["stage"] = "analyzing"
    _process_depth_video(job_id, video_path, model_id, colormap)
    if dl_dir and os.path.exists(dl_dir):
        shutil.rmtree(dl_dir, ignore_errors=True)


@router.post("/depth/video-infer")
async def start_depth_video(
    file:     UploadFile = File(...),
    model_id: str  = Form("depth-anything/Depth-Anything-V2-Small-hf"),
    colormap: str  = Form("inferno"),
):
    try:
        import cv2  # noqa
    except ImportError:
        raise HTTPException(500, "OpenCV (cv2) is not installed")

    suffix   = Path(file.filename or "video.mp4").suffix or ".mp4"
    tmp_path = os.path.join(VIDEO_OUT_DIR, f"depth_in_{uuid.uuid4().hex}{suffix}")
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    job_id = uuid.uuid4().hex
    _depth_jobs[job_id] = {
        "status": "running", "stage": "analyzing",
        "processed": 0, "total_frames": 0, "out_path": None, "error": None,
    }
    threading.Thread(
        target=_process_depth_video,
        args=(job_id, tmp_path, model_id, colormap),
        daemon=True,
    ).start()
    return {"job_id": job_id}


@router.post("/depth/video-infer-url")
async def start_depth_video_url(
    url:      str  = Form(...),
    model_id: str  = Form("depth-anything/Depth-Anything-V2-Small-hf"),
    colormap: str  = Form("inferno"),
):
    try:
        import cv2  # noqa
    except ImportError:
        raise HTTPException(500, "OpenCV (cv2) is not installed")

    job_id = uuid.uuid4().hex
    _depth_jobs[job_id] = {
        "status": "running", "stage": "downloading",
        "processed": 0, "total_frames": 0, "out_path": None, "error": None,
    }
    threading.Thread(
        target=_run_depth_video_url,
        args=(job_id, url, model_id, colormap),
        daemon=True,
    ).start()
    return {"job_id": job_id}


@router.get("/depth/video-infer/{job_id}/status")
def depth_video_status(job_id: str):
    job = _depth_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "status":       job["status"],
        "stage":        job.get("stage", "analyzing"),
        "processed":    job["processed"],
        "total_frames": job["total_frames"],
        "error":        job.get("error"),
    }


@router.get("/depth/video-infer/{job_id}/download")
def depth_video_download(job_id: str):
    job = _depth_jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "Result not ready")
    out_path = job["out_path"]
    if not out_path or not os.path.exists(out_path):
        raise HTTPException(404, "Output file not found")
    return FileResponse(out_path, filename=f"depth_{job_id[:8]}.mp4",
                        media_type="video/mp4")


# ─── 3D Point Cloud ───────────────────────────────────────────────────────────

def _backproject(depth_norm, orig_rgb, subsample: int = 4, fx: float = 0.0):
    """
    Back-project a normalised depth map [0,1] to XYZ point cloud.

    Returns:
        positions  – float32 ndarray [N, 3]  (X, Y, Z in camera space)
        colors     – float32 ndarray [N, 3]  (R, G, B normalised 0–1)

    Coordinate convention (matches Three.js defaults):
        +X right, +Y up, +Z toward camera (right-handed)
    """
    import numpy as np

    h, w = depth_norm.shape
    # Estimate focal length from image size if not provided
    if fx <= 0:
        fx = max(w, h) * 0.85
    fy = fx
    cx, cy = w / 2.0, h / 2.0

    # Sub-sampled pixel grid
    ys = np.arange(0, h, subsample)
    xs = np.arange(0, w, subsample)
    uu, vv = np.meshgrid(xs, ys)

    d = depth_norm[ys[:, None], xs[None, :]]
    mask = d > 0.02   # ignore near-zero depth (sky, background clipped)

    uu_m = uu[mask].astype("float32")
    vv_m = vv[mask].astype("float32")
    d_m  = d[mask].astype("float32")

    X =  (uu_m - cx) * d_m / fx
    Y = -(vv_m - cy) * d_m / fy   # flip Y: image row 0 = top, 3D Y = up
    Z = -d_m                       # flip Z: depth=1 is furthest in -Z

    positions = np.stack([X, Y, Z], axis=1)   # [N, 3]

    rgb = orig_rgb[ys[:, None], xs[None, :]][mask]
    colors = rgb.astype("float32") / 255.0    # [N, 3]

    return positions, colors


def _make_ply_binary(positions, colors_f32):
    """
    Write a binary PLY file containing XYZ + RGB point cloud.
    Returns bytes.
    """
    import numpy as np
    n = len(positions)
    rgb_u8 = (colors_f32 * 255).clip(0, 255).astype(np.uint8)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode()

    # Pack each vertex: 3 × float32 (12 bytes) + 3 × uint8 (3 bytes) = 15 bytes
    data = bytearray()
    for i in range(n):
        data += struct.pack("<fff", positions[i, 0], positions[i, 1], positions[i, 2])
        data += struct.pack("BBB", rgb_u8[i, 0], rgb_u8[i, 1], rgb_u8[i, 2])

    return header + bytes(data)


@router.get("/depth/pointcloud/{result_id}/data")
def depth_pointcloud_data(result_id: str, subsample: int = 4, fx: float = 0.0):
    """
    Return point cloud as raw binary (application/octet-stream).

    Binary layout (little-endian):
        4 bytes  — int32   : N (number of points)
        N*12 bytes — float32[N*3] : positions (x,y,z per point)
        N*12 bytes — float32[N*3] : colors    (r,g,b per point, 0–1)

    The frontend reads this with DataView / Float32Array — no base64 needed.
    """
    import numpy as np
    import struct
    from fastapi.responses import Response

    rec = _depth_results.get(result_id)
    if not rec:
        raise HTTPException(404, "Result not found — re-run depth estimation first.")

    positions, colors = _backproject(rec["depth_norm"], rec["orig_rgb"],
                                     subsample=subsample, fx=fx)
    n = len(positions)

    # Pack as one flat float32 array:
    #   [0]        = N (point count, stored as float32 — exact up to 2^24)
    #   [1 .. N*3] = XYZ positions (stride 3 per point)
    #   [N*3+1 ..] = RGB colors    (stride 3 per point, 0–1)
    #
    # Using a single dtype avoids int32/endianness issues on the JS side —
    # the browser reads it as one Float32Array and slices by index.
    import numpy as np
    count_arr = np.array([float(n)], dtype=np.float32)
    flat = np.concatenate([count_arr,
                           positions.flatten().astype(np.float32),
                           colors.flatten().astype(np.float32)])
    buf = flat.tobytes()

    return Response(content=buf, media_type="application/octet-stream")


@router.get("/depth/pointcloud/{result_id}/download")
def depth_pointcloud_download(result_id: str, subsample: int = 4, fx: float = 0.0):
    """Download a binary PLY file of the point cloud."""
    rec = _depth_results.get(result_id)
    if not rec:
        raise HTTPException(404, "Result not found — re-run depth estimation first.")

    positions, colors = _backproject(rec["depth_norm"], rec["orig_rgb"],
                                     subsample=subsample, fx=fx)
    ply_bytes = _make_ply_binary(positions, colors)
    return StreamingResponse(
        io.BytesIO(ply_bytes),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="pointcloud_{result_id[:8]}.ply"'},
    )
