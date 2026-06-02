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

# Max depth each metric model can represent (sigmoid * max_depth clamp).
# Depth values near these are saturated/unreliable for measurement.
METRIC_MAX_DEPTH = {
    "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf":  20.0,
    "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf": 80.0,
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


def _build_response(orig_pil, depth_np, colormap: str, model_id: str,
                    exif_focal35: float = 0.0):
    """Produce the complete JSON response for single-image endpoints."""
    from PIL import Image as PILImage
    import numpy as np

    # Resize depth to original image dimensions.
    # For METRIC models depth_np is absolute metres — depth_rs preserves that.
    w, h = orig_pil.size
    depth_rs = np.array(
        PILImage.fromarray(depth_np).resize((w, h), PILImage.BILINEAR)
    ).astype("float32")

    depth_u8, uniform, d_min, d_max = _normalize_depth(depth_rs)
    colorized = _colorize(depth_u8, colormap)
    sbs       = _sidebyside(orig_pil, colorized)
    is_metric = model_id in METRIC_MODELS

    # Normalise depth to [0,1] float32 for point cloud generation
    span = max(float(d_max - d_min), 1e-6)
    depth_norm = ((depth_rs - d_min) / span).astype("float32")

    # Store result for point cloud + measure endpoints (keep for 1 hour)
    result_id = uuid.uuid4().hex
    orig_rgb  = np.array(orig_pil.convert("RGB"), dtype=np.uint8)
    # Purge results older than 1 hour to avoid memory creep
    now = time.time()
    expired = [k for k, v in _depth_results.items() if now - v["ts"] > 3600]
    for k in expired:
        del _depth_results[k]
    _depth_results[result_id] = {
        "depth_norm":   depth_norm,   # float32 [0..1] — for point cloud
        # Raw depth in METRES (metric models only) — required for measurement.
        # None for relative models (no real-world scale available).
        "depth_meters": depth_rs.copy() if is_metric else None,
        "orig_rgb":     orig_rgb,     # uint8 RGB
        "is_metric":    is_metric,
        "model_id":     model_id,
        "max_depth":    METRIC_MAX_DEPTH.get(model_id, 0.0),
        "exif_focal35": float(exif_focal35),
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
        "result_id":  result_id,   # use this to fetch point cloud / measure
    }


def _read_exif_focal35(pil_img) -> float:
    """
    Return the 35mm-equivalent focal length from EXIF, or 0.0 if absent.
    This lets us compute true fx = (f35 / 36) * image_width without needing
    the sensor size.  Most social-media images have EXIF stripped → returns 0.
    """
    try:
        exif = pil_img.getexif()
        if not exif:
            return 0.0
        # 0xA405 = FocalLengthIn35mmFilm
        f35 = exif.get(0xA405)
        if f35:
            return float(f35)
    except Exception:
        pass
    return 0.0


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
        img_full = PILImage.open(io.BytesIO(raw))
        focal35  = _read_exif_focal35(img_full)   # read EXIF before convert/resize
        img      = img_full.convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"Cannot read image: {exc}")
    img     = _cap_image(img, 1024)
    pipe    = _get_depth_pipe(model_id)
    depth   = _run_depth(pipe, img)
    return _build_response(img, depth, colormap, model_id, exif_focal35=focal35)


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

    # Sub-sampled pixel grid — use ALL subsampled pixels (no depth mask so
    # count is never 0 regardless of the depth range of the image)
    ys = np.arange(0, h, subsample)
    xs = np.arange(0, w, subsample)
    uu, vv = np.meshgrid(xs, ys)            # both (len_ys, len_xs)

    d = depth_norm[ys[:, None], xs[None, :]]   # (len_ys, len_xs)

    uu_f = uu.flatten().astype("float32")
    vv_f = vv.flatten().astype("float32")
    d_f  = d.flatten().astype("float32")

    X =  (uu_f - cx) * d_f / fx
    Y = -(vv_f - cy) * d_f / fy   # flip Y: image row 0 = top → 3D +Y
    Z = -d_f                       # flip Z: largest depth → most negative Z

    positions = np.stack([X, Y, Z], axis=1)          # [N, 3]

    rgb    = orig_rgb[ys[:, None], xs[None, :]].reshape(-1, 3)
    colors = rgb.astype("float32") / 255.0            # [N, 3]

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
    Return point cloud as JSON  { n, pos, col }
      n   — number of points
      pos — flat list of float32  [x0,y0,z0, x1,y1,z1, ...]
      col — flat list of float32  [r0,g0,b0, ...] values in 0–1
    The frontend converts these directly to Float32Array — no binary parsing.
    """
    import numpy as np

    rec = _depth_results.get(result_id)
    if not rec:
        raise HTTPException(404, "Result not found — re-run depth estimation first.")

    positions, colors = _backproject(rec["depth_norm"], rec["orig_rgb"],
                                     subsample=subsample, fx=fx)
    n = len(positions)

    # Round floats to 5 decimal places to keep JSON compact
    pos_list = [round(float(v), 5) for v in positions.flatten()]
    col_list = [round(float(v), 4) for v in colors.flatten()]

    return {"n": n, "pos": pos_list, "col": col_list}


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


# ─── Depth-of-field / Portrait blur ───────────────────────────────────────────

@router.get("/depth/portrait/{result_id}")
def depth_portrait(result_id: str,
                   focus:    float = 0.6,    # focus plane (0=far, 1=near)
                   strength: int   = 18,     # max blur kernel radius
                   aperture: float = 0.18):  # focus band half-width (depth-of-field)
    """
    Depth-of-field "portrait mode" blur, computed from the stored depth map.

    Pixels whose depth is near the `focus` plane stay sharp; pixels farther
    from it (in depth) get progressively blurred — a realistic bokeh effect.
    Reuses the depth result, so no re-inference is needed (instant slider drag).

    Returns: { image_b64 } — colorized blurred result as base64 PNG.
    """
    import cv2, numpy as np

    rec = _depth_results.get(result_id)
    if not rec:
        raise HTTPException(404, "Result not found — re-run depth estimation first.")

    depth = rec["depth_norm"]                       # [H,W] float32, 1=near
    rgb   = rec["orig_rgb"]                          # [H,W,3] uint8 RGB
    bgr   = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w  = depth.shape

    # Blur amount per pixel = distance from the focus plane in depth space,
    # outside an in-focus band of half-width `aperture`.
    dist     = np.abs(depth - float(focus))
    blur_amt = np.clip((dist - float(aperture)) / max(1e-6, 1.0 - aperture), 0.0, 1.0)

    strength = max(1, min(int(strength), 60))

    # Build a small pyramid of increasingly blurred versions, then blend each
    # pixel from the level matching its blur amount.  Far cheaper than a
    # per-pixel variable kernel and visually equivalent for a preview.
    levels = 6
    blurred_stack = [bgr.astype(np.float32)]
    cur = bgr.copy()
    for i in range(1, levels):
        k = max(1, int(strength * i / (levels - 1)))
        k = k * 2 + 1                                # odd kernel
        cur = cv2.GaussianBlur(bgr, (k, k), 0)
        blurred_stack.append(cur.astype(np.float32))

    # For each pixel pick the two nearest pyramid levels and lerp between them.
    lvl_f = blur_amt * (levels - 1)
    lo    = np.floor(lvl_f).astype(np.int32)
    hi    = np.clip(lo + 1, 0, levels - 1)
    frac  = (lvl_f - lo)[..., None]                  # [H,W,1]

    stack = np.stack(blurred_stack, axis=0)          # [levels,H,W,3]
    lo_img = np.take_along_axis(stack, lo[None, ..., None].repeat(3, -1), axis=0)[0]
    hi_img = np.take_along_axis(stack, hi[None, ..., None].repeat(3, -1), axis=0)[0]
    out    = (lo_img * (1 - frac) + hi_img * frac).astype(np.uint8)

    return {"image_b64": _to_b64(out)}


# ─── Click-to-measure (metric models only) ───────────────────────────────────

def _sample_depth_median(depth_m, px: int, py: int, r: int = 2):
    """5x5 median depth around (px,py) — robust to edge/outlier pixels.
    Returns (median_depth, relative_std) where relative_std flags edge noise."""
    import numpy as np
    h, w = depth_m.shape
    x0, x1 = max(0, px - r), min(w, px + r + 1)
    y0, y1 = max(0, py - r), min(h, py + r + 1)
    patch = depth_m[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0, 0.0
    med = float(np.median(patch))
    rel_std = float(patch.std() / med) if med > 1e-6 else 0.0
    return med, rel_std


@router.get("/depth/measure/{result_id}")
def depth_measure(result_id: str,
                  x1: float, y1: float, x2: float, y2: float,  # normalised 0..1
                  hfov_deg: float = 65.0,    # assumed horizontal FOV if no EXIF
                  scale:    float = 1.0):    # user calibration multiplier
    """
    Measure the real-world distance (metres) between two clicked points,
    using the stored RAW METRIC depth map.  Metric models only.

    Math (pinhole back-projection):
        fx = fy = (W/2) / tan(HFOV/2)   [or EXIF: (f35/36)*W]
        cx, cy = W/2, H/2
        X = (u-cx)*Z/fx,  Y = (v-cy)*Z/fy,  Z = depth_metres
        distance = ||P2 - P1|| * scale

    Depth is sampled as a 5x5 median at each point (robust to edges).
    """
    import numpy as np, math

    rec = _depth_results.get(result_id)
    if not rec:
        raise HTTPException(404, "Result not found — re-run depth estimation first.")
    if not rec.get("is_metric") or rec.get("depth_meters") is None:
        raise HTTPException(400,
            "Distance measurement requires a metric depth model "
            "(Metric Indoor / Metric Outdoor). The selected model only "
            "produces relative depth, which has no real-world scale.")

    depth_m = rec["depth_meters"]            # [H,W] float32 in metres
    h, w    = depth_m.shape

    # ── Resolve focal length (px) ────────────────────────────────────────────
    f35 = rec.get("exif_focal35", 0.0)
    if f35 and f35 > 0:
        fx = (f35 / 36.0) * w               # full-frame width = 36 mm
        fx_source = f"EXIF ({f35:.0f}mm equiv.)"
    else:
        hfov = max(20.0, min(float(hfov_deg), 120.0))
        fx = (w / 2.0) / math.tan(math.radians(hfov) / 2.0)
        fx_source = f"assumed {hfov:.0f}° FOV"
    fy = fx
    cx, cy = w / 2.0, h / 2.0

    # ── Pixel coordinates (clamp normalised → pixel) ────────────────────────
    px1 = int(round(min(max(x1, 0.0), 1.0) * (w - 1)))
    py1 = int(round(min(max(y1, 0.0), 1.0) * (h - 1)))
    px2 = int(round(min(max(x2, 0.0), 1.0) * (w - 1)))
    py2 = int(round(min(max(y2, 0.0), 1.0) * (h - 1)))

    z1, rel1 = _sample_depth_median(depth_m, px1, py1)
    z2, rel2 = _sample_depth_median(depth_m, px2, py2)

    # ── Back-project to camera space (no Y/Z flip — Euclidean is sign-safe) ──
    def to_cam(px, py, z):
        return ((px - cx) * z / fx, (py - cy) * z / fy, z)
    X1, Y1, Z1 = to_cam(px1, py1, z1)
    X2, Y2, Z2 = to_cam(px2, py2, z2)

    dist = math.sqrt((X2 - X1) ** 2 + (Y2 - Y1) ** 2 + (Z2 - Z1) ** 2)
    dist *= max(0.01, float(scale))

    # ── Warnings ─────────────────────────────────────────────────────────────
    warnings = []
    max_depth = rec.get("max_depth", 0.0)
    if max_depth > 0:
        if z1 >= 0.95 * max_depth or z2 >= 0.95 * max_depth:
            warnings.append(f"A point is near the model's {max_depth:.0f} m limit — depth unreliable there.")
    if rel1 > 0.15 or rel2 > 0.15:
        warnings.append("A point sits on a depth edge — click on a flat surface for accuracy.")

    return {
        "distance_m":  round(dist, 3),
        "point1":      {"depth_m": round(z1, 3), "x": px1, "y": py1},
        "point2":      {"depth_m": round(z2, 3), "x": px2, "y": py2},
        "fx_px":       round(fx, 1),
        "fx_source":   fx_source,
        "warnings":    warnings,
        "disclaimer":  ("Estimate only — not for precision, safety, medical, "
                        "or construction use. Front-to-back (depth) distances are "
                        "more reliable than side-to-side. Calibrate with a known "
                        "distance for best accuracy."),
    }
