"""
scan3d.py: "Scan the world". Turn a phone video (or a set of photos) into a
real-world Gaussian-splat scene, then click any object to cut it out in 3D.

Pipeline (no CUDA compiling, no per-scene training):
  1. Frames:    sample the sharpest frames from the video (or take the photos).
  2. Geometry:  MapAnything (facebook/map-anything-apache, Apache-2.0) predicts,
                in one feed-forward pass, metric 3D points for every pixel plus
                camera poses and intrinsics for every frame.
  3. Splats:    every kept pixel becomes one small Gaussian splat (position,
                colour, size from depth / focal length), written as a standard
                3DGS .ply that the Spark viewer renders in the browser.
  4. Segment:   the user clicks an object on one frame; SAM 2.1 (Apache-2.0)
                tracks it through all frames. Each splat remembers the pixel it
                came from, so the object's splats are exactly the ones whose
                source pixel is inside the mask: one identity across all views.

Endpoints
  GET  /scan3d/model-status
  POST /scan3d/start                      (video or images) -> {job_id}
  GET  /scan3d/search?q=...               short YouTube clips to scan (<= 3 min)
  POST /scan3d/start-url                  {url, frames, detail, start?, end?} -> {job_id}  (YouTube etc. via yt-dlp)
  GET  /scan3d/{job}/status
  GET  /scan3d/{job}/scene.ply
  GET  /scan3d/{job}/frame/{i}.jpg
  POST /scan3d/{job}/segment              {frame, points:[[x,y]..] (0..1), labels:[1|0..]}
  GET  /scan3d/{job}/object/{oid}.ply     (?part=object|rest)

Install the models with setup_scan3d.bat (weights download on first use).
"""
from __future__ import annotations

import io
import json
import os
import shutil
import threading
import time
import uuid
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

router = APIRouter(prefix="/scan3d", tags=["scan3d"])

JOBS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scan3d_jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

MAPANYTHING_ID = "facebook/map-anything-apache"
SAM2_ID = "facebook/sam2.1-hiera-small"
SH_C0 = 0.28209479177387814

_jobs: dict = {}                # job_id -> status dict (in memory, mirrored to disk)
_gpu_lock = threading.Lock()    # one heavy model run at a time on the GPU
_ma_model = None
_sam_model = None
_sam_state: dict = {"job": None, "state": None}


# ── status helpers ──────────────────────────────────────────────────────────────
def _job_dir(job_id: str) -> str:
    safe = "".join(c for c in job_id if c.isalnum() or c in "-_")
    return os.path.join(JOBS_DIR, safe)


def _set(job_id: str, **kw):
    j = _jobs.setdefault(job_id, {})
    j.update(kw, ts=time.time())
    try:
        with open(os.path.join(_job_dir(job_id), "status.json"), "w") as f:
            json.dump(j, f)
    except OSError:
        pass


def _get(job_id: str) -> dict:
    if job_id in _jobs:
        return _jobs[job_id]
    p = os.path.join(_job_dir(job_id), "status.json")
    if os.path.isfile(p):
        with open(p) as f:
            _jobs[job_id] = json.load(f)
        return _jobs[job_id]
    raise HTTPException(404, "Scan not found")


def _deps():
    out = {"mapanything": False, "sam2": False, "cuda": False, "gpu": None, "vram_gb": None}
    try:
        import importlib.util
        out["mapanything"] = importlib.util.find_spec("mapanything") is not None
        out["sam2"] = importlib.util.find_spec("sam2") is not None
    except Exception:
        pass
    try:
        import torch
        out["cuda"] = torch.cuda.is_available()
        if out["cuda"]:
            p = torch.cuda.get_device_properties(0)
            out["gpu"] = p.name
            out["vram_gb"] = round(p.total_memory / 1024 ** 3, 1)
    except Exception:
        pass
    return out


@router.get("/model-status")
def model_status():
    d = _deps()
    d["ready"] = d["mapanything"] and d["sam2"]
    d["setup"] = "Run setup_scan3d.bat in the NoCode CV folder, then restart the app."
    return d


# ── frames ──────────────────────────────────────────────────────────────────────
def _sharpness(img: np.ndarray) -> float:
    import cv2
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def _extract_video(path: str, out_dir: str, n: int, long_side: int = 1024,
                   t0: Optional[float] = None, t1: Optional[float] = None) -> int:
    """Pick the sharpest frame in each of n equal time windows (optionally only between t0 and t1 seconds)."""
    import cv2
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        raise ValueError("Could not read the video. Try an .mp4 file.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    f0 = max(0, int((t0 or 0) * fps))
    f1 = min(total, int(t1 * fps)) if t1 else total
    if f1 - f0 < 4:
        raise ValueError("That time range is too short. Pick at least a few seconds of video.")
    n = max(4, min(n, f1 - f0))
    bounds = np.linspace(f0, f1, n + 1).astype(int)
    saved = 0
    for w in range(n):
        best, best_s = None, -1.0
        a, b = bounds[w], max(bounds[w] + 1, bounds[w + 1])
        step = max(1, (b - a) // 6)            # look at up to ~6 candidates per window
        for fi in range(a, b, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, fr = cap.read()
            if not ok:
                continue
            s = _sharpness(fr)
            if s > best_s:
                best, best_s = fr, s
        if best is None:
            continue
        h, wd = best.shape[:2]
        sc = long_side / max(h, wd)
        if sc < 1:
            best = cv2.resize(best, (int(wd * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(out_dir, f"src_{saved:05d}.jpg"), best, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1
    cap.release()
    return saved


# ── splats ──────────────────────────────────────────────────────────────────────
def build_splats(preds: List[dict], stride: int = 2, max_splats: int = 1_500_000, conf_pct: float = 15.0):
    """Turn MapAnything per-view predictions into per-pixel splats.

    Each pred holds numpy arrays: pts3d (H,W,3), img (H,W,3 in 0..1), mask (H,W),
    conf (H,W), depth (H,W), fx (float). Returns dict of flat arrays.
    """
    xs, cols, scs, ops, sv, sy, sx, cf = [], [], [], [], [], [], [], []
    for v, p in enumerate(preds):
        H, W = p["depth"].shape
        yy, xx = np.mgrid[0:H:stride, 0:W:stride]
        yy, xx = yy.ravel(), xx.ravel()
        conf = p["conf"][yy, xx]
        keep = p["mask"][yy, xx] & np.isfinite(p["depth"][yy, xx]) & (p["depth"][yy, xx] > 0)
        if keep.any():
            thr = np.percentile(conf[keep], conf_pct)
            keep &= conf >= thr
        yy, xx = yy[keep], xx[keep]
        if not len(yy):
            continue
        d = p["depth"][yy, xx]
        xs.append(p["pts3d"][yy, xx].astype(np.float32))
        cols.append(np.clip(p["img"][yy, xx] * 255, 0, 255).astype(np.uint8))
        scs.append((d * stride / p["fx"] * 0.62).astype(np.float32))     # half a footprint, slight overlap
        c = p["conf"][yy, xx]
        ops.append(np.clip(0.55 + 0.4 * (c - c.min()) / (np.ptp(c) + 1e-6), 0.5, 0.95).astype(np.float32))
        sv.append(np.full(len(yy), v, np.int16)); sy.append(yy.astype(np.int16)); sx.append(xx.astype(np.int16))
        cf.append(c.astype(np.float32))
    if not xs:
        raise ValueError("No usable 3D points. Try more light, slower movement, or more frames.")
    out = {k: np.concatenate(a) for k, a in
           dict(xyz=xs, rgb=cols, scale=scs, opacity=ops, view=sv, py=sy, px=sx, conf=cf).items()}

    # Many views see the same surface: keep the most confident splat per voxel.
    n = len(out["xyz"])
    if n > max_splats:
        vox = float(np.median(out["scale"])) * 1.6
        q = np.floor(out["xyz"] / max(vox, 1e-6)).astype(np.int64)
        order = np.argsort(-out["conf"], kind="stable")
        _, first = np.unique(q[order], axis=0, return_index=True)
        idx = order[first]
        if len(idx) > max_splats:
            idx = np.random.default_rng(0).choice(idx, max_splats, replace=False)
        idx.sort()
        out = {k: a[idx] for k, a in out.items()}
    return out


def write_splat_ply(path: str, s: dict, idx: Optional[np.ndarray] = None):
    """Standard 3D Gaussian Splatting PLY (degree-0 SH), readable by Spark and most viewers."""
    if idx is not None:
        s = {k: a[idx] for k, a in s.items()}
    n = len(s["xyz"])
    props = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    arr = np.zeros((n, len(props)), np.float32)
    arr[:, 0:3] = s["xyz"]
    arr[:, 6:9] = (s["rgb"].astype(np.float32) / 255.0 - 0.5) / SH_C0
    o = np.clip(s["opacity"], 1e-4, 1 - 1e-4)
    arr[:, 9] = np.log(o / (1 - o))
    arr[:, 10:13] = np.log(np.maximum(s["scale"], 1e-7))[:, None]
    arr[:, 13] = 1.0
    header = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n%send_header\n" % (
        n, "".join(f"property float {p}\n" for p in props))
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())


def bounds(xyz: np.ndarray):
    lo, hi = np.percentile(xyz, 2, axis=0), np.percentile(xyz, 98, axis=0)
    c = (lo + hi) / 2
    return c.tolist(), float(np.linalg.norm(hi - lo) / 2)


def object_size(xyz: np.ndarray):
    """Oriented extents (metres) of an object's splats, robust to stray points."""
    if len(xyz) < 10:
        return [0.0, 0.0, 0.0]
    c = xyz - np.median(xyz, axis=0)
    _, _, vt = np.linalg.svd(c[np.random.default_rng(0).choice(len(c), min(len(c), 20000), replace=False)], full_matrices=False)
    proj = c @ vt.T
    ext = np.percentile(proj, 98, axis=0) - np.percentile(proj, 2, axis=0)
    return sorted([float(e) for e in ext], reverse=True)


# ── models ──────────────────────────────────────────────────────────────────────
def _load_mapanything():
    global _ma_model
    if _ma_model is None:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        import torch
        from mapanything.models import MapAnything
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _ma_model = MapAnything.from_pretrained(MAPANYTHING_ID).to(dev).eval()
    return _ma_model


def _load_sam2():
    global _sam_model
    if _sam_model is None:
        import torch
        from sam2.build_sam import build_sam2_video_predictor_hf
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _sam_model = build_sam2_video_predictor_hf(SAM2_ID, device=dev)
    return _sam_model


# ── the scan job ────────────────────────────────────────────────────────────────
_VIDEO_FMT = "bestvideo[height<=1080][ext=mp4]/bestvideo[height<=1080]/best[height<=1080][ext=mp4]/best[height<=1080]/best"
# YouTube blocks some player clients from time to time (HTTP 403). Try the
# yt-dlp default first, then clients that often still work.
_YT_CLIENTS = [None, "tv_samsung", "visionos", "web_embedded", "mweb", "android"]
_ytdlp_updated = False


def _js_runtime_args() -> list:
    """yt-dlp needs a JavaScript runtime to pass YouTube's checks. Deno comes from
    pip (the 'deno' package puts deno.exe next to python.exe); Node also works."""
    import sys
    args = []
    exe = "deno.exe" if os.name == "nt" else "deno"
    deno = os.path.join(os.path.dirname(sys.executable), exe)
    if not os.path.isfile(deno):
        deno = shutil.which("deno")
    if deno:
        args += ["--js-runtimes", f"deno:{deno}"]
    node = shutil.which("node")
    if node:
        args += ["--js-runtimes", f"node:{node}"]
    return args


def _pip_update_ytdlp(job_id: str) -> bool:
    import subprocess, sys
    _set(job_id, stage="download", msg="YouTube blocked the download. Updating the downloader and trying again…")
    try:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-U", "--quiet", "yt-dlp[default]", "deno"],
                           capture_output=True, text=True, timeout=300)
        return r.returncode == 0
    except Exception:
        return False


def _ytdlp_once(job_id: str, url: str, out_dir: str, client: Optional[str]) -> tuple:
    """Run yt-dlp in its own process (so an update takes effect without restarting the app).
    Returns (ok, title, error_text)."""
    import subprocess, sys
    for f in os.listdir(out_dir):
        if f.startswith("in_url."):
            try: os.remove(os.path.join(out_dir, f))
            except OSError: pass
    cmd = [sys.executable, "-m", "yt_dlp", url,
           "-f", _VIDEO_FMT, "-o", os.path.join(out_dir, "in_url.%(ext)s"),
           "--no-playlist", "--no-part", "--newline", "--quiet", "--progress", "--no-warnings",
           "--match-filters", "duration <=? 10800",
           "--progress-template", "download:PCT %(progress._percent_str)s",
           "--print", "after_move:TITLE %(title)s",
           "--remote-components", "ejs:github", *_js_runtime_args()]
    if client:
        cmd += ["--extractor-args", f"youtube:player_client={client}"]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    flags = 0x08000000 if os.name == "nt" else 0   # CREATE_NO_WINDOW
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         encoding="utf-8", errors="replace", env=env, creationflags=flags)
    title, errs = None, []
    t = threading.Thread(target=lambda: errs.extend(p.stderr.readlines()), daemon=True)
    t.start()
    for line in p.stdout:
        line = line.strip()
        if line.startswith("PCT"):
            try:
                pct = float(line[3:].strip().rstrip("%")) / 100
                _set(job_id, stage="download", progress=round(0.01 + 0.09 * pct, 3),
                     msg=f"Downloading the video… {pct * 100:.0f}%" + (f"  (client: {client})" if client else ""))
            except ValueError:
                pass
        elif line.startswith("TITLE "):
            title = line[6:]
    p.wait(); t.join(timeout=5)
    err = " ".join(l.strip() for l in errs if "ERROR" in l) or " ".join(l.strip() for l in errs[-3:])
    ok = p.returncode == 0 and any(f.startswith("in_url.") for f in os.listdir(out_dir))
    return ok, title, err


def _download_url(job_id: str, url: str, out_dir: str) -> str:
    """Fetch a video from YouTube or any site yt-dlp supports. Video stream only (no audio, no ffmpeg merge)."""
    global _ytdlp_updated
    import importlib.util
    if importlib.util.find_spec("yt_dlp") is None:
        raise RuntimeError("yt-dlp is not installed. Run setup_scan3d.bat again.")
    is_yt = any(h in url for h in ("youtube.com", "youtu.be"))
    clients = _YT_CLIENTS if is_yt else [None]
    err = ""
    for attempt in range(2):
        for c in clients:
            _set(job_id, stage="download", msg="Downloading the video…" + (f"  (trying client: {c})" if c else ""))
            ok, title, err = _ytdlp_once(job_id, url, out_dir, c)
            if ok:
                files = [f for f in os.listdir(out_dir) if f.startswith("in_url.")]
                _set(job_id, source=title or url)
                return files[0]
            blocked = "403" in err or "Forbidden" in err or "Sign in" in err or "po token" in err.lower()
            if not blocked:
                break   # a real error (private video, bad link…): trying other clients won't help
        if attempt == 0 and blocked and not _ytdlp_updated:
            _ytdlp_updated = True
            if _pip_update_ytdlp(job_id):
                continue
        break
    if not err:
        raise RuntimeError("The download produced no video. Check the link is public, plays in a browser, and is under 3 hours.")
    if "403" in err or "Forbidden" in err:
        raise RuntimeError("YouTube blocked the download (HTTP 403) with every method. Try another video, "
                           "or download it yourself and use Upload. Details: " + err[-300:])
    raise RuntimeError(err[-400:])


def _run_scan(job_id: str, n_frames: int, stride: int, max_splats: int,
              url: Optional[str] = None, t0: Optional[float] = None, t1: Optional[float] = None):
    jd = _job_dir(job_id)
    src_dir, frames_dir = os.path.join(jd, "src"), os.path.join(jd, "frames")
    try:
        if url:
            _set(job_id, stage="download", progress=0.01, msg="Connecting to the video site…")
            _download_url(job_id, url, src_dir)
        with _gpu_lock:
            import torch
            from mapanything.utils.image import load_images
            from PIL import Image

            srcs = sorted(f for f in os.listdir(src_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))
            videos = [f for f in os.listdir(src_dir) if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"))]
            if videos:
                _set(job_id, stage="frames", progress=0.05, msg="Picking the sharpest frames from the video…")
                got = _extract_video(os.path.join(src_dir, videos[0]), src_dir, n_frames, t0=t0, t1=t1)
                srcs = sorted(f for f in os.listdir(src_dir) if f.startswith("src_"))
                if got < 4:
                    raise ValueError("The video gave fewer than 4 usable frames.")
            elif len(srcs) > n_frames:
                keep = np.linspace(0, len(srcs) - 1, n_frames).round().astype(int)
                srcs = [srcs[i] for i in sorted(set(keep))]
            if len(srcs) < 2:
                raise ValueError("Add at least 2 photos, or a video.")

            _set(job_id, stage="model", progress=0.12, msg="Loading MapAnything (first run downloads ~5 GB)…")
            model = _load_mapanything()

            _set(job_id, stage="geometry", progress=0.25, msg=f"Reconstructing {len(srcs)} views in 3D…")
            views = load_images([os.path.join(src_dir, f) for f in srcs])
            with torch.inference_mode():
                preds = model.infer(views, memory_efficient_inference=True, use_amp=True, amp_dtype="bf16",
                                    apply_mask=True, mask_edges=True)

            _set(job_id, stage="splats", progress=0.75, msg="Turning pixels into splats…")
            os.makedirs(frames_dir, exist_ok=True)
            per = []
            for i, p in enumerate(preds):
                img = p["img_no_norm"][0].float().cpu().numpy()
                Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(
                    os.path.join(frames_dir, f"{i:05d}.jpg"), quality=92)
                per.append(dict(
                    pts3d=p["pts3d"][0].float().cpu().numpy(),
                    img=img,
                    mask=p["mask"][0, ..., 0].bool().cpu().numpy() if "mask" in p else np.ones(img.shape[:2], bool),
                    conf=p["conf"][0].float().cpu().numpy(),
                    depth=p["depth_z"][0, ..., 0].float().cpu().numpy(),
                    fx=float(p["intrinsics"][0, 0, 0]),
                ))
            del preds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            s = build_splats(per, stride=stride, max_splats=max_splats)
            np.savez_compressed(os.path.join(jd, "splats.npz"), **s)
            write_splat_ply(os.path.join(jd, "scene.ply"), s)
            center, radius = bounds(s["xyz"])
            H, W = per[0]["depth"].shape
            _set(job_id, status="done", stage="done", progress=1.0,
                 msg=f"{len(s['xyz']):,} splats from {len(per)} views",
                 n_views=len(per), width=W, height=H, n_splats=int(len(s["xyz"])),
                 center=center, radius=radius, objects=[])
    except Exception as e:  # noqa: BLE001 - surface any failure to the UI
        _set(job_id, status="error", stage="error", error=f"{type(e).__name__}: {e}")


@router.post("/start")
async def start(files: List[UploadFile] = File(...), frames: int = Form(40), detail: int = Form(2),
                max_splats: int = Form(1_500_000), start: Optional[float] = Form(None), end: Optional[float] = Form(None)):
    d = _deps()
    if not (d["mapanything"] and d["sam2"]):
        raise HTTPException(400, "Scan models are not installed. Run setup_scan3d.bat, then restart NoCode CV.")
    job_id = uuid.uuid4().hex[:12]
    jd = _job_dir(job_id)
    src_dir = os.path.join(jd, "src")
    os.makedirs(src_dir, exist_ok=True)
    for i, f in enumerate(files):
        name = os.path.basename(f.filename or f"file{i}")
        ext = os.path.splitext(name)[1].lower() or ".jpg"
        with open(os.path.join(src_dir, f"in_{i:05d}{ext}"), "wb") as out:
            shutil.copyfileobj(f.file, out)
    _set(job_id, status="running", stage="queued", progress=0.0, msg="Queued…", error=None)
    stride = 1 if detail >= 3 else 2 if detail == 2 else 3
    threading.Thread(target=_run_scan, daemon=True, args=(
        job_id, max(4, min(int(frames), 96)), stride, max(200_000, min(int(max_splats), 4_000_000)),
        None, start, end)).start()
    return {"job_id": job_id}


class UrlReq(BaseModel):
    url: str
    frames: int = 40
    detail: int = 2
    start: Optional[float] = None     # seconds into the video
    end: Optional[float] = None
    max_splats: int = 1_500_000


@router.post("/start-url")
def start_url(req: UrlReq):
    d = _deps()
    if not (d["mapanything"] and d["sam2"]):
        raise HTTPException(400, "Scan models are not installed. Run setup_scan3d.bat, then restart NoCode CV.")
    url = req.url.strip()
    if not url.lower().startswith(("http://", "https://")):
        raise HTTPException(400, "Paste a full link that starts with https://")
    if req.start is not None and req.end is not None and req.end - req.start < 2:
        raise HTTPException(400, "The end time must be at least 2 seconds after the start.")
    job_id = uuid.uuid4().hex[:12]
    src_dir = os.path.join(_job_dir(job_id), "src")
    os.makedirs(src_dir, exist_ok=True)
    _set(job_id, status="running", stage="download", progress=0.0, msg="Queued…", error=None, source=url)
    stride = 1 if req.detail >= 3 else 2 if req.detail == 2 else 3
    threading.Thread(target=_run_scan, daemon=True, args=(
        job_id, max(4, min(int(req.frames), 96)), stride, max(200_000, min(int(req.max_splats), 4_000_000)),
        url, req.start, req.end)).start()
    return {"job_id": job_id}


@router.get("/search")
def search(q: str, max_seconds: int = 180, n: int = 12):
    """Search YouTube for short clips (default up to 3 minutes) to scan."""
    q = q.strip()
    if len(q) < 2:
        raise HTTPException(400, "Type at least 2 characters to search.")
    try:
        import yt_dlp
    except ImportError:
        raise HTTPException(500, "yt-dlp is not installed. Run: venv\\Scripts\\pip install yt-dlp")
    want = max(1, min(int(n), 24))
    opts = {"quiet": True, "no_warnings": True, "extract_flat": "in_playlist", "skip_download": True}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            # ask for more than needed: long videos are filtered out below
            res = ydl.extract_info(f"ytsearch{want * 4}:{q}", download=False) or {}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"YouTube search failed: {e}")
    out = []
    for e in res.get("entries") or []:
        dur = e.get("duration")
        if not e.get("id") or dur is None or dur > max_seconds or dur < 5:
            continue
        thumbs = e.get("thumbnails") or []
        thumb = (thumbs[-1].get("url") if thumbs else None) or f"https://i.ytimg.com/vi/{e['id']}/hqdefault.jpg"
        out.append({
            "id": e["id"], "title": e.get("title") or "", "channel": e.get("channel") or e.get("uploader") or "",
            "duration": int(dur), "thumbnail": thumb,
            "url": e.get("url") if str(e.get("url", "")).startswith("http") else f"https://www.youtube.com/watch?v={e['id']}",
        })
        if len(out) >= want:
            break
    return {"query": q, "results": out}


@router.get("/{job_id}/status")
def status(job_id: str):
    return _get(job_id)


@router.get("/{job_id}/scene.ply")
def scene_ply(job_id: str):
    p = os.path.join(_job_dir(job_id), "scene.ply")
    if not os.path.isfile(p):
        raise HTTPException(404, "Scene not ready")
    return FileResponse(p, media_type="application/octet-stream", filename=f"scan_{job_id}.ply")


@router.get("/{job_id}/frame/{i}.jpg")
def frame(job_id: str, i: int):
    p = os.path.join(_job_dir(job_id), "frames", f"{int(i):05d}.jpg")
    if not os.path.isfile(p):
        raise HTTPException(404, "Frame not found")
    return FileResponse(p, media_type="image/jpeg")


class SegmentReq(BaseModel):
    frame: int
    points: List[List[float]]     # normalised 0..1 (x, y) on that frame
    labels: List[int]             # 1 = include, 0 = exclude
    name: Optional[str] = None


def masks_to_selection(s: dict, masks: dict) -> np.ndarray:
    """A splat belongs to the object when its source pixel is inside the mask of its own frame."""
    sel = np.zeros(len(s["xyz"]), bool)
    for v, m in masks.items():
        rows = s["view"] == v
        if rows.any():
            sel[rows] = m[s["py"][rows], s["px"][rows]]
    return sel


@router.post("/{job_id}/segment")
def segment(job_id: str, req: SegmentReq):
    info = _get(job_id)
    if info.get("status") != "done":
        raise HTTPException(400, "The scan is not finished yet.")
    if not req.points or len(req.points) != len(req.labels):
        raise HTTPException(400, "Click at least one point on the object.")
    jd = _job_dir(job_id)
    W, H = info["width"], info["height"]
    try:
        with _gpu_lock:
            import cv2
            import torch
            pred = _load_sam2()
            with torch.inference_mode():
                if _sam_state["job"] != job_id:
                    _sam_state["state"] = pred.init_state(video_path=os.path.join(jd, "frames"))
                    _sam_state["job"] = job_id
                st = _sam_state["state"]
                pred.reset_state(st)
                pts = np.array([[x * W, y * H] for x, y in req.points], np.float32)
                pred.add_new_points_or_box(st, frame_idx=int(req.frame), obj_id=1, points=pts,
                                           labels=np.array(req.labels, np.int32))
                masks, wide = {}, {}
                kernel = np.ones((3, 3), np.uint8)
                for rev in (False, True):
                    for f, _ids, logits in pred.propagate_in_video(st, start_frame_idx=int(req.frame), reverse=rev):
                        m = (logits[0, 0] > 0).cpu().numpy().astype(np.uint8)
                        masks[int(f)] = cv2.erode(m, kernel).astype(bool)              # object: trim the fuzzy edge
                        wide[int(f)] = cv2.dilate(m, kernel, iterations=2).astype(bool)  # removal: take the edge too
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"Segmentation failed: {type(e).__name__}: {e}")

    s = dict(np.load(os.path.join(jd, "splats.npz")))
    sel = masks_to_selection(s, masks)
    keep_rest = ~masks_to_selection(s, wide)
    n = int(sel.sum())
    if n == 0:
        raise HTTPException(422, "Nothing was found at that point. Try clicking the middle of the object.")
    oid = f"obj{len(info.get('objects', [])) + 1}"
    os.makedirs(os.path.join(jd, "objects"), exist_ok=True)
    write_splat_ply(os.path.join(jd, "objects", f"{oid}_object.ply"), s, np.nonzero(sel)[0])
    write_splat_ply(os.path.join(jd, "objects", f"{oid}_rest.ply"), s, np.nonzero(keep_rest)[0])
    for v, m in masks.items():
        cv2.imwrite(os.path.join(jd, "objects", f"{oid}_mask_{v:05d}.png"), m.astype(np.uint8) * 255)
    size = object_size(s["xyz"][sel])
    center, radius = bounds(s["xyz"][sel])
    obj = dict(id=oid, name=req.name or f"Object {len(info.get('objects', [])) + 1}", n_splats=n,
               size_m=size, center=center, radius=radius, frames_seen=int(sum(1 for m in masks.values() if m.any())))
    _set(job_id, objects=[*info.get("objects", []), obj])
    return obj


@router.get("/{job_id}/object/{oid}.ply")
def object_ply(job_id: str, oid: str, part: str = "object"):
    safe = "".join(c for c in oid if c.isalnum())
    p = os.path.join(_job_dir(job_id), "objects", f"{safe}_{'rest' if part == 'rest' else 'object'}.ply")
    if not os.path.isfile(p):
        raise HTTPException(404, "Object not found")
    return FileResponse(p, media_type="application/octet-stream", filename=f"{safe}_{part}.ply")
