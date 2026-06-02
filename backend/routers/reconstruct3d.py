"""
reconstruct3d.py — DUSt3R multi-view 3D reconstruction router

POST  /reconstruct3d/start                → starts background job, returns {job_id}
GET   /reconstruct3d/{job_id}/status      → {status, msg, error}
GET   /reconstruct3d/{job_id}/result      → {n, pos, col} JSON for Three.js viewer
GET   /reconstruct3d/{job_id}/download    → binary PLY file download

DUSt3R must be installed first — run setup_dust3r.bat.
Model: naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt (~330 MB, auto-downloaded)
"""
from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse
from typing import List
import os, io, uuid, threading, struct, time

router = APIRouter(tags=["reconstruct3d"])

# ── In-memory job registry ────────────────────────────────────────────────────
_recon_jobs: dict = {}   # job_id → {status, msg, result, xyz, rgb, error, ts}

# ── Cached DUSt3R model (loaded once on first use) ───────────────────────────
_model      = None
_model_lock = threading.Lock()

MAX_VIEWER_PTS = 200_000   # subsample result to keep Three.js fast


# ─── croco 'models' namespace-package collision fix ──────────────────────────
#
# DUSt3R's croco submodule imports its internal packages with bare names:
#     from models.dpt_block import DPTOutputAdapter
#     from models.blocks import ...
# croco/models/ is an implicit namespace package (no __init__.py).
#
# Our backend has its OWN backend/models.py (the SQLModel DB definitions).
# Once the server imports that, sys.modules['models'] points to our file —
# a regular module, NOT a package — so croco's `from models.X import` fails
# with "ModuleNotFoundError: 'models' is not a package".
#
# This context manager temporarily removes our 'models' from sys.modules and
# prepends croco's path so the bare `models` / `utils` imports resolve to
# croco's namespace packages, then restores our 'models' afterwards.  Class
# references bound during croco's import survive the restore, so subsequent
# model use is unaffected.
import sys
import importlib.machinery
import importlib.util
from contextlib import contextmanager


@contextmanager
def _croco_import_context():
    """
    Make croco's bare `from models.X import` / `from utils.X import` resolve
    to croco's namespace packages instead of backend/models.py.

    Path-ordering alone does not work: a regular module file (backend/models.py)
    always takes precedence over a namespace-directory portion during import
    search.  So we INJECT croco's namespace packages directly into sys.modules,
    bypassing the path search entirely, then restore the originals afterward.
    """
    import dust3r
    croco_path = os.path.normpath(
        os.path.join(os.path.dirname(dust3r.__file__), "..", "croco")
    )

    shadowed = ("models", "utils")

    # Snapshot originals (backend's 'models' etc.) and any submodules.
    saved = {k: v for k, v in sys.modules.items()
             if k in shadowed or any(k.startswith(s + ".") for s in shadowed)}
    for k in list(saved):
        del sys.modules[k]

    # Inject a namespace package for each shadowed name pointing at croco/<name>/.
    injected = []
    for name in shadowed:
        pkg_dir = os.path.join(croco_path, name)
        if not os.path.isdir(pkg_dir):
            continue
        spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
        spec.submodule_search_locations = [pkg_dir]
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        injected.append(name)

    path_added = croco_path not in sys.path
    if path_added:
        sys.path.insert(0, croco_path)

    try:
        yield
    finally:
        # Remove croco's transient modules, restore the originals.
        for k in list(sys.modules.keys()):
            if k in shadowed or any(k.startswith(s + ".") for s in shadowed):
                del sys.modules[k]
        sys.modules.update(saved)
        if path_added:
            try:
                sys.path.remove(croco_path)
            except ValueError:
                pass


# ─── Model loader ─────────────────────────────────────────────────────────────
def _ensure_model():
    """Load and cache the DUSt3R model.  Thread-safe, lazy, raises on error."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model

        # 1. Is the dust3r package importable at all?
        try:
            import dust3r  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "DUSt3R is not installed. "
                "Run setup_dust3r.bat (Windows) or setup_dust3r.sh (Mac/Linux) "
                "from the NoCode CV folder, then restart the server."
            )

        # 2. dust3r IS installed — load the model with the croco import fix.
        try:
            import torch
            with _croco_import_context():
                from dust3r.model import AsymmetricCroCo3DStereo
            device = "cuda" if torch.cuda.is_available() else "cpu"
            m = AsymmetricCroCo3DStereo.from_pretrained(
                "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
            ).to(device)
            m.eval()
            _model = m
            return _model
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            raise RuntimeError(
                f"DUSt3R is installed but failed to load the model: "
                f"{type(exc).__name__}: {exc}\n\n{tb}"
            )


# ─── Point cloud cleanup helpers ─────────────────────────────────────────────
def _remove_statistical_outliers(xyz, rgb, k: int = 16, std_ratio: float = 2.0):
    """
    Remove floating noise points (the #1 cause of ugly DUSt3R output).

    For each point, compute the mean distance to its k nearest neighbours.
    Points whose mean distance exceeds (global_mean + std_ratio*global_std)
    are statistical outliers — isolated floaters — and get removed.
    Same algorithm as Open3D's remove_statistical_outlier, using scipy.
    """
    import numpy as np
    if len(xyz) < k + 1:
        return xyz, rgb
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return xyz, rgb
    tree = cKDTree(xyz)
    # k+1 because the nearest neighbour of a point is itself (distance 0)
    dists, _ = tree.query(xyz, k=k + 1, workers=-1)
    mean_d   = dists[:, 1:].mean(axis=1)          # drop self-distance
    thresh   = mean_d.mean() + std_ratio * mean_d.std()
    keep     = mean_d < thresh
    return xyz[keep], rgb[keep]


def _voxel_downsample(xyz, rgb, voxel: float):
    """
    Downsample to one averaged point per voxel cell — gives uniform point
    density (much cleaner than naive [::step] which keeps dense/sparse patches).
    """
    import numpy as np
    if voxel <= 0 or len(xyz) == 0:
        return xyz, rgb
    keys = np.floor(xyz / voxel).astype(np.int64)
    # Map each unique voxel to the indices of points inside it.
    # numpy 2.x can return a 2-D inverse with axis= — flatten to 1-D.
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    n_vox = int(inverse.max()) + 1
    out_xyz = np.zeros((n_vox, 3), dtype=np.float32)
    out_rgb = np.zeros((n_vox, 3), dtype=np.float32)
    counts  = np.zeros(n_vox, dtype=np.int64)
    np.add.at(out_xyz, inverse, xyz)
    np.add.at(out_rgb, inverse, rgb)
    np.add.at(counts, inverse, 1)
    counts = counts[:, None].clip(min=1)
    return out_xyz / counts, out_rgb / counts


# ─── Core reconstruction (runs in background thread) ─────────────────────────
def _run_reconstruction(job_id: str, image_bytes_list: list, niter: int,
                        conf_thr: float = 1.5, cleanup: bool = True):
    def push(msg: str):
        if job_id in _recon_jobs:
            _recon_jobs[job_id]["msg"] = msg

    try:
        import torch, numpy as np, tempfile
        from PIL import Image as PILImage

        # ── Lazy imports — fail fast with clear error ──────────────────────
        try:
            from dust3r.utils.image import load_images
            from dust3r.image_pairs import make_pairs
            from dust3r.inference import inference
            from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
        except ImportError as e:
            _recon_jobs[job_id].update({
                "status": "failed",
                "error":  f"DUSt3R not installed: {e}. Run setup_dust3r.bat.",
            })
            return

        push("Loading DUSt3R model…")
        model  = _ensure_model()
        device = next(model.parameters()).device

        # ── Write images to temp files (load_images needs file paths) ────
        push("Preparing images…")
        tmp_paths = []
        try:
            for img_bytes in image_bytes_list:
                pil = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    pil.save(f.name)
                    tmp_paths.append(f.name)

            n_imgs = len(tmp_paths)
            push(f"Loading {n_imgs} images…")
            images = load_images(tmp_paths, size=512, verbose=False)
        finally:
            for p in tmp_paths:
                try: os.unlink(p)
                except: pass

        # ── Pairwise inference ────────────────────────────────────────────
        graph  = "complete" if n_imgs <= 12 else "oneref"
        n_pairs = n_imgs * (n_imgs - 1) if graph == "complete" else n_imgs - 1
        push(f"Running pairwise inference ({n_pairs} pairs)…")
        pairs  = make_pairs(images, scene_graph=graph, symmetrize=True)

        with torch.no_grad():
            output = inference(pairs, model, device, batch_size=1)

        # ── Global alignment ──────────────────────────────────────────────
        mode = (
            GlobalAlignerMode.PairViewer
            if n_imgs <= 2 else
            GlobalAlignerMode.PointCloudOptimizer
        )
        push("Computing global alignment…")
        scene = global_aligner(output, device=device, mode=mode)
        if mode == GlobalAlignerMode.PointCloudOptimizer:
            scene.compute_global_alignment(
                init="mst", niter=niter, schedule="cosine", lr=0.01
            )

        # ── Extract XYZ + RGB ─────────────────────────────────────────────
        push("Extracting point cloud…")
        # Confidence threshold: lower = more points (more detail, more noise),
        # higher = fewer points (cleaner).  Driven by the UI "Detail" slider.
        scene.min_conf_thr = float(conf_thr)

        pts3d_list = [p.detach().cpu().numpy() for p in scene.get_pts3d()]
        conf_masks = [m.detach().cpu().numpy() for m in scene.get_masks()]
        rgb_list   = scene.imgs   # list of [H, W, 3] float32 in [0, 1]

        xyz_parts, rgb_parts = [], []
        for pts, rgb, mask in zip(pts3d_list, rgb_list, conf_masks):
            xyz_parts.append(pts[mask])
            rgb_parts.append(rgb[mask])

        xyz = np.concatenate(xyz_parts, axis=0).astype(np.float32)  # [N, 3]
        rgb = np.concatenate(rgb_parts, axis=0).astype(np.float32)  # [N, 3] 0-1

        n_raw = len(xyz)

        # ── OpenCV → Three.js coordinate flip ────────────────────────────
        # DUSt3R: Z-forward, Y-down  →  Three.js: Z-toward-viewer, Y-up
        xyz[:, 1] *= -1
        xyz[:, 2] *= -1

        # ── Normalise to unit sphere for stable camera placement ──────────
        centroid = xyz.mean(axis=0)
        xyz -= centroid
        scale = float(np.abs(xyz).max())
        if scale > 1e-8:
            xyz /= scale

        # ── Statistical outlier removal — strips floating noise points ────
        if cleanup and n_raw > 100:
            push("Cleaning point cloud (removing noise)…")
            xyz, rgb = _remove_statistical_outliers(xyz, rgb, k=16, std_ratio=2.0)

        n_full = len(xyz)

        # ── Voxel downsample for the viewer (uniform density, not [::step]) ─
        if n_full > MAX_VIEWER_PTS:
            push("Optimising for display…")
            # Pick a voxel size that lands near the target point budget.
            # Coordinates are normalised to roughly [-1, 1] → span ~2.
            voxel = 2.0 / (MAX_VIEWER_PTS ** (1 / 3)) * 1.1
            xyz_view, rgb_view = _voxel_downsample(xyz, rgb, voxel)
            # If still too many, fall back to a uniform stride.
            if len(xyz_view) > MAX_VIEWER_PTS:
                step = len(xyz_view) // MAX_VIEWER_PTS
                xyz_view = xyz_view[::step]
                rgb_view = rgb_view[::step]
        else:
            xyz_view = xyz
            rgb_view = rgb

        n_view  = len(xyz_view)
        pos_list = [round(float(v), 5) for v in xyz_view.flatten()]
        col_list = [round(float(v), 4) for v in rgb_view.flatten()]

        removed = n_raw - n_full
        clean_note = f" · {removed:,} noise pts removed" if (cleanup and removed > 0) else ""
        _recon_jobs[job_id].update({
            "status": "done",
            "msg":    f"Done — {n_full:,} pts from {n_imgs} images{clean_note} "
                      f"({n_view:,} shown)",
            "result": {"n": n_view, "pos": pos_list, "col": col_list},
            "xyz":    xyz,           # cleaned full-res for PLY download
            "rgb":    rgb,
        })

    except Exception as exc:
        import traceback
        _recon_jobs[job_id].update({
            "status": "failed",
            "error":  str(exc),
            "msg":    f"Failed: {exc}",
        })


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.get("/reconstruct3d/model-status")
def model_status():
    """
    Report whether DUSt3R is installed and whether its model is cached.
    Lets the UI warn the user BEFORE they upload photos and hit Reconstruct.

    Returns:
        installed     — is the dust3r package importable?
        model_cached  — are the model weights already in the HF cache?
        loaded        — is the model already loaded in memory (instant)?
    """
    # 1. dust3r package present?
    try:
        import dust3r  # noqa: F401
        installed = True
    except ImportError:
        installed = False

    # 2. model weights in HF cache? (check without downloading or loading)
    model_cached = False
    if installed:
        try:
            from huggingface_hub import try_to_load_from_cache
            # The model repo's main weight file
            hit = try_to_load_from_cache(
                "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt",
                "model.safetensors",
            )
            if hit is None:
                hit = try_to_load_from_cache(
                    "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt",
                    "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
                )
            model_cached = isinstance(hit, str) and os.path.exists(hit)
        except Exception:
            model_cached = False

    return {
        "installed":    installed,
        "model_cached": model_cached,
        "loaded":       _model is not None,
    }


@router.post("/reconstruct3d/start")
async def start_reconstruction(
    files:    List[UploadFile] = File(...),
    niter:    int              = Form(200),
    conf_thr: float            = Form(1.5),    # detail level (lower=more pts)
    cleanup:  bool             = Form(True),   # statistical outlier removal
):
    """Accept 2–12 images and start a background DUSt3R reconstruction job."""
    if not (2 <= len(files) <= 12):
        raise HTTPException(400, "Upload between 2 and 12 images (more angles = better quality)")

    # Read all files up-front before the thread starts
    image_bytes_list = []
    for f in files:
        if f.content_type and not f.content_type.startswith("image/"):
            raise HTTPException(400, f"{f.filename} is not an image")
        image_bytes_list.append(await f.read())

    job_id = uuid.uuid4().hex
    _recon_jobs[job_id] = {
        "status":   "queued",
        "msg":      "Queued…",
        "result":   None,
        "xyz":      None,
        "rgb":      None,
        "error":    None,
        "n_images": len(files),
        "ts":       time.time(),
    }

    # Purge jobs older than 2 hours
    now = time.time()
    stale = [k for k, v in _recon_jobs.items() if now - v.get("ts", 0) > 7200]
    for k in stale:
        del _recon_jobs[k]

    threading.Thread(
        target=_run_reconstruction,
        args=(job_id, image_bytes_list, niter, conf_thr, cleanup),
        daemon=True,
    ).start()

    return {"job_id": job_id, "n_images": len(files)}


@router.get("/reconstruct3d/{job_id}/status")
def recon_status(job_id: str):
    job = _recon_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "status":   job["status"],
        "msg":      job.get("msg", ""),
        "error":    job.get("error"),
        "n_images": job.get("n_images", 0),
    }


@router.get("/reconstruct3d/{job_id}/result")
def recon_result(job_id: str):
    """Return {n, pos, col} JSON for the Three.js point cloud viewer."""
    job = _recon_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != "done":
        raise HTTPException(400, f"Job not ready yet: {job['status']}")
    return job["result"]


@router.get("/reconstruct3d/{job_id}/download")
def recon_download(job_id: str):
    """Download the full-resolution point cloud as a binary PLY file."""
    job = _recon_jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "Result not ready")

    import numpy as np
    xyz    = job["xyz"]
    rgb_f  = job["rgb"]
    n      = len(xyz)
    rgb_u8 = (rgb_f * 255).clip(0, 255).astype(np.uint8)

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

    body = bytearray()
    for i in range(n):
        body += struct.pack("<fff", xyz[i, 0], xyz[i, 1], xyz[i, 2])
        body += struct.pack("BBB",  rgb_u8[i, 0], rgb_u8[i, 1], rgb_u8[i, 2])

    return StreamingResponse(
        io.BytesIO(header + bytes(body)),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition":
                f'attachment; filename="reconstruction_{job_id[:8]}.ply"'
        },
    )
