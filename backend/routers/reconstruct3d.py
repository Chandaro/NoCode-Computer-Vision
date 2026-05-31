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


# ─── Model loader ─────────────────────────────────────────────────────────────
def _ensure_model():
    """Load and cache the DUSt3R model.  Thread-safe, lazy, raises on error."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        try:
            import torch
            from dust3r.model import AsymmetricCroCo3DStereo
            device = "cuda" if torch.cuda.is_available() else "cpu"
            m = AsymmetricCroCo3DStereo.from_pretrained(
                "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
            ).to(device)
            m.eval()
            _model = m
            return _model
        except ImportError:
            raise RuntimeError(
                "DUSt3R is not installed. "
                "Run setup_dust3r.bat (Windows) or setup_dust3r.sh (Mac/Linux) "
                "from the NoCode CV folder, then restart the server."
            )


# ─── Core reconstruction (runs in background thread) ─────────────────────────
def _run_reconstruction(job_id: str, image_bytes_list: list, niter: int):
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
        scene.min_conf_thr = 1.5

        pts3d_list = [p.detach().cpu().numpy() for p in scene.get_pts3d()]
        conf_masks = [m.detach().cpu().numpy() for m in scene.get_masks()]
        rgb_list   = scene.imgs   # list of [H, W, 3] float32 in [0, 1]

        xyz_parts, rgb_parts = [], []
        for pts, rgb, mask in zip(pts3d_list, rgb_list, conf_masks):
            xyz_parts.append(pts[mask])
            rgb_parts.append(rgb[mask])

        xyz = np.concatenate(xyz_parts, axis=0).astype(np.float32)  # [N, 3]
        rgb = np.concatenate(rgb_parts, axis=0).astype(np.float32)  # [N, 3] 0-1

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

        # ── Subsample for the viewer (keep full data for PLY download) ────
        n_full = len(xyz)
        if n_full > MAX_VIEWER_PTS:
            step     = n_full // MAX_VIEWER_PTS
            xyz_view = xyz[::step]
            rgb_view = rgb[::step]
        else:
            xyz_view = xyz
            rgb_view = rgb

        n_view  = len(xyz_view)
        pos_list = [round(float(v), 5) for v in xyz_view.flatten()]
        col_list = [round(float(v), 4) for v in rgb_view.flatten()]

        _recon_jobs[job_id].update({
            "status": "done",
            "msg":    f"Done — {n_full:,} pts from {n_imgs} images "
                      f"({n_view:,} shown in viewer)",
            "result": {"n": n_view, "pos": pos_list, "col": col_list},
            "xyz":    xyz,           # full-res for PLY download
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

@router.post("/reconstruct3d/start")
async def start_reconstruction(
    files: List[UploadFile] = File(...),
    niter: int              = Form(200),
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
        args=(job_id, image_bytes_list, niter),
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
