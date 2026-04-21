import torch as _torch

# PyTorch 2.6+ defaults weights_only=True which blocks YOLOv8 .pt files.
# Patch torch.load globally so ultralytics can load models without errors.
_orig_torch_load = _torch.load
def _permissive_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
_torch.load = _permissive_load

import os
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from database import create_db
from routers import projects, images, annotations, training
from routers import analytics, evaluation, export, classification, infer, external_models

app = FastAPI(title="NoCode CV Trainer")

app.add_middleware(
    CORSMiddleware,
    # Allow both dev (Vite :5173) and prod (same-origin :8000)
    allow_origins=["http://localhost:5173", "http://localhost:8000",
                   "http://127.0.0.1:5173", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    create_db()


# ─── API routes under /api prefix ─────────────────────────────────────────────
# The frontend uses axios baseURL='/api', so every call is /api/projects/...
# In dev mode Vite's proxy strips /api before forwarding to FastAPI.
# In production (single-server mode) FastAPI must handle /api/... directly.
api = APIRouter(prefix="/api")
api.include_router(projects.router)
api.include_router(images.router)
api.include_router(annotations.router)
api.include_router(training.router)
api.include_router(analytics.router)
api.include_router(evaluation.router)
api.include_router(export.router)
api.include_router(export.dataset_router)
api.include_router(classification.router)
api.include_router(infer.router)
api.include_router(external_models.router)
app.include_router(api)


# /health stays at root — used by the launcher to poll server readiness
@app.get("/health")
def health():
    return {"status": "ok"}


# ─── Serve built frontend (production mode) ───────────────────────────────────
import mimetypes

# Force-register correct MIME types — Windows registry often has wrong values
# for .js files (returns text/plain instead of application/javascript),
# which causes browsers to block ES module execution → blank page.
_MIME = {
    ".js":    "application/javascript",
    ".mjs":   "application/javascript",
    ".css":   "text/css",
    ".html":  "text/html; charset=utf-8",
    ".svg":   "image/svg+xml",
    ".png":   "image/png",
    ".jpg":   "image/jpeg",
    ".jpeg":  "image/jpeg",
    ".ico":   "image/x-icon",
    ".json":  "application/json",
    ".woff":  "font/woff",
    ".woff2": "font/woff2",
    ".ttf":   "font/ttf",
    ".map":   "application/json",
}

_DIST = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
)


def _file_response(path: str) -> FileResponse:
    """Return FileResponse with correct MIME type regardless of OS registry."""
    ext = os.path.splitext(path)[1].lower()
    media_type = _MIME.get(ext) or mimetypes.guess_type(path)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type)


def _dist_ok() -> bool:
    return os.path.isdir(_DIST) and os.path.isfile(os.path.join(_DIST, "index.html"))


# Explicit root route — avoids any ambiguity with the path parameter
@app.get("/", include_in_schema=False)
async def serve_root():
    if not _dist_ok():
        return JSONResponse({"error": "Frontend not built. Run: cd frontend && npm run build"}, 503)
    return _file_response(os.path.join(_DIST, "index.html"))


# All other non-API paths: serve matching file or fall back to index.html (SPA routing)
@app.get("/{full_path:path}", include_in_schema=False)
async def serve_spa(full_path: str):
    if not _dist_ok():
        return JSONResponse({"error": "Frontend not built. Run: cd frontend && npm run build"}, 503)

    candidate = os.path.normpath(os.path.join(_DIST, full_path))

    # Security: block path traversal attempts
    if not candidate.startswith(_DIST):
        return JSONResponse({"error": "forbidden"}, status_code=403)

    # Serve exact file if it exists (JS, CSS, images, icons, etc.)
    if os.path.isfile(candidate):
        return _file_response(candidate)

    # SPA fallback — React Router handles the path client-side
    return _file_response(os.path.join(_DIST, "index.html"))
