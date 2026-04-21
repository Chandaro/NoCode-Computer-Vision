from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from sqlmodel import Session, select
import os, threading, json, zipfile
from io import BytesIO

from database import get_session, DATABASE_URL
from models import TrainingRun, Project, Image, Annotation

# ── Per-run model export (ONNX / TFLite / TensorRT) ──────────────────────────
router = APIRouter(
    prefix="/projects/{project_id}/training/runs/{run_id}/export",
    tags=["export"],
)

# ── Project-level dataset export ─────────────────────────────────────────────
dataset_router = APIRouter(prefix="/projects/{project_id}", tags=["dataset-export"])

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")

# (run_id, fmt) → { status, error, path }
_export_state: dict = {}


def _do_export(run_id: int, model_path: str, fmt: str):
    key = f"{run_id}_{fmt}"
    _export_state[key] = {"status": "running", "error": "", "path": ""}
    try:
        from ultralytics import YOLO
        model   = YOLO(model_path)
        out     = model.export(format=fmt, imgsz=640)
        out_str = str(out)
        _export_state[key] = {"status": "done", "error": "", "path": out_str}
        if fmt == "onnx":
            _persist_onnx(run_id, out_str)
    except Exception as exc:
        _export_state[key] = {"status": "failed", "error": str(exc), "path": ""}


def _persist_onnx(run_id: int, onnx_path: str):
    try:
        from sqlmodel import create_engine, Session as S
        engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
        with S(engine) as session:
            run = session.get(TrainingRun, run_id)
            if run and os.path.exists(onnx_path):
                run.onnx_path = onnx_path
                session.add(run); session.commit()
    except Exception:
        pass


def _get_run(project_id: int, run_id: int, session: Session) -> TrainingRun:
    run = session.get(TrainingRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(404, "Run not found")
    if run.status != "done":
        raise HTTPException(400, "Run has not completed")
    if not run.model_path or not os.path.exists(run.model_path):
        raise HTTPException(404, "Model file not found")
    return run


def _find_file(run_id: int, fmt: str, model_path: str, ext: str) -> str | None:
    path = _export_state.get(f"{run_id}_{fmt}", {}).get("path", "")
    if path and os.path.exists(path):
        return path
    derived = os.path.splitext(model_path)[0] + ext
    return derived if os.path.exists(derived) else None


# ── ONNX ─────────────────────────────────────────────────────────────────────
@router.post("/onnx")
def start_onnx(project_id: int, run_id: int, session: Session = Depends(get_session)):
    run = _get_run(project_id, run_id, session)
    if run.onnx_path and os.path.exists(run.onnx_path):
        return {"status": "done"}
    if _export_state.get(f"{run_id}_onnx", {}).get("status") == "running":
        return {"status": "running"}
    threading.Thread(target=_do_export, args=(run_id, run.model_path, "onnx"), daemon=True).start()
    return {"status": "running"}


@router.get("/onnx/status")
def onnx_status(project_id: int, run_id: int, session: Session = Depends(get_session)):
    run = session.get(TrainingRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(404)
    if run.onnx_path and os.path.exists(run.onnx_path):
        return {"status": "done"}
    st = _export_state.get(f"{run_id}_onnx", {})
    return {"status": st.get("status", "idle"), "error": st.get("error", "")}


@router.get("/onnx/download")
def download_onnx(project_id: int, run_id: int, session: Session = Depends(get_session)):
    run = session.get(TrainingRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(404)
    path = run.onnx_path or _find_file(run_id, "onnx", run.model_path or "", ".onnx")
    if not path:
        raise HTTPException(404, "ONNX file not found — run export first")
    return FileResponse(path, filename=f"model_run{run_id}.onnx",
                        media_type="application/octet-stream")


# ── TFLite ────────────────────────────────────────────────────────────────────
@router.post("/tflite")
def start_tflite(project_id: int, run_id: int, session: Session = Depends(get_session)):
    run = _get_run(project_id, run_id, session)
    if _export_state.get(f"{run_id}_tflite", {}).get("status") == "running":
        return {"status": "running"}
    threading.Thread(target=_do_export, args=(run_id, run.model_path, "tflite"), daemon=True).start()
    return {"status": "running"}


@router.get("/tflite/status")
def tflite_status(project_id: int, run_id: int, session: Session = Depends(get_session)):
    run = session.get(TrainingRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(404)
    st = _export_state.get(f"{run_id}_tflite", {})
    return {"status": st.get("status", "idle"), "error": st.get("error", "")}


@router.get("/tflite/download")
def download_tflite(project_id: int, run_id: int, session: Session = Depends(get_session)):
    run = session.get(TrainingRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(404)
    path = _find_file(run_id, "tflite", run.model_path or "", ".tflite")
    if not path and run.model_path:
        for root, _, files in os.walk(os.path.dirname(run.model_path)):
            for f in files:
                if f.endswith(".tflite"):
                    path = os.path.join(root, f)
                    break
    if not path or not os.path.exists(path):
        raise HTTPException(404, "TFLite file not found — run export first")
    return FileResponse(path, filename=f"model_run{run_id}.tflite",
                        media_type="application/octet-stream")


# ── TensorRT ──────────────────────────────────────────────────────────────────
@router.post("/tensorrt")
def start_tensorrt(project_id: int, run_id: int, session: Session = Depends(get_session)):
    run = _get_run(project_id, run_id, session)
    if _export_state.get(f"{run_id}_engine", {}).get("status") == "running":
        return {"status": "running"}
    threading.Thread(target=_do_export, args=(run_id, run.model_path, "engine"), daemon=True).start()
    return {"status": "running"}


@router.get("/tensorrt/status")
def tensorrt_status(project_id: int, run_id: int, session: Session = Depends(get_session)):
    run = session.get(TrainingRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(404)
    st = _export_state.get(f"{run_id}_engine", {})
    return {"status": st.get("status", "idle"), "error": st.get("error", "")}


@router.get("/tensorrt/download")
def download_tensorrt(project_id: int, run_id: int, session: Session = Depends(get_session)):
    run = session.get(TrainingRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(404)
    path = _find_file(run_id, "engine", run.model_path or "", ".engine")
    if not path:
        raise HTTPException(404, "TensorRT engine not found — run export first")
    return FileResponse(path, filename=f"model_run{run_id}.engine",
                        media_type="application/octet-stream")


# ── Dataset export ────────────────────────────────────────────────────────────
@dataset_router.get("/export/dataset")
def export_dataset(
    project_id: int,
    format: str = "yolo",
    session: Session = Depends(get_session),
):
    """Download annotated dataset as zip. format: 'yolo' (default) or 'coco'"""
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    images = session.exec(select(Image).where(Image.project_id == project_id)).all()
    if not images:
        raise HTTPException(400, "No images in project")
    return _export_coco(project, images, session) if format == "coco" \
        else _export_yolo(project, images, session)


def _export_yolo(project, images, session):
    buf  = BytesIO()
    names = project.classes
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("data.yaml",
                    f"nc: {len(names)}\nnames: {json.dumps(names)}\n")
        for img in images:
            src = os.path.join(UPLOAD_DIR, img.filename)
            if not os.path.exists(src):
                continue
            zf.write(src, f"images/{img.original_name}")
            anns = session.exec(
                select(Annotation).where(Annotation.image_id == img.id)).all()
            if not anns:
                continue
            lines = []
            for a in anns:
                if a.shape_type == "polygon":
                    pts = json.loads(a.points_json or "[]")
                    if len(pts) >= 3:
                        lines.append(f"{a.class_id} " +
                                     " ".join(f"{p[0]:.6f} {p[1]:.6f}" for p in pts))
                elif a.shape_type == "point":
                    pts = json.loads(a.points_json or "[]")
                    if pts:
                        x, y = pts[0]
                        lines.append(f"{a.class_id} {x:.6f} {y:.6f} 0.008 0.008")
                else:
                    lines.append(
                        f"{a.class_id} {a.x_center:.6f} {a.y_center:.6f} "
                        f"{a.width:.6f} {a.height:.6f}"
                    )
            stem = os.path.splitext(img.original_name)[0]
            zf.writestr(f"labels/{stem}.txt", "\n".join(lines))

    buf.seek(0)
    slug = project.name.replace(" ", "_")
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition":
                                      f'attachment; filename="{slug}_yolo.zip"'})


def _export_coco(project, images, session):
    coco = {
        "info":        {"description": project.name, "version": "1.0"},
        "categories":  [{"id": i + 1, "name": c}
                        for i, c in enumerate(project.classes)],
        "images":      [],
        "annotations": [],
    }
    ann_id = 1
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for img in images:
            src = os.path.join(UPLOAD_DIR, img.filename)
            if not os.path.exists(src):
                continue
            w = img.width or 640
            h = img.height or 640
            coco["images"].append({"id": img.id, "file_name": img.original_name,
                                   "width": w, "height": h})
            zf.write(src, f"images/{img.original_name}")
            for a in session.exec(
                select(Annotation).where(Annotation.image_id == img.id)).all():
                if a.shape_type != "bbox":
                    continue
                bx = round((a.x_center - a.width  / 2) * w, 2)
                by = round((a.y_center - a.height / 2) * h, 2)
                bw = round(a.width  * w, 2)
                bh = round(a.height * h, 2)
                coco["annotations"].append({
                    "id": ann_id, "image_id": img.id,
                    "category_id": a.class_id + 1,
                    "bbox": [bx, by, bw, bh],
                    "area": round(bw * bh, 2), "iscrowd": 0,
                })
                ann_id += 1
        zf.writestr("annotations.json", json.dumps(coco, indent=2))

    buf.seek(0)
    slug = project.name.replace(" ", "_")
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition":
                                      f'attachment; filename="{slug}_coco.zip"'})
