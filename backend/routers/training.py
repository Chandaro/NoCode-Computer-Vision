from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from sqlmodel import Session, select
from pydantic import BaseModel
from typing import Optional
import os, json, shutil, threading, time, uuid
from datetime import datetime

from database import get_session, DATABASE_URL
from models import TrainingRun, Project, Image, Annotation

router = APIRouter(prefix="/projects/{project_id}/training", tags=["training"])

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")
RUNS_DIR   = os.path.join(os.path.dirname(__file__), "..", "runs")
os.makedirs(RUNS_DIR, exist_ok=True)

_state: dict = {}


class TrainConfig(BaseModel):
    epochs: int = 50
    imgsz: int = 640
    batch: int = 16
    model_base: str = "yolov8n.pt"
    val_split: float = 0.2
    # Augmentation params (YOLO defaults)
    fliplr: float = 0.5
    flipud: float = 0.0
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    mosaic: float = 1.0
    mixup: float = 0.0
    resume_run_id: Optional[int] = None


class TrainingRunOut(BaseModel):
    id: int; project_id: int; status: str; epochs: int; imgsz: int
    batch: int; model_base: str; model_path: str; results: dict
    created_at: str; run_dir: str = ""; onnx_path: str = ""

    class Config:
        from_attributes = True


# ─── Dataset builder ─────────────────────────────────────────────────────────
def _build_dataset(project_id: int, val_split: float, session: Session) -> str:
    project     = session.get(Project, project_id)
    dataset_dir = os.path.join(RUNS_DIR, f"dataset_{project_id}_{uuid.uuid4().hex[:6]}")

    for split in ["train", "val"]:
        os.makedirs(os.path.join(dataset_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, "labels", split), exist_ok=True)

    images = session.exec(select(Image).where(Image.project_id == project_id)).all()
    n_val  = max(1, int(len(images) * val_split))
    val_ids = {img.id for img in images[:n_val]}

    for img in images:
        split = "val" if img.id in val_ids else "train"
        src = os.path.join(UPLOAD_DIR, img.filename)
        if not os.path.exists(src):
            continue
        shutil.copy2(src, os.path.join(dataset_dir, "images", split, img.filename))
        anns = session.exec(select(Annotation).where(Annotation.image_id == img.id)).all()
        label_path = os.path.join(dataset_dir, "labels", split,
                                  os.path.splitext(img.filename)[0] + ".txt")
        with open(label_path, "w") as f:
            for ann in anns:
                if ann.shape_type == "polygon":
                    pts = json.loads(ann.points_json or "[]")
                    if len(pts) >= 3:
                        f.write(f"{ann.class_id} " + " ".join(f"{p[0]:.6f} {p[1]:.6f}" for p in pts) + "\n")
                elif ann.shape_type == "point":
                    pts = json.loads(ann.points_json or "[]")
                    if pts:
                        x, y = pts[0]
                        f.write(f"{ann.class_id} {x:.6f} {y:.6f} 0.008 0.008\n")
                else:
                    f.write(f"{ann.class_id} {ann.x_center:.6f} {ann.y_center:.6f} "
                            f"{ann.width:.6f} {ann.height:.6f}\n")

    classes   = project.classes
    yaml_path = os.path.join(dataset_dir, f"{project.name.replace(' ','_')}.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {dataset_dir}\ntrain: images/train\nval: images/val\n\n")
        f.write(f"nc: {len(classes)}\nnames: {json.dumps(classes)}\n")

    return yaml_path


# ─── Training thread ──────────────────────────────────────────────────────────
def _run_training(run_id: int, project_id: int, config: TrainConfig):
    from sqlmodel import create_engine, Session as S
    from ultralytics import YOLO

    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    _state[run_id] = {"logs": [], "done": False, "epoch": 0,
                      "total_epochs": config.epochs, "map50": 0.0}

    def push(msg: str):
        _state[run_id]["logs"].append(msg)

    try:
        with S(engine) as session:
            push("⏳ Building dataset…")
            yaml_path = _build_dataset(project_id, config.val_split, session)
            push(f"✅ Dataset ready ({yaml_path})")
            run = session.get(TrainingRun, run_id)
            run.status = "running"
            run.aug_config_json = json.dumps({
                "fliplr": config.fliplr, "flipud": config.flipud,
                "degrees": config.degrees, "translate": config.translate,
                "scale": config.scale, "hsv_h": config.hsv_h,
                "hsv_s": config.hsv_s, "hsv_v": config.hsv_v,
                "mosaic": config.mosaic, "mixup": config.mixup,
            })
            session.add(run); session.commit()

        import torch
        device = "0" if torch.cuda.is_available() else "cpu"
        push(f"📦 Loading base model: {config.model_base}  |  device: {'GPU (CUDA)' if device == '0' else 'CPU'}")

        if config.resume_run_id:
            with S(engine) as sess:
                prev = sess.get(TrainingRun, config.resume_run_id)
            if prev and prev.run_dir:
                last_pt = os.path.join(prev.run_dir, "weights", "last.pt")
                if not os.path.exists(last_pt):
                    last_pt = os.path.join(prev.run_dir, "last.pt")
                if os.path.exists(last_pt):
                    push(f"📦 Resuming from Run #{config.resume_run_id} ({last_pt})")
                    model = YOLO(last_pt)
                else:
                    push(f"⚠️ last.pt not found for run #{config.resume_run_id}, using base model")
                    model = YOLO(config.model_base)
            else:
                push(f"⚠️ Run #{config.resume_run_id} not found, using base model")
                model = YOLO(config.model_base)
        else:
            model = YOLO(config.model_base)

        def on_epoch_end(trainer):
            ep  = trainer.epoch + 1
            tot = trainer.epochs
            try:
                m50  = float((trainer.metrics or {}).get("metrics/mAP50(B)", 0))
                prec = float((trainer.metrics or {}).get("metrics/precision(B)", 0))
                rec  = float((trainer.metrics or {}).get("metrics/recall(B)", 0))
            except Exception:
                m50 = prec = rec = 0.0
            _state[run_id]["epoch"] = ep
            _state[run_id]["map50"] = m50
            push(f"__PROGRESS__:{ep}/{tot}:{m50:.4f}:{prec:.4f}:{rec:.4f}")

        model.add_callback("on_train_epoch_end", on_epoch_end)

        output_dir = os.path.join(RUNS_DIR, f"train_{run_id}")
        os.makedirs(output_dir, exist_ok=True)

        push(f"🚀 Training started — {config.epochs} epochs, imgsz={config.imgsz}, batch={config.batch}")
        results = model.train(
            data=yaml_path, epochs=config.epochs, imgsz=config.imgsz,
            batch=config.batch, project=output_dir, name="weights",
            exist_ok=True, verbose=False, device=device,
            fliplr=config.fliplr, flipud=config.flipud, degrees=config.degrees,
            translate=config.translate, scale=config.scale,
            hsv_h=config.hsv_h, hsv_s=config.hsv_s, hsv_v=config.hsv_v,
            mosaic=config.mosaic, mixup=config.mixup,
        )

        # ── Locate artifacts ──────────────────────────────────────────────
        run_dir  = os.path.join(output_dir, "weights")
        best_pt  = os.path.join(run_dir, "weights", "best.pt")
        if not os.path.exists(best_pt):
            best_pt = os.path.join(run_dir, "best.pt")

        # ── Overall metrics ───────────────────────────────────────────────
        metrics = {}
        try:
            rd = results.results_dict or {}
            metrics = {
                "mAP50":     float(rd.get("metrics/mAP50(B)", 0)),
                "mAP50_95":  float(rd.get("metrics/mAP50-95(B)", 0)),
                "precision": float(rd.get("metrics/precision(B)", 0)),
                "recall":    float(rd.get("metrics/recall(B)", 0)),
            }
        except Exception:
            pass

        # ── Per-class metrics ─────────────────────────────────────────────
        try:
            box = results.box
            names = results.names or {}
            per_class = {}
            for i, cls_idx in enumerate(box.ap_class_index):
                cls_name = names.get(int(cls_idx), f"cls{cls_idx}")
                per_class[cls_name] = {
                    "ap50":      round(float(box.ap50[i]),  4),
                    "precision": round(float(box.p[i]),     4) if hasattr(box, 'p') else 0,
                    "recall":    round(float(box.r[i]),     4) if hasattr(box, 'r') else 0,
                }
            metrics["per_class"] = per_class
        except Exception:
            pass

        with S(engine) as session:
            run = session.get(TrainingRun, run_id)
            run.status       = "done"
            run.model_path   = best_pt if os.path.exists(best_pt) else ""
            run.run_dir      = run_dir
            run.results_json = json.dumps(metrics)
            session.add(run); session.commit()

        push(f"✅ Training complete! mAP50={metrics.get('mAP50', 0):.4f}  "
             f"precision={metrics.get('precision', 0):.4f}  recall={metrics.get('recall', 0):.4f}")
        push(f"__DONE__:{json.dumps(metrics)}")

    except Exception as exc:
        push(f"❌ Error: {exc}")
        push("__FAILED__")
        try:
            with S(engine) as session:
                run = session.get(TrainingRun, run_id)
                if run:
                    run.status = "failed"
                    session.add(run); session.commit()
        except Exception:
            pass
    finally:
        _state[run_id]["done"] = True


# ─── Routes ───────────────────────────────────────────────────────────────────
@router.post("/start", response_model=TrainingRunOut)
def start_training(project_id: int, config: TrainConfig, session: Session = Depends(get_session)):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    images    = session.exec(select(Image).where(Image.project_id == project_id)).all()
    annotated = [img for img in images
                 if session.exec(select(Annotation).where(Annotation.image_id == img.id)).first()]
    if len(annotated) < 2:
        raise HTTPException(400, "Need at least 2 annotated images to train")

    run = TrainingRun(
        project_id=project_id, status="pending",
        epochs=config.epochs, imgsz=config.imgsz, batch=config.batch,
        model_base=config.model_base, created_at=datetime.now().isoformat(),
    )
    session.add(run); session.commit(); session.refresh(run)

    threading.Thread(target=_run_training, args=(run.id, project_id, config), daemon=True).start()

    return _run_to_out(run)


@router.get("/runs", response_model=list[TrainingRunOut])
def list_runs(project_id: int, session: Session = Depends(get_session)):
    runs = session.exec(select(TrainingRun).where(TrainingRun.project_id == project_id)).all()
    return [_run_to_out(r) for r in runs]


@router.get("/runs/{run_id}/logs")
def stream_logs(project_id: int, run_id: int, session: Session = Depends(get_session)):
    run = session.get(TrainingRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(404, "Run not found")

    def event_stream():
        sent = 0
        for _ in range(30):
            if run_id in _state:
                break
            time.sleep(0.5)
        else:
            yield f"data: Run status: {run.status}\n\n"
            yield "data: __END__\n\n"
            return

        deadline = time.time() + 4 * 3600
        while time.time() < deadline:
            st   = _state[run_id]
            logs = st["logs"]
            while sent < len(logs):
                yield f"data: {logs[sent]}\n\n"
                sent += 1
            if st["done"]:
                yield "data: __END__\n\n"
                _state.pop(run_id, None)
                return
            time.sleep(0.5)

        yield "data: ⚠️ Stream timeout (4 h limit reached)\n\n"
        yield "data: __END__\n\n"
        _state.pop(run_id, None)

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/runs/{run_id}/download")
def download_model(project_id: int, run_id: int, session: Session = Depends(get_session)):
    run = session.get(TrainingRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(404, "Run not found")
    if not run.model_path or not os.path.exists(run.model_path):
        raise HTTPException(404, "Model file not found")
    return FileResponse(run.model_path, filename=f"model_run{run_id}.pt",
                        media_type="application/octet-stream")


def _run_to_out(r: TrainingRun) -> TrainingRunOut:
    return TrainingRunOut(
        id=r.id, project_id=r.project_id, status=r.status,
        epochs=r.epochs, imgsz=r.imgsz, batch=r.batch,
        model_base=r.model_base, model_path=r.model_path,
        results=json.loads(r.results_json), created_at=r.created_at,
        run_dir=r.run_dir or "", onnx_path=r.onnx_path or "",
    )
