from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from sqlmodel import Session, select
from pydantic import BaseModel
from typing import Optional, List
import os, json, shutil, threading, time, uuid, random, io, base64
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
    model_base: str = "yolo11n.pt"
    val_split: float = 0.2
    # Optimizer
    optimizer: str = "auto"       # auto | SGD | Adam | AdamW | NAdam | RAdam | RMSProp
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    patience: int = 50            # early stopping patience (0 = off)
    # Geometric augmentation
    fliplr: float = 0.5
    flipud: float = 0.0
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    # Color augmentation
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    # Mixing / cutout
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0
    erasing: float = 0.4
    resume_run_id: Optional[int] = None


class TrainingRunOut(BaseModel):
    id: int; project_id: int; status: str; epochs: int; imgsz: int
    batch: int; model_base: str; model_path: str; results: dict
    created_at: str; run_dir: str = ""; onnx_path: str = ""

    class Config:
        from_attributes = True


# ─── Dataset builder ─────────────────────────────────────────────────────────
def _build_dataset(project_id: int, val_split: float, session: Session,
                   task: str = "detect") -> str:
    """Build a YOLO dataset directory.

    task='detect'  → labels use  <cls> cx cy w h
    task='segment' → labels use  <cls> x1 y1 x2 y2 … (polygon, ≥3 pts)
                     bbox annotations are converted to 4-corner polygons.
    """
    project     = session.get(Project, project_id)
    dataset_dir = os.path.join(RUNS_DIR, f"dataset_{project_id}_{uuid.uuid4().hex[:6]}")

    for split in ["train", "val"]:
        os.makedirs(os.path.join(dataset_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, "labels", split), exist_ok=True)

    all_images = session.exec(select(Image).where(Image.project_id == project_id)).all()
    images = [img for img in all_images
              if session.exec(select(Annotation).where(Annotation.image_id == img.id)).first()]

    shuffled = list(images)
    random.shuffle(shuffled)
    n_val   = max(1, int(len(shuffled) * val_split))
    val_ids = {img.id for img in shuffled[:n_val]}

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
                        f.write(f"{ann.class_id} " +
                                " ".join(f"{p[0]:.6f} {p[1]:.6f}" for p in pts) + "\n")
                elif ann.shape_type == "point":
                    pts = json.loads(ann.points_json or "[]")
                    if pts:
                        x, y = pts[0]
                        f.write(f"{ann.class_id} {x:.6f} {y:.6f} 0.008 0.008\n")
                else:
                    # bbox annotation
                    cx, cy, w, h = ann.x_center, ann.y_center, ann.width, ann.height
                    if task == "segment":
                        # Convert bbox to 4-corner polygon for YOLO-seg
                        x1, y1 = cx - w / 2, cy - h / 2
                        x2, y2 = cx + w / 2, cy - h / 2
                        x3, y3 = cx + w / 2, cy + h / 2
                        x4, y4 = cx - w / 2, cy + h / 2
                        f.write(f"{ann.class_id} "
                                f"{x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} "
                                f"{x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}\n")
                    else:
                        f.write(f"{ann.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    classes   = project.classes
    yaml_path = os.path.join(dataset_dir, f"{project.name.replace(' ','_')}.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {dataset_dir}\ntrain: images/train\nval: images/val\n\n")
        f.write(f"nc: {len(classes)}\nnames: {json.dumps(classes)}\n")
        if task == "segment":
            f.write("task: segment\n")

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
        task = "segment" if "seg" in config.model_base else "detect"
        with S(engine) as session:
            push("Building dataset…")
            yaml_path = _build_dataset(project_id, config.val_split, session, task=task)
            push(f"Dataset ready — {yaml_path}")
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
        # Verify CUDA actually works — torch.cuda.is_available() can return True
        # even when the installed PyTorch CUDA version doesn't match the driver.
        if device == "0":
            try:
                torch.zeros(1).cuda()
            except RuntimeError:
                device = "cpu"
                push("[WARN] CUDA detected but kernel incompatible — falling back to CPU")
        push(f"Loading base model: {config.model_base}  |  device: {'GPU (CUDA)' if device == '0' else 'CPU'}")

        if config.resume_run_id:
            with S(engine) as sess:
                prev = sess.get(TrainingRun, config.resume_run_id)
            if prev and prev.run_dir:
                last_pt = os.path.join(prev.run_dir, "weights", "last.pt")
                if not os.path.exists(last_pt):
                    last_pt = os.path.join(prev.run_dir, "last.pt")
                if os.path.exists(last_pt):
                    push(f"Resuming from Run #{config.resume_run_id} ({last_pt})")
                    model = YOLO(last_pt)
                else:
                    push(f"[WARN] last.pt not found for run #{config.resume_run_id}, using base model")
                    model = YOLO(config.model_base)
            else:
                push(f"[WARN] Run #{config.resume_run_id} not found, using base model")
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
            # Honour stop requests between epochs
            if _state[run_id].get("stop_requested"):
                trainer.stop = True

        model.add_callback("on_train_epoch_end", on_epoch_end)

        output_dir = os.path.join(RUNS_DIR, f"train_{run_id}")
        os.makedirs(output_dir, exist_ok=True)

        push(f"Training started — {config.epochs} epochs  imgsz={config.imgsz}  "
             f"batch={config.batch}  optimizer={config.optimizer}  lr0={config.lr0}")
        results = model.train(
            data=yaml_path, epochs=config.epochs, imgsz=config.imgsz,
            batch=config.batch, project=output_dir, name="weights",
            exist_ok=True, verbose=False, device=device,
            # Optimizer
            optimizer=config.optimizer, lr0=config.lr0, lrf=config.lrf,
            momentum=config.momentum, weight_decay=config.weight_decay,
            warmup_epochs=config.warmup_epochs,
            patience=config.patience,
            # Geometric augmentation
            fliplr=config.fliplr, flipud=config.flipud, degrees=config.degrees,
            translate=config.translate, scale=config.scale,
            shear=config.shear, perspective=config.perspective,
            # Color augmentation
            hsv_h=config.hsv_h, hsv_s=config.hsv_s, hsv_v=config.hsv_v,
            # Mixing / cutout
            mosaic=config.mosaic, mixup=config.mixup,
            copy_paste=config.copy_paste, erasing=config.erasing,
        )

        # ── Locate artifacts ──────────────────────────────────────────────
        # model.train(project=output_dir, name="weights") saves to output_dir/weights/
        run_dir = os.path.join(output_dir, "weights")
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

        push(f"[DONE] Training complete  mAP50={metrics.get('mAP50', 0):.4f}  "
             f"precision={metrics.get('precision', 0):.4f}  recall={metrics.get('recall', 0):.4f}")
        push(f"__DONE__:{json.dumps(metrics)}")

    except Exception as exc:
        push(f"[ERROR] {exc}")
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

        yield "data: [WARN] Stream timeout (4 h limit reached)\n\n"
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


@router.post("/runs/{run_id}/stop")
def stop_run(project_id: int, run_id: int, session: Session = Depends(get_session)):
    run = session.get(TrainingRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(404, "Run not found")
    if run_id in _state:
        _state[run_id]["stop_requested"] = True
    # Also mark as failed immediately so UI updates even before the epoch ends
    if run.status == "running":
        run.status = "stopped"
        session.add(run)
        session.commit()
    return {"ok": True}


@router.delete("/runs/{run_id}")
def delete_run(project_id: int, run_id: int, session: Session = Depends(get_session)):
    run = session.get(TrainingRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(404, "Run not found")
    if run.status == "running":
        raise HTTPException(400, "Stop the run before deleting it")
    # Remove run directory from disk
    output_dir = os.path.join(RUNS_DIR, f"train_{run_id}")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
    session.delete(run)
    session.commit()
    return {"ok": True}


# ─── Augmentation preview ────────────────────────────────────────────────────
class AugPreviewBody(BaseModel):
    fliplr:     float = 0.5
    flipud:     float = 0.0
    degrees:    float = 0.0
    translate:  float = 0.1
    scale:      float = 0.5
    hsv_h:      float = 0.015
    hsv_s:      float = 0.7
    hsv_v:      float = 0.4
    mosaic:     float = 1.0
    n:          int   = 6


def _apply_aug_transforms(img_path: str, body: AugPreviewBody):
    """Apply approximate YOLO augmentations using torchvision + PIL."""
    import torchvision.transforms as T
    from PIL import Image as PILImage

    img = PILImage.open(img_path).convert("RGB")
    w, h = img.size
    augs = []

    if body.fliplr > 0:
        augs.append(T.RandomHorizontalFlip(p=body.fliplr))
    if body.flipud > 0:
        augs.append(T.RandomVerticalFlip(p=body.flipud))

    affine_kw: dict = {"degrees": body.degrees if body.degrees > 0 else 0}
    if body.translate > 0:
        affine_kw["translate"] = (body.translate, body.translate)
    if body.scale > 0:
        affine_kw["scale"] = (max(0.1, 1 - body.scale), 1 + body.scale)
    if body.degrees > 0 or body.translate > 0 or body.scale > 0:
        affine_kw["fill"] = [114, 114, 114]
        augs.append(T.RandomAffine(**affine_kw))

    jitter_kw: dict = {}
    if body.hsv_v > 0:
        jitter_kw["brightness"] = min(body.hsv_v, 0.8)
        jitter_kw["contrast"]   = min(body.hsv_v * 0.5, 0.5)
    if body.hsv_s > 0:
        jitter_kw["saturation"] = min(body.hsv_s * 0.6, 0.8)
    if body.hsv_h > 0:
        jitter_kw["hue"] = min(body.hsv_h * 3, 0.5)
    if jitter_kw:
        augs.append(T.ColorJitter(**jitter_kw))

    if augs:
        img = T.Compose(augs)(img)

    # Simulate mosaic: paste 4 tiny copies into one image
    if body.mosaic > 0 and random.random() < body.mosaic * 0.6:
        half = (w // 2, h // 2)
        canvas = PILImage.new("RGB", (w, h), (114, 114, 114))
        for row in range(2):
            for col in range(2):
                tile = img.resize(half, PILImage.LANCZOS)
                canvas.paste(tile, (col * half[0], row * half[1]))
        img = canvas

    img.thumbnail((400, 400), PILImage.LANCZOS)
    return img


@router.post("/augmentation-preview")
def augmentation_preview(
    project_id: int,
    body: AugPreviewBody,
    session: Session = Depends(get_session),
):
    """Return N base64-encoded preview thumbnails with augmentation applied."""
    all_images = session.exec(select(Image).where(Image.project_id == project_id)).all()
    annotated  = [img for img in all_images
                  if session.exec(select(Annotation).where(Annotation.image_id == img.id)).first()]
    if not annotated:
        raise HTTPException(400, "No annotated images in this project")

    samples  = random.choices(annotated, k=min(body.n, len(annotated)))
    previews: List[str] = []

    for img_rec in samples:
        img_path = os.path.join(UPLOAD_DIR, img_rec.filename)
        if not os.path.exists(img_path):
            continue
        try:
            pil_img = _apply_aug_transforms(img_path, body)
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=82)
            previews.append("data:image/jpeg;base64," +
                            base64.b64encode(buf.getvalue()).decode())
        except Exception:
            pass

    return {"previews": previews}


def _run_to_out(r: TrainingRun) -> TrainingRunOut:
    return TrainingRunOut(
        id=r.id, project_id=r.project_id, status=r.status,
        epochs=r.epochs, imgsz=r.imgsz, batch=r.batch,
        model_base=r.model_base, model_path=r.model_path,
        results=json.loads(r.results_json), created_at=r.created_at,
        run_dir=r.run_dir or "", onnx_path=r.onnx_path or "",
    )
