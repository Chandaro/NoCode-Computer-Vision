from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from sqlmodel import Session, select
from pydantic import BaseModel
from typing import Optional
import os, json, shutil, threading, time, random
from datetime import datetime

from database import get_session, DATABASE_URL
from models import ClassificationRun, Project, Image, Annotation

router = APIRouter(prefix="/projects/{project_id}/classification", tags=["classification"])

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")
RUNS_DIR   = os.path.join(os.path.dirname(__file__), "..", "runs")
os.makedirs(RUNS_DIR, exist_ok=True)

_cls_state: dict = {}

BASE_MODELS = [
    "resnet18", "resnet34", "resnet50",
    "mobilenet_v3_small",
    "efficientnet_b0", "efficientnet_b1",
    "convnext_tiny",
]


class ClsConfig(BaseModel):
    epochs: int = 10
    imgsz: int = 224
    batch: int = 32
    base_model: str = "resnet18"
    lr: float = 0.001
    freeze_backbone: bool = True
    val_split: float = 0.2
    patience: int = 0               # early stop; 0 = disabled
    resume_run_id: Optional[int] = None
    # Optimizer
    optimizer: str = "Adam"         # Adam | AdamW | SGD
    weight_decay: float = 0.0
    momentum: float = 0.9           # for SGD
    warmup_epochs: int = 0
    # LR scheduler
    lr_scheduler: str = "cosine"    # cosine | step | none
    step_size: int = 10             # for StepLR
    step_gamma: float = 0.1         # for StepLR
    # Regularisation
    label_smoothing: float = 0.0    # 0..0.3
    dropout_head: float = 0.0       # dropout before classifier head
    # Augmentation
    fliplr: float = 0.5
    flipud: float = 0.0
    degrees: float = 0.0
    translate: float = 0.0
    scale: float = 0.0
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    erasing: float = 0.0
    mixup: float = 0.0


class ClsRunOut(BaseModel):
    id: int; project_id: int; status: str; epochs: int; imgsz: int
    batch: int; base_model: str; lr: float; freeze_backbone: bool
    model_path: str; results: dict; created_at: str

    class Config:
        from_attributes = True


def _run_to_out(r: ClassificationRun) -> ClsRunOut:
    return ClsRunOut(
        id=r.id, project_id=r.project_id, status=r.status,
        epochs=r.epochs, imgsz=r.imgsz, batch=r.batch,
        base_model=r.base_model, lr=r.lr, freeze_backbone=r.freeze_backbone,
        model_path=r.model_path, results=json.loads(r.results_json),
        created_at=r.created_at,
    )


# ─── Model builder helper ─────────────────────────────────────────────────────
def _build_model(base_model: str, n_classes: int, dropout_head: float = 0.0):
    import torch.nn as nn
    import torchvision.models as M

    weights_map = {
        "resnet18":           M.ResNet18_Weights.DEFAULT,
        "resnet34":           M.ResNet34_Weights.DEFAULT,
        "resnet50":           M.ResNet50_Weights.DEFAULT,
        "mobilenet_v3_small": M.MobileNet_V3_Small_Weights.DEFAULT,
        "efficientnet_b0":    M.EfficientNet_B0_Weights.DEFAULT,
        "efficientnet_b1":    M.EfficientNet_B1_Weights.DEFAULT,
        "convnext_tiny":      M.ConvNeXt_Tiny_Weights.DEFAULT,
    }
    model_fn_map = {
        "resnet18":           M.resnet18,
        "resnet34":           M.resnet34,
        "resnet50":           M.resnet50,
        "mobilenet_v3_small": M.mobilenet_v3_small,
        "efficientnet_b0":    M.efficientnet_b0,
        "efficientnet_b1":    M.efficientnet_b1,
        "convnext_tiny":      M.convnext_tiny,
    }
    weights   = weights_map.get(base_model, M.ResNet18_Weights.DEFAULT)
    model_fn  = model_fn_map.get(base_model, M.resnet18)
    model     = model_fn(weights=weights)

    # Replace classifier head (with optional dropout)
    if hasattr(model, "fc"):
        in_f = model.fc.in_features
        model.fc = (
            nn.Sequential(nn.Dropout(dropout_head), nn.Linear(in_f, n_classes))
            if dropout_head > 0 else nn.Linear(in_f, n_classes)
        )
    elif hasattr(model, "classifier"):
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = (
            nn.Sequential(nn.Dropout(dropout_head), nn.Linear(in_f, n_classes))
            if dropout_head > 0 else nn.Linear(in_f, n_classes)
        )

    return model


# ─── Dataset builder ─────────────────────────────────────────────────────────
def _build_cls_dataset(project_id: int, run_id: int, classes: list,
                       val_split: float, session: Session) -> tuple:
    """Build ImageFolder layout: run_dir/train/<class_name>/img.jpg."""
    dataset_dir = os.path.join(RUNS_DIR, f"cls_dataset_{project_id}_{run_id}")
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)

    images = list(session.exec(select(Image).where(Image.project_id == project_id)).all())
    random.shuffle(images)

    for split in ["train", "val"]:
        for cls in classes:
            os.makedirs(os.path.join(dataset_dir, split, cls), exist_ok=True)

    n_val    = max(1, int(len(images) * val_split))
    val_ids  = {img.id for img in images[:n_val]}
    skipped  = 0

    for img in images:
        anns = session.exec(select(Annotation).where(Annotation.image_id == img.id)).all()
        if not anns:
            skipped += 1; continue
        cls_id = anns[0].class_id
        if cls_id >= len(classes):
            skipped += 1; continue
        cls_name = classes[cls_id]
        split    = "val" if img.id in val_ids else "train"
        src      = os.path.join(UPLOAD_DIR, img.filename)
        if not os.path.exists(src):
            skipped += 1; continue
        shutil.copy2(src, os.path.join(dataset_dir, split, cls_name, img.filename))

    return dataset_dir, skipped


# ─── Training thread ──────────────────────────────────────────────────────────
def _run_classification(run_id: int, project_id: int, config: ClsConfig):
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    import torchvision.datasets as D
    from torch.utils.data import DataLoader
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, StepLR
    from sqlmodel import create_engine, Session as S

    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    _cls_state[run_id] = {"logs": [], "done": False, "stop_requested": False}

    def push(msg: str):
        _cls_state[run_id]["logs"].append(msg)

    try:
        with S(engine) as session:
            project = session.get(Project, project_id)
            classes = project.classes
            run     = session.get(ClassificationRun, run_id)
            run.status = "running"
            session.add(run); session.commit()

            push("Building classification dataset…")
            dataset_dir, skipped = _build_cls_dataset(
                project_id, run_id, classes, config.val_split, session)
            push(f"Dataset ready — {skipped} images skipped (no annotation)")

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            try:
                torch.zeros(1).cuda()
            except RuntimeError:
                use_cuda = False
                push("[WARN] CUDA detected but kernel incompatible — falling back to CPU")
        device = torch.device("cuda" if use_cuda else "cpu")
        push(f"Loading base model: {config.base_model}  |  device: {device}")

        # ── Augmentation transforms ────────────────────────────────────────
        aug = [T.Resize((config.imgsz, config.imgsz))]
        if config.fliplr > 0:
            aug.append(T.RandomHorizontalFlip(p=config.fliplr))
        if config.flipud > 0:
            aug.append(T.RandomVerticalFlip(p=config.flipud))
        affine_kw: dict = {}
        if config.degrees > 0:
            affine_kw["degrees"] = config.degrees
        if config.translate > 0:
            affine_kw["translate"] = (config.translate, config.translate)
        if config.scale > 0:
            affine_kw["scale"] = (1 - config.scale, 1 + config.scale)
        if affine_kw:
            affine_kw.setdefault("degrees", 0)
            aug.append(T.RandomAffine(**affine_kw))
        if config.brightness > 0 or config.contrast > 0 or config.saturation > 0:
            aug.append(T.ColorJitter(
                brightness=config.brightness,
                contrast=config.contrast,
                saturation=config.saturation,
            ))
        aug.extend([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        if config.erasing > 0:
            aug.append(T.RandomErasing(p=config.erasing))

        train_tf = T.Compose(aug)
        val_tf   = T.Compose([
            T.Resize((config.imgsz, config.imgsz)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        train_ds = D.ImageFolder(os.path.join(dataset_dir, "train"), transform=train_tf)
        val_ds   = D.ImageFolder(os.path.join(dataset_dir, "val"),   transform=val_tf)
        train_dl = DataLoader(train_ds, batch_size=config.batch, shuffle=True,  num_workers=0)
        val_dl   = DataLoader(val_ds,   batch_size=config.batch, shuffle=False, num_workers=0)

        n_classes = len(classes)
        push(f"Train: {len(train_ds)} images  |  Val: {len(val_ds)} images  |  Classes: {n_classes}")

        # ── Build model ────────────────────────────────────────────────────
        model = _build_model(config.base_model, n_classes, config.dropout_head)

        if config.freeze_backbone:
            for name, param in model.named_parameters():
                if "fc" not in name and "classifier" not in name:
                    param.requires_grad = False

        model = model.to(device)

        # ── Resume from previous run ───────────────────────────────────────
        if config.resume_run_id:
            with S(engine) as session:
                prev = session.get(ClassificationRun, config.resume_run_id)
                if prev and prev.model_path and os.path.exists(prev.model_path):
                    model.load_state_dict(torch.load(
                        prev.model_path, map_location=device, weights_only=False))
                    push(f"Resumed weights from Run #{config.resume_run_id}")
                else:
                    push("[WARN] Resume run not found — starting from scratch")

        # ── Optimizer ─────────────────────────────────────────────────────
        trainable = list(filter(lambda p: p.requires_grad, model.parameters()))
        opt_name  = config.optimizer.lower()
        if opt_name == "sgd":
            optim = torch.optim.SGD(
                trainable, lr=config.lr,
                momentum=config.momentum, weight_decay=config.weight_decay)
        elif opt_name == "adamw":
            optim = torch.optim.AdamW(
                trainable, lr=config.lr, weight_decay=config.weight_decay)
        else:
            optim = torch.optim.Adam(
                trainable, lr=config.lr, weight_decay=config.weight_decay)

        # ── LR scheduler ──────────────────────────────────────────────────
        total_epochs = config.epochs
        wu = max(0, min(config.warmup_epochs, total_epochs - 1))
        sched_name = config.lr_scheduler.lower()

        if sched_name == "step":
            base_sched = StepLR(optim, step_size=max(1, config.step_size),
                                gamma=config.step_gamma)
        elif sched_name == "none":
            base_sched = None
        else:  # cosine (default)
            base_sched = CosineAnnealingLR(
                optim, T_max=max(1, total_epochs - wu), eta_min=config.lr * 0.01)

        if wu > 0 and base_sched is not None:
            warmup_sched = LinearLR(optim, start_factor=0.01, total_iters=wu)
            scheduler    = SequentialLR(optim, schedulers=[warmup_sched, base_sched],
                                        milestones=[wu])
        else:
            scheduler = base_sched

        # ── Loss ──────────────────────────────────────────────────────────
        criterion = nn.CrossEntropyLoss(
            label_smoothing=max(0.0, min(0.5, config.label_smoothing)))

        run_dir   = os.path.join(RUNS_DIR, f"cls_train_{run_id}")
        os.makedirs(run_dir, exist_ok=True)
        best_acc  = 0.0
        best_path = os.path.join(run_dir, "best.pth")
        patience_counter = 0

        push(f"Training — {config.epochs} ep  {config.optimizer}  lr={config.lr}  "
             f"sched={config.lr_scheduler}  wd={config.weight_decay}  "
             f"smooth={config.label_smoothing:.2f}"
             + (f"  warmup={wu} ep" if wu > 0 else "")
             + (f" | early-stop patience={config.patience}" if config.patience > 0 else ""))

        for epoch in range(1, config.epochs + 1):
            # ── Train ──────────────────────────────────────────────────────
            model.train()
            train_loss = 0.0
            for imgs, labels in train_dl:
                imgs, labels = imgs.to(device), labels.to(device)

                # Mixup
                if config.mixup > 0 and torch.rand(1).item() < config.mixup:
                    lam    = float(torch.distributions.Beta(
                        torch.tensor(0.4), torch.tensor(0.4)).sample())
                    idx    = torch.randperm(imgs.size(0), device=device)
                    mixed  = lam * imgs + (1 - lam) * imgs[idx]
                    loss   = lam * criterion(model(mixed), labels) + \
                             (1 - lam) * criterion(model(mixed), labels[idx])
                    optim.zero_grad(); loss.backward(); optim.step()
                else:
                    optim.zero_grad()
                    loss = criterion(model(imgs), labels)
                    loss.backward(); optim.step()
                train_loss += loss.item()

            if scheduler is not None:
                scheduler.step()
            current_lr = optim.param_groups[0]["lr"]

            # ── Validate ───────────────────────────────────────────────────
            model.eval()
            correct = total = 0
            val_loss = 0.0
            with torch.no_grad():
                for imgs, labels in val_dl:
                    imgs, labels = imgs.to(device), labels.to(device)
                    out       = model(imgs)
                    val_loss += criterion(out, labels).item()
                    correct  += (out.argmax(dim=1) == labels).sum().item()
                    total    += labels.size(0)

            acc   = correct / max(total, 1)
            vloss = val_loss / max(len(val_dl), 1)
            tloss = train_loss / max(len(train_dl), 1)

            if acc > best_acc:
                best_acc = acc
                patience_counter = 0
                torch.save(model.state_dict(), best_path)
            else:
                patience_counter += 1

            # __PROGRESS__ format: epoch/total:acc:train_loss:val_loss
            push(f"__PROGRESS__:{epoch}/{config.epochs}:{acc:.4f}:{tloss:.4f}:{vloss:.4f}")
            push(f"  Epoch {epoch}/{config.epochs} — "
                 f"val_acc={acc*100:.1f}%  val_loss={vloss:.4f}  "
                 f"train_loss={tloss:.4f}  lr={current_lr:.6f}")

            if _cls_state[run_id].get("stop_requested"):
                push("🛑 Stop requested — saving best weights and exiting early")
                break

            if config.patience > 0 and patience_counter >= config.patience:
                push(f"🛑 Early stopping at epoch {epoch} "
                     f"(no improvement for {config.patience} epochs)")
                break

        # ── Final evaluation (per-class, top-5, confusion matrix, F1) ────
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, weights_only=False))
        model.eval()
        class_correct = [0] * n_classes
        class_total   = [0] * n_classes
        all_true: list = []
        all_pred: list = []
        top5_correct = top5_total = 0
        k = min(5, n_classes)

        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                out   = model(imgs)
                preds = out.argmax(dim=1)
                top_k = out.topk(k, dim=1).indices
                for t, p, pk in zip(labels, preds, top_k):
                    ti, pi = int(t), int(p)
                    class_total[ti]  += 1
                    class_correct[ti] += int(ti == pi)
                    all_true.append(ti)
                    all_pred.append(pi)
                    top5_correct += int(t in pk)
                    top5_total   += 1

        # Confusion matrix
        cm = [[0] * n_classes for _ in range(n_classes)]
        for t, p in zip(all_true, all_pred):
            if 0 <= t < n_classes and 0 <= p < n_classes:
                cm[t][p] += 1

        # Per-class precision, recall, F1
        per_class_metrics: dict = {}
        for i in range(n_classes):
            tp = cm[i][i]
            fp = sum(cm[j][i] for j in range(n_classes) if j != i)
            fn = sum(cm[i][j] for j in range(n_classes) if j != i)
            precision = tp / max(tp + fp, 1)
            recall    = tp / max(tp + fn, 1)
            f1        = 2 * precision * recall / max(precision + recall, 1e-9)
            per_class_metrics[classes[i]] = {
                "accuracy":  round(class_correct[i] / max(class_total[i], 1), 4),
                "precision": round(precision, 4),
                "recall":    round(recall, 4),
                "f1":        round(f1, 4),
                "support":   class_total[i],
            }

        metrics = {
            "top1_acc":         round(best_acc, 4),
            "top5_acc":         round(top5_correct / max(top5_total, 1), 4),
            "per_class":        per_class_metrics,
            "confusion_matrix": cm,
        }

        with S(engine) as session:
            run = session.get(ClassificationRun, run_id)
            run.status       = "done"
            run.model_path   = best_path
            run.run_dir      = run_dir
            run.results_json = json.dumps(metrics)
            session.add(run); session.commit()

        push(f"[DONE] Best val accuracy: {best_acc*100:.1f}%")
        push(f"__DONE__:{json.dumps(metrics)}")

    except Exception as exc:
        push(f"[ERROR] {exc}")
        push("__FAILED__")
        try:
            with S(engine) as session:
                run = session.get(ClassificationRun, run_id)
                if run:
                    run.status = "failed"
                    session.add(run); session.commit()
        except Exception:
            pass
    finally:
        _cls_state[run_id]["done"] = True


# ─── Routes ───────────────────────────────────────────────────────────────────
@router.post("/start", response_model=ClsRunOut)
def start_classification(project_id: int, config: ClsConfig,
                         session: Session = Depends(get_session)):
    if config.base_model not in BASE_MODELS:
        raise HTTPException(400, f"base_model must be one of {BASE_MODELS}")
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    if len(project.classes) < 2:
        raise HTTPException(400, "Need at least 2 classes for classification")

    images    = session.exec(select(Image).where(Image.project_id == project_id)).all()
    annotated = [img for img in images
                 if session.exec(select(Annotation).where(Annotation.image_id == img.id)).first()]
    if len(annotated) < 4:
        raise HTTPException(400, "Need at least 4 annotated images (2 per class minimum)")

    run = ClassificationRun(
        project_id=project_id, status="pending",
        epochs=config.epochs, imgsz=config.imgsz, batch=config.batch,
        base_model=config.base_model, lr=config.lr, freeze_backbone=config.freeze_backbone,
        created_at=datetime.now().isoformat(),
    )
    session.add(run); session.commit(); session.refresh(run)

    threading.Thread(
        target=_run_classification, args=(run.id, project_id, config), daemon=True
    ).start()

    return _run_to_out(run)


@router.get("/runs", response_model=list[ClsRunOut])
def list_cls_runs(project_id: int, session: Session = Depends(get_session)):
    runs = session.exec(
        select(ClassificationRun).where(ClassificationRun.project_id == project_id)
    ).all()
    return [_run_to_out(r) for r in runs]


@router.get("/runs/{run_id}/logs")
def stream_cls_logs(project_id: int, run_id: int,
                    session: Session = Depends(get_session)):
    run = session.get(ClassificationRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(404, "Run not found")

    def event_stream():
        sent = 0
        for _ in range(30):
            if run_id in _cls_state:
                break
            time.sleep(0.5)
        else:
            yield f"data: Run status: {run.status}\n\n"
            yield "data: __END__\n\n"
            return

        deadline = time.time() + 4 * 3600
        while time.time() < deadline:
            st   = _cls_state[run_id]
            logs = st["logs"]
            while sent < len(logs):
                yield f"data: {logs[sent]}\n\n"
                sent += 1
            if st["done"]:
                yield "data: __END__\n\n"
                _cls_state.pop(run_id, None)
                return
            time.sleep(0.5)

        yield "data: [WARN] Stream timeout\n\n"
        yield "data: __END__\n\n"
        _cls_state.pop(run_id, None)

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/runs/{run_id}/stop")
def stop_cls_run(project_id: int, run_id: int, session: Session = Depends(get_session)):
    run = session.get(ClassificationRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(404, "Run not found")
    st = _cls_state.get(run_id, {})
    if st:
        st["stop_requested"] = True
    if run.status == "running":
        run.status = "stopped"
        session.add(run); session.commit()
    return {"ok": True}


@router.delete("/runs/{run_id}")
def delete_cls_run(project_id: int, run_id: int, session: Session = Depends(get_session)):
    run = session.get(ClassificationRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(404, "Run not found")
    if run.status == "running":
        raise HTTPException(400, "Stop the run before deleting")
    session.delete(run); session.commit()
    return {"ok": True}


@router.get("/runs/{run_id}/download")
def download_cls_model(project_id: int, run_id: int,
                       session: Session = Depends(get_session)):
    run = session.get(ClassificationRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(404, "Run not found")
    if not run.model_path or not os.path.exists(run.model_path):
        raise HTTPException(404, "Model file not found")
    return FileResponse(run.model_path, filename=f"cls_model_run{run_id}.pth",
                        media_type="application/octet-stream")


@router.post("/runs/{run_id}/export-onnx")
def export_cls_onnx(project_id: int, run_id: int,
                    session: Session = Depends(get_session)):
    """Export the best .pth checkpoint to ONNX and return it as a file download."""
    import torch

    run = session.get(ClassificationRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(404, "Run not found")
    if not run.model_path or not os.path.exists(run.model_path):
        raise HTTPException(404, "Model file not found — train the run first")

    project   = session.get(Project, project_id)
    n_classes = len(project.classes)
    onnx_path = run.model_path.replace(".pth", ".onnx")

    if not os.path.exists(onnx_path):
        model = _build_model(run.base_model, n_classes)
        model.load_state_dict(torch.load(run.model_path, map_location="cpu", weights_only=False))
        model.eval()
        dummy = torch.randn(1, 3, run.imgsz, run.imgsz)
        torch.onnx.export(
            model, dummy, onnx_path,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17,
        )

    return FileResponse(onnx_path, filename=f"cls_model_run{run_id}.onnx",
                        media_type="application/octet-stream")
