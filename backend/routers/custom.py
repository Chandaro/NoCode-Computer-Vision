from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from sqlmodel import Session, select
from pydantic import BaseModel
import os, json, shutil, threading, time, random
from datetime import datetime

from database import get_session, DATABASE_URL
from models import CustomModelConfig, CustomTrainingRun, Project, Image, Annotation

router = APIRouter(prefix="/projects/{project_id}/custom", tags=["custom"])

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")
RUNS_DIR   = os.path.join(os.path.dirname(__file__), "..", "runs")
os.makedirs(RUNS_DIR, exist_ok=True)

_custom_state: dict = {}


# ── Pydantic models ────────────────────────────────────────────────────────────

class ConfigBody(BaseModel):
    name: str = "My Model"
    layers: list = []
    input_h: int = 64
    input_w: int = 64


class RunBody(BaseModel):
    config_id: int
    epochs: int = 20
    batch: int = 32
    lr: float = 0.001
    val_split: float = 0.2
    patience: int = 0           # early stop; 0 = disabled
    # Optimizer
    optimizer: str = "Adam"     # Adam | AdamW | SGD
    weight_decay: float = 0.0
    momentum: float = 0.9
    warmup_epochs: int = 0
    # LR Scheduler
    lr_scheduler: str = "cosine"  # cosine | step | none
    step_size: int = 10
    step_gamma: float = 0.1
    # Regularisation
    label_smoothing: float = 0.0
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


class ConfigOut(BaseModel):
    id: int
    project_id: int
    name: str
    layers: list
    input_h: int
    input_w: int
    created_at: str

    class Config:
        from_attributes = True


class RunOut(BaseModel):
    id: int
    config_id: int
    project_id: int
    status: str
    epochs: int
    batch: int
    lr: float
    model_path: str
    results: dict
    created_at: str

    class Config:
        from_attributes = True


def _cfg_to_out(c: CustomModelConfig) -> ConfigOut:
    return ConfigOut(
        id=c.id, project_id=c.project_id, name=c.name,
        layers=json.loads(c.layers_json),
        input_h=c.input_h, input_w=c.input_w,
        created_at=c.created_at,
    )


def _run_to_out(r: CustomTrainingRun) -> RunOut:
    return RunOut(
        id=r.id, config_id=r.config_id, project_id=r.project_id,
        status=r.status, epochs=r.epochs, batch=r.batch, lr=r.lr,
        model_path=r.model_path, results=json.loads(r.results_json),
        created_at=r.created_at,
    )


# ── Config CRUD ────────────────────────────────────────────────────────────────

@router.get("/configs", response_model=list[ConfigOut])
def list_configs(project_id: int, session: Session = Depends(get_session)):
    cfgs = session.exec(
        select(CustomModelConfig).where(CustomModelConfig.project_id == project_id)
    ).all()
    return [_cfg_to_out(c) for c in cfgs]


@router.post("/configs", response_model=ConfigOut)
def create_config(project_id: int, body: ConfigBody,
                  session: Session = Depends(get_session)):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    cfg = CustomModelConfig(
        project_id=project_id,
        name=body.name,
        layers_json=json.dumps(body.layers),
        input_h=body.input_h,
        input_w=body.input_w,
        created_at=datetime.now().isoformat(),
    )
    session.add(cfg); session.commit(); session.refresh(cfg)
    return _cfg_to_out(cfg)


@router.put("/configs/{config_id}", response_model=ConfigOut)
def update_config(project_id: int, config_id: int, body: ConfigBody,
                  session: Session = Depends(get_session)):
    cfg = session.get(CustomModelConfig, config_id)
    if not cfg or cfg.project_id != project_id:
        raise HTTPException(404, "Config not found")
    cfg.name        = body.name
    cfg.layers_json = json.dumps(body.layers)
    cfg.input_h     = body.input_h
    cfg.input_w     = body.input_w
    session.add(cfg); session.commit(); session.refresh(cfg)
    return _cfg_to_out(cfg)


@router.delete("/configs/{config_id}")
def delete_config(project_id: int, config_id: int,
                  session: Session = Depends(get_session)):
    cfg = session.get(CustomModelConfig, config_id)
    if not cfg or cfg.project_id != project_id:
        raise HTTPException(404, "Config not found")
    session.delete(cfg); session.commit()
    return {"ok": True}


# ── Run list ───────────────────────────────────────────────────────────────────

@router.get("/runs", response_model=list[RunOut])
def list_runs(project_id: int, session: Session = Depends(get_session)):
    runs = session.exec(
        select(CustomTrainingRun).where(CustomTrainingRun.project_id == project_id)
    ).all()
    return [_run_to_out(r) for r in runs]


@router.get("/runs/{run_id}", response_model=RunOut)
def get_run(project_id: int, run_id: int, session: Session = Depends(get_session)):
    run = session.get(CustomTrainingRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(404, "Run not found")
    return _run_to_out(run)


# ── Dataset builder ────────────────────────────────────────────────────────────

def _build_custom_dataset(project_id: int, run_id: int, classes: list,
                           input_h: int, input_w: int,
                           val_split: float, session: Session) -> tuple[str, int, dict]:
    """Build ImageFolder layout resized to input_h×input_w.
    Returns (dataset_dir, total_skipped, skip_reason_counts).
    """
    from PIL import Image as PILImage

    dataset_dir = os.path.join(RUNS_DIR, f"custom_dataset_{project_id}_{run_id}")
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)

    images = list(session.exec(select(Image).where(Image.project_id == project_id)).all())
    random.shuffle(images)

    for split in ["train", "val"]:
        for cls in classes:
            safe = cls.replace("/", "_").replace("\\", "_")
            os.makedirs(os.path.join(dataset_dir, split, safe), exist_ok=True)

    n_val = max(1, int(len(images) * val_split))
    val_ids = {img.id for img in images[:n_val]}

    reasons = {"no_annotation": 0, "class_id_out_of_range": 0, "file_not_found": 0, "corrupt": 0}
    placed  = 0

    for img in images:
        anns = session.exec(select(Annotation).where(Annotation.image_id == img.id)).all()
        if not anns:
            reasons["no_annotation"] += 1
            continue
        cls_id = anns[0].class_id
        if cls_id >= len(classes):
            reasons["class_id_out_of_range"] += 1
            continue
        cls_name = classes[cls_id].replace("/", "_").replace("\\", "_")
        split    = "val" if img.id in val_ids else "train"
        src      = os.path.join(UPLOAD_DIR, img.filename)
        if not os.path.exists(src):
            reasons["file_not_found"] += 1
            continue

        try:
            pil = PILImage.open(src).convert("RGB")
            pil = pil.resize((input_w, input_h), PILImage.BILINEAR)
            dst = os.path.join(dataset_dir, split, cls_name, img.filename + ".jpg")
            pil.save(dst, "JPEG")
            placed += 1
        except Exception:
            reasons["corrupt"] += 1
            continue

    total_skipped = sum(reasons.values())
    return dataset_dir, total_skipped, reasons, placed


# ── PyTorch model builder ──────────────────────────────────────────────────────

def _build_torch_model(layers: list, input_h: int, input_w: int, num_classes: int):
    """Build nn.Sequential from layers_json, auto-wiring channels."""
    import torch.nn as nn

    modules = []
    in_channels = 3
    current_channels = 3
    spatial_h = input_h
    spatial_w = input_w
    flattened = False
    flat_size = None

    for layer in layers:
        lt = layer.get("type", "")
        p  = layer.get("params", {})

        if lt == "conv2d":
            filters     = int(p.get("filters", 32))
            kernel_size = int(p.get("kernel_size", 3))
            stride      = int(p.get("stride", 1))
            padding     = int(p.get("padding", 1))
            if flattened:
                raise ValueError(f"Cannot add conv2d after flatten")
            modules.append(nn.Conv2d(current_channels, filters, kernel_size, stride, padding))
            current_channels = filters
            spatial_h = (spatial_h + 2 * padding - kernel_size) // stride + 1
            spatial_w = (spatial_w + 2 * padding - kernel_size) // stride + 1

        elif lt == "batchnorm2d":
            if flattened:
                raise ValueError("Cannot add batchnorm2d after flatten")
            modules.append(nn.BatchNorm2d(current_channels))

        elif lt in ("maxpool2d", "avgpool2d"):
            if flattened:
                raise ValueError(f"Cannot add {lt} after flatten")
            ks = int(p.get("kernel_size", 2))
            st = int(p.get("stride", ks))
            pool_cls = nn.MaxPool2d if lt == "maxpool2d" else nn.AvgPool2d
            modules.append(pool_cls(ks, st))
            spatial_h = max(1, spatial_h // st)
            spatial_w = max(1, spatial_w // st)

        elif lt == "relu":
            modules.append(nn.ReLU())

        elif lt == "gelu":
            modules.append(nn.GELU())

        elif lt == "sigmoid":
            modules.append(nn.Sigmoid())

        elif lt == "dropout":
            prob = float(p.get("p", 0.5))
            modules.append(nn.Dropout(prob))

        elif lt == "flatten":
            modules.append(nn.Flatten())
            flat_size = current_channels * spatial_h * spatial_w
            flattened = True

        elif lt == "linear":
            out_features = int(p.get("out_features", 128))
            if not flattened:
                # Auto-flatten first
                modules.append(nn.Flatten())
                flat_size = current_channels * spatial_h * spatial_w
                flattened = True
                in_f = flat_size
            else:
                in_f = flat_size
            modules.append(nn.Linear(in_f, out_features))
            flat_size = out_features

    # Classifier head
    if not flattened:
        modules.append(nn.Flatten())
        flat_size = current_channels * spatial_h * spatial_w
    modules.append(nn.Linear(flat_size, num_classes))

    return nn.Sequential(*modules)


# ── Training thread ────────────────────────────────────────────────────────────

def _run_custom_training(run_id: int, project_id: int, body: RunBody):
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    import torchvision.datasets as D
    from torch.utils.data import DataLoader
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, StepLR
    from sqlmodel import create_engine, Session as S

    epochs    = body.epochs
    batch     = body.batch
    lr        = body.lr
    config_id = body.config_id

    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    _custom_state[run_id] = {"logs": [], "done": False, "stop_requested": False}

    def push(msg: str):
        _custom_state[run_id]["logs"].append(msg)

    try:
        with S(engine) as session:
            project = session.get(Project, project_id)
            if not project:
                raise ValueError("Project not found")
            classes  = project.classes
            run      = session.get(CustomTrainingRun, run_id)
            cfg      = session.get(CustomModelConfig, config_id)
            if not cfg:
                raise ValueError("Config not found")
            layers   = json.loads(cfg.layers_json)
            input_h  = cfg.input_h
            input_w  = cfg.input_w

            run.status = "running"
            session.add(run); session.commit()

            push("Building dataset…")
            dataset_dir, skipped, reasons, placed = _build_custom_dataset(
                project_id, run_id, classes, input_h, input_w, body.val_split, session
            )
            push(f"Dataset ready — {placed} images placed, {skipped} skipped")
            if reasons["no_annotation"] > 0:
                push(f"  ⚠ {reasons['no_annotation']} images have no annotation (open Annotate and label them)")
            if reasons["file_not_found"] > 0:
                push(f"  ⚠ {reasons['file_not_found']} image files missing from uploads folder")
            if reasons["class_id_out_of_range"] > 0:
                push(f"  ⚠ {reasons['class_id_out_of_range']} annotations have a class_id that doesn't match project classes")
            if reasons["corrupt"] > 0:
                push(f"  ⚠ {reasons['corrupt']} images could not be opened (corrupt files)")
            if placed == 0:
                hint = []
                if reasons["no_annotation"] == skipped:
                    hint.append("None of your images have annotations. Go to the Annotate page and draw at least one bounding box per image to assign a class label.")
                elif reasons["file_not_found"] == skipped:
                    hint.append("Image files are missing from the uploads folder. Try re-uploading your images.")
                else:
                    hint.append(f"Breakdown: {reasons}")
                raise ValueError("No images could be placed into the dataset. " + " ".join(hint))

        # CUDA detection
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            try:
                torch.zeros(1).cuda()
            except RuntimeError:
                use_cuda = False
                push("CUDA detected but kernel incompatible — falling back to CPU")
        device = torch.device("cuda" if use_cuda else "cpu")
        push(f"Device: {device}  |  Classes: {len(classes)}  |  Input: {input_h}x{input_w}")

        # ── Augmentation transforms ────────────────────────────────────────
        aug = [T.Resize((input_h, input_w))]
        if body.fliplr > 0:
            aug.append(T.RandomHorizontalFlip(p=body.fliplr))
        if body.flipud > 0:
            aug.append(T.RandomVerticalFlip(p=body.flipud))
        affine_kw: dict = {}
        if body.degrees > 0:
            affine_kw["degrees"] = body.degrees
        if body.translate > 0:
            affine_kw["translate"] = (body.translate, body.translate)
        if body.scale > 0:
            affine_kw["scale"] = (1 - body.scale, 1 + body.scale)
        if affine_kw:
            affine_kw.setdefault("degrees", 0)
            aug.append(T.RandomAffine(**affine_kw))
        if body.brightness > 0 or body.contrast > 0 or body.saturation > 0:
            aug.append(T.ColorJitter(
                brightness=body.brightness,
                contrast=body.contrast,
                saturation=body.saturation,
            ))
        aug.extend([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        if body.erasing > 0:
            aug.append(T.RandomErasing(p=body.erasing))

        train_tf = T.Compose(aug)
        val_tf   = T.Compose([
            T.Resize((input_h, input_w)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        train_ds = D.ImageFolder(os.path.join(dataset_dir, "train"), transform=train_tf)
        val_ds   = D.ImageFolder(os.path.join(dataset_dir, "val"),   transform=val_tf)

        if len(train_ds) == 0:
            raise ValueError("No training images found — annotate more images")
        if len(val_ds) == 0:
            raise ValueError("No validation images found — need more annotated images")

        train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=0)
        val_dl   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=0)

        # Build model
        push("Building custom CNN…")
        model = _build_torch_model(layers, input_h, input_w, len(classes))
        model = model.to(device)

        # Count params
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        push(f"Model built — {n_params:,} trainable parameters")

        # ── Optimizer ─────────────────────────────────────────────────────
        opt_name = body.optimizer.lower()
        if opt_name == "sgd":
            optim = torch.optim.SGD(
                model.parameters(), lr=lr,
                momentum=body.momentum, weight_decay=body.weight_decay)
        elif opt_name == "adamw":
            optim = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=body.weight_decay)
        else:
            optim = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=body.weight_decay)

        # ── LR scheduler ──────────────────────────────────────────────────
        wu = max(0, min(body.warmup_epochs, epochs - 1))
        sched_name = body.lr_scheduler.lower()

        if sched_name == "step":
            base_sched = StepLR(optim, step_size=max(1, body.step_size), gamma=body.step_gamma)
        elif sched_name == "none":
            base_sched = None
        else:  # cosine (default)
            base_sched = CosineAnnealingLR(optim, T_max=max(1, epochs - wu), eta_min=lr * 0.01)

        if wu > 0 and base_sched is not None:
            warmup_sched = LinearLR(optim, start_factor=0.01, total_iters=wu)
            scheduler    = SequentialLR(optim, schedulers=[warmup_sched, base_sched],
                                        milestones=[wu])
        else:
            scheduler = base_sched

        criterion = nn.CrossEntropyLoss(
            label_smoothing=max(0.0, min(0.5, body.label_smoothing)))
        run_dir   = os.path.join(RUNS_DIR, f"custom_train_{run_id}")
        os.makedirs(run_dir, exist_ok=True)
        best_acc  = 0.0
        best_path = os.path.join(run_dir, "best.pth")
        patience_counter = 0

        push(f"Training — {epochs} ep  {body.optimizer}  lr={lr}  "
             f"sched={body.lr_scheduler}  wd={body.weight_decay}  "
             f"smooth={body.label_smoothing:.2f}"
             + (f"  warmup={wu} ep" if wu > 0 else "")
             + (f"  early-stop patience={body.patience}" if body.patience > 0 else ""))

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for imgs, labels in train_dl:
                imgs, labels = imgs.to(device), labels.to(device)

                # Mixup
                if body.mixup > 0 and torch.rand(1).item() < body.mixup:
                    lam = float(torch.distributions.Beta(
                        torch.tensor(0.4), torch.tensor(0.4)).sample())
                    idx  = torch.randperm(imgs.size(0), device=device)
                    imgs = lam * imgs + (1 - lam) * imgs[idx]
                    loss = lam * criterion(model(imgs), labels) + \
                           (1 - lam) * criterion(model(imgs), labels[idx])
                    optim.zero_grad(); loss.backward(); optim.step()
                else:
                    optim.zero_grad()
                    loss = criterion(model(imgs), labels)
                    loss.backward(); optim.step()
                train_loss += loss.item()

            if scheduler is not None:
                scheduler.step()
            current_lr = optim.param_groups[0]["lr"]

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

            # format: epoch/total:acc:train_loss:val_loss
            push(f"__PROGRESS__:{epoch}/{epochs}:{acc:.4f}:{tloss:.4f}:{vloss:.4f}")
            push(f"  Epoch {epoch}/{epochs} — val_acc={acc*100:.1f}%  "
                 f"val_loss={vloss:.4f}  train_loss={tloss:.4f}  lr={current_lr:.6f}")

            if _custom_state[run_id].get("stop_requested"):
                push("[STOPPED] Stop requested — saving best weights and exiting early")
                break

            if body.patience > 0 and patience_counter >= body.patience:
                push(f"[STOPPED] Early stopping at epoch {epoch} "
                     f"(no improvement for {body.patience} epochs)")
                break

        # ── Final per-class evaluation ─────────────────────────────────────
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, weights_only=False))
        model.eval()
        n_classes        = len(classes)
        class_correct    = [0] * n_classes
        class_total      = [0] * n_classes
        all_true: list   = []
        all_pred: list   = []
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

        cm = [[0] * n_classes for _ in range(n_classes)]
        for t, p in zip(all_true, all_pred):
            if 0 <= t < n_classes and 0 <= p < n_classes:
                cm[t][p] += 1

        per_class_metrics: dict = {}
        for i in range(n_classes):
            tp        = cm[i][i]
            fp        = sum(cm[j][i] for j in range(n_classes) if j != i)
            fn        = sum(cm[i][j] for j in range(n_classes) if j != i)
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
            run = session.get(CustomTrainingRun, run_id)
            run.status       = "done"
            run.model_path   = best_path
            run.run_dir      = run_dir
            run.results_json = json.dumps(metrics)
            session.add(run); session.commit()

        push(f"[DONE] Best val accuracy: {best_acc*100:.1f}%")
        push(f"__DONE__:{json.dumps(metrics)}")

    except Exception as exc:
        push(f"Error: {exc}")
        push("__FAILED__")
        try:
            from sqlmodel import create_engine, Session as S2
            eng2 = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
            with S2(eng2) as session:
                run = session.get(CustomTrainingRun, run_id)
                if run:
                    run.status = "failed"
                    session.add(run); session.commit()
        except Exception:
            pass
    finally:
        _custom_state[run_id]["done"] = True


# ── Start training run ─────────────────────────────────────────────────────────

@router.post("/runs", response_model=RunOut)
def start_run(project_id: int, body: RunBody,
              session: Session = Depends(get_session)):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    cfg = session.get(CustomModelConfig, body.config_id)
    if not cfg or cfg.project_id != project_id:
        raise HTTPException(404, "Config not found")
    if len(project.classes) < 2:
        raise HTTPException(400, "Need at least 2 classes")

    run = CustomTrainingRun(
        config_id=body.config_id,
        project_id=project_id,
        status="pending",
        epochs=body.epochs,
        batch=body.batch,
        lr=body.lr,
        created_at=datetime.now().isoformat(),
    )
    session.add(run); session.commit(); session.refresh(run)

    threading.Thread(
        target=_run_custom_training,
        args=(run.id, project_id, body),
        daemon=True,
    ).start()

    return _run_to_out(run)


# ── SSE log stream ─────────────────────────────────────────────────────────────

@router.get("/runs/{run_id}/stream")
def stream_run(project_id: int, run_id: int,
               session: Session = Depends(get_session)):
    run = session.get(CustomTrainingRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(404, "Run not found")

    def event_stream():
        # Wait up to 15s for training thread to register state
        for _ in range(30):
            if run_id in _custom_state:
                break
            time.sleep(0.5)
        else:
            yield f"data: Run status: {run.status}\n\n"
            yield "data: __END__\n\n"
            return

        sent     = 0
        deadline = time.time() + 4 * 3600
        while time.time() < deadline:
            st   = _custom_state[run_id]
            logs = st["logs"]
            while sent < len(logs):
                yield f"data: {logs[sent]}\n\n"
                sent += 1
            if st["done"]:
                yield "data: __END__\n\n"
                _custom_state.pop(run_id, None)
                return
            time.sleep(0.5)

        yield "data: Stream timeout\n\n"
        yield "data: __END__\n\n"
        _custom_state.pop(run_id, None)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Stop / Delete ──────────────────────────────────────────────────────────────

@router.post("/runs/{run_id}/stop")
def stop_run(project_id: int, run_id: int, session: Session = Depends(get_session)):
    run = session.get(CustomTrainingRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(404, "Run not found")
    st = _custom_state.get(run_id, {})
    if st:
        st["stop_requested"] = True
    if run.status == "running":
        run.status = "stopped"
        session.add(run); session.commit()
    return {"ok": True}


@router.delete("/runs/{run_id}")
def delete_run(project_id: int, run_id: int, session: Session = Depends(get_session)):
    run = session.get(CustomTrainingRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(404, "Run not found")
    if run.status == "running":
        raise HTTPException(400, "Stop the run before deleting")
    session.delete(run); session.commit()
    return {"ok": True}


@router.get("/runs/{run_id}/download")
def download_run_model(project_id: int, run_id: int,
                       session: Session = Depends(get_session)):
    run = session.get(CustomTrainingRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(404, "Run not found")
    if not run.model_path or not os.path.exists(run.model_path):
        raise HTTPException(404, "Model file not found")
    return FileResponse(run.model_path, filename=f"custom_model_run{run_id}.pth",
                        media_type="application/octet-stream")


@router.post("/runs/{run_id}/export-onnx")
def export_run_onnx(project_id: int, run_id: int,
                    session: Session = Depends(get_session)):
    """Export best checkpoint to ONNX."""
    import torch

    run = session.get(CustomTrainingRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(404, "Run not found")
    if not run.model_path or not os.path.exists(run.model_path):
        raise HTTPException(404, "Model file not found — train the run first")

    project   = session.get(Project, project_id)
    cfg       = session.get(CustomModelConfig, run.config_id)
    if not cfg:
        raise HTTPException(404, "Config not found")

    n_classes = len(project.classes)
    layers    = json.loads(cfg.layers_json)
    onnx_path = run.model_path.replace(".pth", ".onnx")

    if not os.path.exists(onnx_path):
        model = _build_torch_model(layers, cfg.input_h, cfg.input_w, n_classes)
        model.load_state_dict(torch.load(run.model_path, map_location="cpu", weights_only=False))
        model.eval()
        dummy = torch.randn(1, 3, cfg.input_h, cfg.input_w)
        torch.onnx.export(
            model, dummy, onnx_path,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17,
        )

    return FileResponse(onnx_path, filename=f"custom_model_run{run_id}.onnx",
                        media_type="application/octet-stream")


