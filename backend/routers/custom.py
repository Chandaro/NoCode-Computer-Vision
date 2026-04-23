from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlmodel import Session, select
from pydantic import BaseModel
import os, json, shutil, threading, time
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
                           input_h: int, input_w: int, session: Session) -> tuple[str, int]:
    """Build ImageFolder layout resized to input_h×input_w."""
    from PIL import Image as PILImage

    dataset_dir = os.path.join(RUNS_DIR, f"custom_dataset_{project_id}_{run_id}")
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)

    images = session.exec(select(Image).where(Image.project_id == project_id)).all()

    for split in ["train", "val"]:
        for cls in classes:
            safe = cls.replace("/", "_").replace("\\", "_")
            os.makedirs(os.path.join(dataset_dir, split, safe), exist_ok=True)

    n_val = max(1, int(len(images) * 0.2))
    val_ids = {img.id for img in images[:n_val]}

    skipped = 0
    for img in images:
        anns = session.exec(select(Annotation).where(Annotation.image_id == img.id)).all()
        if not anns:
            skipped += 1
            continue
        cls_id = anns[0].class_id
        if cls_id >= len(classes):
            skipped += 1
            continue
        cls_name = classes[cls_id].replace("/", "_").replace("\\", "_")
        split    = "val" if img.id in val_ids else "train"
        src      = os.path.join(UPLOAD_DIR, img.filename)
        if not os.path.exists(src):
            skipped += 1
            continue

        try:
            pil = PILImage.open(src).convert("RGB")
            pil = pil.resize((input_w, input_h), PILImage.BILINEAR)
            dst = os.path.join(dataset_dir, split, cls_name, img.filename + ".jpg")
            pil.save(dst, "JPEG")
        except Exception:
            skipped += 1
            continue

    return dataset_dir, skipped


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

def _run_custom_training(run_id: int, project_id: int, config_id: int,
                          epochs: int, batch: int, lr: float):
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    import torchvision.datasets as D
    from torch.utils.data import DataLoader
    from sqlmodel import create_engine, Session as S

    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    _custom_state[run_id] = {"logs": [], "done": False}

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
            dataset_dir, skipped = _build_custom_dataset(
                project_id, run_id, classes, input_h, input_w, session
            )
            push(f"Dataset ready — {skipped} images skipped")

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

        # Transforms
        tf = T.Compose([
            T.Resize((input_h, input_w)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        train_ds = D.ImageFolder(os.path.join(dataset_dir, "train"), transform=tf)
        val_ds   = D.ImageFolder(os.path.join(dataset_dir, "val"),   transform=tf)

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

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        run_dir   = os.path.join(RUNS_DIR, f"custom_train_{run_id}")
        os.makedirs(run_dir, exist_ok=True)
        best_acc  = 0.0
        best_path = os.path.join(run_dir, "best.pth")

        push(f"Training started — {epochs} epochs, batch={batch}, lr={lr}")

        for epoch in range(1, epochs + 1):
            model.train()
            for imgs, labels in train_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                out  = model(imgs)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()

            model.eval()
            correct = total = 0
            val_loss = 0.0
            with torch.no_grad():
                for imgs, labels in val_dl:
                    imgs, labels = imgs.to(device), labels.to(device)
                    out      = model(imgs)
                    val_loss += criterion(out, labels).item()
                    preds    = out.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total   += labels.size(0)

            acc   = correct / max(total, 1)
            vloss = val_loss / max(len(val_dl), 1)

            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), best_path)

            push(f"__PROGRESS__:{epoch}/{epochs}:{acc:.4f}")
            push(f"  Epoch {epoch}/{epochs} — val_acc={acc*100:.1f}%  val_loss={vloss:.4f}")

        metrics = {"top1_acc": round(best_acc, 4), "epochs": epochs}

        with S(engine) as session:
            run = session.get(CustomTrainingRun, run_id)
            run.status       = "done"
            run.model_path   = best_path
            run.run_dir      = run_dir
            run.results_json = json.dumps(metrics)
            session.add(run); session.commit()

        push(f"Done! Best val accuracy: {best_acc*100:.1f}%")
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
        args=(run.id, project_id, body.config_id, body.epochs, body.batch, body.lr),
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
