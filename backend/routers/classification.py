from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from sqlmodel import Session, select
from pydantic import BaseModel
import os, json, shutil, threading, time
from datetime import datetime

from database import get_session, DATABASE_URL
from models import ClassificationRun, Project, Image, Annotation

router = APIRouter(prefix="/projects/{project_id}/classification", tags=["classification"])

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")
RUNS_DIR   = os.path.join(os.path.dirname(__file__), "..", "runs")
os.makedirs(RUNS_DIR, exist_ok=True)

_cls_state: dict = {}

BASE_MODELS = ["resnet18", "resnet50", "mobilenet_v3_small", "efficientnet_b0"]


class ClsConfig(BaseModel):
    epochs: int = 10
    imgsz: int = 224
    batch: int = 32
    base_model: str = "resnet18"
    lr: float = 0.001
    freeze_backbone: bool = True


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


# ─── Dataset builder ─────────────────────────────────────────────────────────
def _build_cls_dataset(project_id: int, run_id: int, classes: list, session: Session) -> str:
    """Build ImageFolder layout: run_dir/train/<class_name>/img.jpg."""
    dataset_dir = os.path.join(RUNS_DIR, f"cls_dataset_{project_id}_{run_id}")
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)

    images = session.exec(select(Image).where(Image.project_id == project_id)).all()

    for split in ["train", "val"]:
        for cls in classes:
            os.makedirs(os.path.join(dataset_dir, split, cls), exist_ok=True)

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
        cls_name = classes[cls_id]
        split    = "val" if img.id in val_ids else "train"
        src      = os.path.join(UPLOAD_DIR, img.filename)
        if not os.path.exists(src):
            skipped += 1
            continue
        shutil.copy2(src, os.path.join(dataset_dir, split, cls_name, img.filename))

    return dataset_dir, skipped


# ─── Training thread ──────────────────────────────────────────────────────────
def _run_classification(run_id: int, project_id: int, config: ClsConfig):
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    import torchvision.datasets as D
    from torch.utils.data import DataLoader
    import torchvision.models as M
    from sqlmodel import create_engine, Session as S

    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    _cls_state[run_id] = {"logs": [], "done": False}

    def push(msg: str):
        _cls_state[run_id]["logs"].append(msg)

    try:
        with S(engine) as session:
            project = session.get(Project, project_id)
            classes = project.classes
            run     = session.get(ClassificationRun, run_id)
            run.status = "running"
            session.add(run); session.commit()

            push("⏳ Building classification dataset…")
            dataset_dir, skipped = _build_cls_dataset(project_id, run_id, classes, session)
            push(f"✅ Dataset ready — {skipped} images skipped (no annotation)")

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            try:
                torch.zeros(1).cuda()
            except RuntimeError:
                use_cuda = False
                push("⚠️ CUDA detected but kernel incompatible — falling back to CPU")
        device = torch.device("cuda" if use_cuda else "cpu")
        push(f"📦 Loading base model: {config.base_model}  |  device: {device}")

        # ── Transforms ────────────────────────────────────────────────────
        train_tf = T.Compose([
            T.Resize((config.imgsz, config.imgsz)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        val_tf = T.Compose([
            T.Resize((config.imgsz, config.imgsz)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        train_ds = D.ImageFolder(os.path.join(dataset_dir, "train"), transform=train_tf)
        val_ds   = D.ImageFolder(os.path.join(dataset_dir, "val"),   transform=val_tf)
        train_dl = DataLoader(train_ds, batch_size=config.batch, shuffle=True,  num_workers=0)
        val_dl   = DataLoader(val_ds,   batch_size=config.batch, shuffle=False, num_workers=0)

        n_classes = len(classes)

        # ── Load pretrained model ──────────────────────────────────────────
        weights_map = {
            "resnet18":          M.ResNet18_Weights.DEFAULT,
            "resnet50":          M.ResNet50_Weights.DEFAULT,
            "mobilenet_v3_small": M.MobileNet_V3_Small_Weights.DEFAULT,
            "efficientnet_b0":   M.EfficientNet_B0_Weights.DEFAULT,
        }
        model_fn_map = {
            "resnet18":          M.resnet18,
            "resnet50":          M.resnet50,
            "mobilenet_v3_small": M.mobilenet_v3_small,
            "efficientnet_b0":   M.efficientnet_b0,
        }
        weights = weights_map.get(config.base_model, M.ResNet18_Weights.DEFAULT)
        model   = model_fn_map.get(config.base_model, M.resnet18)(weights=weights)

        # Replace classifier head
        if hasattr(model, "fc"):
            model.fc = nn.Linear(model.fc.in_features, n_classes)
        elif hasattr(model, "classifier"):
            in_feat = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_feat, n_classes)

        if config.freeze_backbone:
            for name, param in model.named_parameters():
                if "fc" not in name and "classifier" not in name:
                    param.requires_grad = False

        model = model.to(device)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr
        )
        criterion = nn.CrossEntropyLoss()

        run_dir  = os.path.join(RUNS_DIR, f"cls_train_{run_id}")
        os.makedirs(run_dir, exist_ok=True)
        best_acc = 0.0
        best_path = os.path.join(run_dir, "best.pth")

        push(f"🚀 Training started — {config.epochs} epochs, imgsz={config.imgsz}, batch={config.batch}, lr={config.lr}")

        for epoch in range(1, config.epochs + 1):
            # ── Train ──────────────────────────────────────────────────────
            model.train()
            total_loss = 0.0
            for imgs, labels in train_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                out  = model(imgs)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # ── Validate ───────────────────────────────────────────────────
            model.eval()
            correct = total = 0
            val_loss = 0.0
            with torch.no_grad():
                for imgs, labels in val_dl:
                    imgs, labels = imgs.to(device), labels.to(device)
                    out  = model(imgs)
                    val_loss += criterion(out, labels).item()
                    preds    = out.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total   += labels.size(0)

            acc  = correct / max(total, 1)
            vloss = val_loss / max(len(val_dl), 1)

            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), best_path)

            push(f"__PROGRESS__:{epoch}/{config.epochs}:{acc:.4f}:{acc:.4f}:{acc:.4f}")
            push(f"  Epoch {epoch}/{config.epochs} — val_acc={acc*100:.1f}%  val_loss={vloss:.4f}")

        # ── Final evaluation (per-class, top-5, confusion matrix) ────────────
        model.load_state_dict(torch.load(best_path, weights_only=False))
        model.eval()
        class_correct = [0] * n_classes
        class_total   = [0] * n_classes
        all_true: list[int] = []
        all_pred: list[int] = []
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

        per_class = {
            classes[i]: round(class_correct[i] / max(class_total[i], 1), 4)
            for i in range(n_classes)
        }

        cm = [[0] * n_classes for _ in range(n_classes)]
        for t, p in zip(all_true, all_pred):
            if 0 <= t < n_classes and 0 <= p < n_classes:
                cm[t][p] += 1

        metrics = {
            "top1_acc":         round(best_acc, 4),
            "top5_acc":         round(top5_correct / max(top5_total, 1), 4),
            "per_class":        per_class,
            "confusion_matrix": cm,
        }

        with S(engine) as session:
            run = session.get(ClassificationRun, run_id)
            run.status       = "done"
            run.model_path   = best_path
            run.run_dir      = run_dir
            run.results_json = json.dumps(metrics)
            session.add(run); session.commit()

        push(f"✅ Done! Best val accuracy: {best_acc*100:.1f}%")
        push(f"__DONE__:{json.dumps(metrics)}")

    except Exception as exc:
        push(f"❌ Error: {exc}")
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

        yield "data: ⚠️ Stream timeout\n\n"
        yield "data: __END__\n\n"
        _cls_state.pop(run_id, None)

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


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
