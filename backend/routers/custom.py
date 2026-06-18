from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from sqlmodel import Session, select
from pydantic import BaseModel
import os, json, shutil, threading, time, random
from datetime import datetime

from database import get_session, DATABASE_URL
from models import CustomModelConfig, CustomTrainingRun, Project, Image, Annotation

router = APIRouter(prefix="/projects/{project_id}/custom", tags=["custom"])

UPLOAD_DIR   = os.path.join(os.path.dirname(__file__), "..", "uploads")
RUNS_DIR     = os.path.join(os.path.dirname(__file__), "..", "runs")
CLS_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "cls_uploads")
os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(CLS_DATA_DIR, exist_ok=True)

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

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
    lr_scheduler: str = "cosine"  # cosine | onecycle | step | none
    step_size: int = 10
    step_gamma: float = 0.1
    # Regularisation
    label_smoothing: float = 0.0
    # Quality
    amp: bool = True              # mixed precision (auto-used only on CUDA)
    grad_clip: float = 1.0        # gradient clipping max-norm; 0 = off
    class_weights: bool = False   # weight the loss by inverse class frequency
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


@router.get("/estimate")
def custom_estimate(project_id: int, input_h: int = 96, input_w: int = 96,
                    batch: int = 32, pretrained: bool = False, backbone: int = 0,
                    freeze: bool = False, params: int = 0,
                    session: Session = Depends(get_session)):
    """Pre-flight GPU VRAM / RAM estimate for a Conv Builder training run.
    Handles both transfer-learning backbones and from-scratch custom CNNs."""
    from routers import hw_estimate as hw
    import os as _os
    side = max(input_h, input_w)
    if pretrained:
        bb = BACKBONES[max(0, min(backbone, len(BACKBONES) - 1))]
        est_vram, overhead, per_eff = hw.estimate_classification(bb, side, batch, freeze=freeze)
    else:
        est_vram, overhead, per_eff = hw.estimate_custom_cnn(side, batch, params_m=params / 1e6)
    workers = 0  # custom/classification loaders run with num_workers=0
    result = hw.device_status()
    result.update({"est_vram_gb": est_vram, "batch": batch, "imgsz": side,
                   "ram": {**result.get("ram", {}), "est_gb": round(2.5 + workers * 0.8 + 0.4, 2)},
                   "note": "Estimate only — real usage varies with architecture/augmentation."})
    if result["gpu"]:
        v, sug = hw.verdict(est_vram, result["gpu"]["free_gb"], overhead, per_eff)
        result["gpu"].update({"verdict": v, "suggested_batch": sug})
    else:
        result["note"] = "Training on CPU — no GPU detected. Expect slow epochs."
    return result


@router.get("/runs/{run_id}", response_model=RunOut)
def get_run(project_id: int, run_id: int, session: Session = Depends(get_session)):
    run = session.get(CustomTrainingRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(404, "Run not found")
    return _run_to_out(run)


# ── Dataset builder ────────────────────────────────────────────────────────────

def _build_custom_dataset(project_id: int, run_id: int, classes: list,
                           input_h: int, input_w: int,
                           val_split: float, session: Session) -> tuple[str, int, dict, int]:
    """Build an ImageFolder from the project's classification dataset
    (cls_uploads/<project>/<class>/...), resized to input_h×input_w.
    Returns (dataset_dir, total_skipped, skip_reason_counts, placed).
    """
    from PIL import Image as PILImage

    dataset_dir = os.path.join(RUNS_DIR, f"custom_dataset_{project_id}_{run_id}")
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)

    for split in ["train", "val"]:
        for cls in classes:
            safe = cls.replace("/", "_").replace("\\", "_")
            os.makedirs(os.path.join(dataset_dir, split, safe), exist_ok=True)

    reasons = {"file_not_found": 0, "corrupt": 0}
    placed  = 0

    for cls in classes:
        safe    = cls.replace("/", "_").replace("\\", "_")
        cls_dir = os.path.join(CLS_DATA_DIR, str(project_id), cls)
        if not os.path.exists(cls_dir):
            continue
        imgs = [f for f in os.listdir(cls_dir)
                if os.path.splitext(f)[1].lower() in IMG_EXTS]
        random.shuffle(imgs)
        n_val = max(1, int(len(imgs) * val_split))
        # Cache at a resolution >= the model input (never below) so the on-the-fly
        # Resize keeps detail and augmentation (affine/scale/erasing) has real
        # pixels to work with. Big originals are capped at 256 to bound disk; we
        # never upscale small originals beyond their own size (but keep >= input).
        need = max(input_h, input_w)
        cap  = max(256, need)
        for i, fname in enumerate(imgs):
            split = "val" if i < n_val else "train"
            src   = os.path.join(cls_dir, fname)
            if not os.path.exists(src):
                reasons["file_not_found"] += 1
                continue
            try:
                pil = PILImage.open(src).convert("RGB")
                long_side = max(pil.size)
                side = max(need, min(cap, long_side))
                if (side, side) != pil.size:
                    pil = pil.resize((side, side), PILImage.BILINEAR)
                # Quality 95 (not the default 75) to avoid stacking JPEG artifacts.
                pil.save(os.path.join(dataset_dir, split, safe, fname + ".jpg"),
                         "JPEG", quality=95)
                placed += 1
            except Exception:
                reasons["corrupt"] += 1

    total_skipped = sum(reasons.values())
    return dataset_dir, total_skipped, reasons, placed


# ── Macro blocks (composite nn.Modules with internal branches) ──────────────────

def _make_blocks():
    """Define the macro-block classes. Defined lazily so torch imports stay local."""
    import torch.nn as nn

    class ConvBlock(nn.Module):
        """Conv → BatchNorm → ReLU, the most common building block."""
        def __init__(self, in_c, out_c, k=3, stride=1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_c, out_c, k, stride, k // 2, bias=False),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))
        def forward(self, x): return self.net(x)

    class ResidualBlock(nn.Module):
        """ResNet basic block: out = ReLU(conv(x) + shortcut(x))."""
        def __init__(self, in_c, out_c, stride=1):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False), nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False), nn.BatchNorm2d(out_c))
            self.short = (nn.Identity() if (stride == 1 and in_c == out_c) else
                          nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                                        nn.BatchNorm2d(out_c)))
            self.act = nn.ReLU(inplace=True)
        def forward(self, x): return self.act(self.conv(x) + self.short(x))

    class DWSepBlock(nn.Module):
        """Depthwise-separable block (MobileNet style): cheap conv."""
        def __init__(self, in_c, out_c, stride=1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_c, in_c, 3, stride, 1, groups=in_c, bias=False),
                nn.BatchNorm2d(in_c), nn.ReLU(inplace=True),
                nn.Conv2d(in_c, out_c, 1, bias=False),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))
        def forward(self, x): return self.net(x)

    class SEBlock(nn.Module):
        """Squeeze-and-Excitation channel attention (keeps shape)."""
        def __init__(self, c, r=16):
            super().__init__()
            h = max(1, c // r)
            self.fc = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(c, h), nn.ReLU(inplace=True),
                nn.Linear(h, c), nn.Sigmoid())
        def forward(self, x):
            return x * self.fc(x).view(x.size(0), x.size(1), 1, 1)

    return ConvBlock, ResidualBlock, DWSepBlock, SEBlock


# ── Pretrained backbones (transfer learning) ────────────────────────────────────
#
# Industry-standard workflow: start from an ImageNet-pretrained backbone and
# train a small custom head. On small datasets this routinely beats a
# from-scratch CNN by 20–40 accuracy points.
#
# Encoded as a single layers_json entry: {"type": "pretrained",
#   "params": {"model": 0|1|2, "freeze": 0|1}}
#   model:  0 = ResNet18 · 1 = MobileNetV3-Small · 2 = EfficientNet-B0
#   freeze: 0 = linear probe (freeze backbone, train head only — fast)
#           1 = fine-tune (train everything — slower, best accuracy)

BACKBONES = ["resnet18", "mobilenet_v3_small", "efficientnet_b0"]


def _build_pretrained(model_idx: int, freeze_idx: int, num_classes: int,
                      load_imagenet: bool):
    """Build a torchvision backbone with a fresh classification head.

    load_imagenet=True downloads/uses ImageNet weights (training time).
    load_imagenet=False builds the bare architecture (inference/export reload,
    where the trained state dict supplies all weights)."""
    import torch.nn as nn
    import torchvision.models as M

    name = BACKBONES[max(0, min(model_idx, len(BACKBONES) - 1))]
    weights = "DEFAULT" if load_imagenet else None

    if name == "resnet18":
        net = M.resnet18(weights=weights)
        in_f = net.fc.in_features
        head = nn.Linear(in_f, num_classes)
        net.fc = head
    elif name == "mobilenet_v3_small":
        net = M.mobilenet_v3_small(weights=weights)
        in_f = net.classifier[-1].in_features
        head = nn.Linear(in_f, num_classes)
        net.classifier[-1] = head
    else:  # efficientnet_b0
        net = M.efficientnet_b0(weights=weights)
        in_f = net.classifier[-1].in_features
        head = nn.Linear(in_f, num_classes)
        net.classifier[-1] = head

    # Linear probe: freeze everything except the new head
    if freeze_idx == 0:
        for p in net.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True

    return net, name


# ── PyTorch model builder ──────────────────────────────────────────────────────

def _build_torch_model(layers: list, input_h: int, input_w: int, num_classes: int,
                       load_imagenet: bool = False):
    """Build an nn.Sequential of primitive layers and macro blocks, auto-wiring
    channels and tracking spatial size. A single 'pretrained' entry instead
    builds a torchvision backbone with a custom head (transfer learning)."""
    import torch.nn as nn

    # Transfer-learning mode: the whole model IS the backbone + head
    if layers and layers[0].get("type") == "pretrained":
        p = layers[0].get("params", {})
        net, _ = _build_pretrained(int(p.get("model", 0)), int(p.get("freeze", 0)),
                                   num_classes, load_imagenet)
        return net

    ConvBlock, ResidualBlock, DWSepBlock, SEBlock = _make_blocks()

    # Simple activation factory for the extra activations
    ACTIVATIONS = {
        "relu": nn.ReLU, "gelu": nn.GELU, "sigmoid": nn.Sigmoid,
        "leakyrelu": nn.LeakyReLU, "silu": nn.SiLU, "mish": nn.Mish,
        "hardswish": nn.Hardswish, "elu": nn.ELU, "tanh": nn.Tanh,
    }

    modules = []
    current_channels = 3
    spatial_h = input_h
    spatial_w = input_w
    flattened = False
    flat_size = None

    def _check_spatial(name):
        if spatial_h < 1 or spatial_w < 1:
            raise ValueError(
                f"The image has shrunk to nothing at the {name} layer. For a "
                f"{input_h}×{input_w} input, remove some pooling/stride layers or "
                "increase the input size.")

    def _need_spatial(name):
        if flattened:
            raise ValueError(f"Cannot add {name} after Flatten. It must come before "
                             "Flatten / Linear / Global Avg Pool.")

    for layer in layers:
        lt = layer.get("type", "")
        p  = layer.get("params", {})

        if lt == "conv2d" or lt == "conv1x1":
            _need_spatial("a Conv layer")
            filters     = int(p.get("filters", 32))
            if lt == "conv1x1":
                kernel_size, stride, padding = 1, 1, 0
            else:
                kernel_size = int(p.get("kernel_size", 3))
                stride      = int(p.get("stride", 1))
                padding     = int(p.get("padding", 1))
            modules.append(nn.Conv2d(current_channels, filters, kernel_size, stride, padding))
            current_channels = filters
            spatial_h = (spatial_h + 2 * padding - kernel_size) // stride + 1
            spatial_w = (spatial_w + 2 * padding - kernel_size) // stride + 1
            _check_spatial("Conv")

        elif lt in ("conv_block", "residual_block", "dwsep_block"):
            _need_spatial("a block")
            filters = int(p.get("filters", current_channels))
            stride  = int(p.get("stride", 1))
            if lt == "conv_block":
                modules.append(ConvBlock(current_channels, filters, 3, stride))
            elif lt == "residual_block":
                modules.append(ResidualBlock(current_channels, filters, stride))
            else:
                modules.append(DWSepBlock(current_channels, filters, stride))
            current_channels = filters
            spatial_h = spatial_h // stride
            spatial_w = spatial_w // stride
            _check_spatial(lt)

        elif lt == "se_block":
            _need_spatial("SE block")
            modules.append(SEBlock(current_channels))   # keeps shape

        elif lt == "batchnorm2d":
            _need_spatial("Batch Norm")
            modules.append(nn.BatchNorm2d(current_channels))

        elif lt == "groupnorm":
            _need_spatial("Group Norm")
            groups = int(p.get("groups", 8))
            # groups must divide channels; fall back to a valid divisor
            while groups > 1 and current_channels % groups != 0:
                groups -= 1
            modules.append(nn.GroupNorm(max(1, groups), current_channels))

        elif lt in ("maxpool2d", "avgpool2d"):
            _need_spatial("Max/Avg Pool")
            ks = int(p.get("kernel_size", 2))
            st = int(p.get("stride", ks))
            pool_cls = nn.MaxPool2d if lt == "maxpool2d" else nn.AvgPool2d
            modules.append(pool_cls(ks, st))
            spatial_h = spatial_h // st
            spatial_w = spatial_w // st
            _check_spatial("Pool")

        elif lt == "gap":
            # Global Average Pooling: collapse HxW to 1x1, then flatten to [C].
            _need_spatial("Global Avg Pool")
            modules.append(nn.AdaptiveAvgPool2d(1))
            modules.append(nn.Flatten())
            flat_size = current_channels
            flattened = True

        elif lt in ACTIVATIONS:
            modules.append(ACTIVATIONS[lt]())

        elif lt == "dropout":
            modules.append(nn.Dropout(float(p.get("p", 0.5))))

        elif lt == "flatten":
            modules.append(nn.Flatten())
            flat_size = current_channels * spatial_h * spatial_w
            flattened = True

        elif lt == "linear":
            out_features = int(p.get("out_features", 128))
            if not flattened:
                modules.append(nn.Flatten())
                flat_size = current_channels * spatial_h * spatial_w
                flattened = True
            modules.append(nn.Linear(flat_size, out_features))
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
    from torch.optim.lr_scheduler import (
        CosineAnnealingLR, LinearLR, SequentialLR, StepLR, OneCycleLR)
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
            if reasons["file_not_found"] > 0:
                push(f"  ⚠ {reasons['file_not_found']} image files missing")
            if reasons["corrupt"] > 0:
                push(f"  ⚠ {reasons['corrupt']} images could not be opened (corrupt files)")
            if placed == 0:
                # This dataset comes from the Classification Dataset section
                # (per-class image folders), NOT from bounding-box annotations.
                if reasons["corrupt"] == skipped and skipped > 0:
                    hint = "All images were corrupt or unreadable. Try re-uploading them."
                else:
                    hint = ("Upload images for at least 2 classes in the Classification "
                            "Dataset section (on the Image Classification page), then come "
                            "back and train.")
                raise ValueError("No images could be placed into the dataset. " + hint)

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
        # Geometric/colour augmentation runs FIRST, on the cached full-resolution
        # image, then we resize to the model input last. Augmenting before the
        # downscale keeps edges clean instead of warping a tiny 64px image.
        aug: list = []
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
            T.Resize((input_h, input_w)),
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

        # ImageFolder sorts class folders alphabetically — use that as the
        # authoritative class list so training and inference indices match.
        classes   = train_ds.classes
        n_classes = len(classes)

        if len(train_ds) == 0:
            raise ValueError("No training images found — annotate more images")
        if len(val_ds) == 0:
            raise ValueError("No validation images found — need more annotated images")

        train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=0)
        val_dl   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=0)

        # Build model
        is_pretrained = bool(layers) and layers[0].get("type") == "pretrained"
        if is_pretrained:
            bp = layers[0].get("params", {})
            bb_name = BACKBONES[max(0, min(int(bp.get("model", 0)), len(BACKBONES) - 1))]
            mode = "linear probe (backbone frozen)" if int(bp.get("freeze", 0)) == 0 else "full fine-tune"
            push(f"Transfer learning — {bb_name} pretrained on ImageNet · {mode}")
            push("Downloading pretrained weights on first use (cached afterwards)…")
        else:
            push("Building custom CNN…")
        model = _build_torch_model(layers, input_h, input_w, len(classes),
                                   load_imagenet=is_pretrained)
        model = model.to(device)

        # Count params
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total  = sum(p.numel() for p in model.parameters())
        if is_pretrained and n_params < n_total:
            push(f"Model built — {n_params:,} trainable of {n_total:,} total parameters")
        else:
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
        onecycle = False

        if sched_name == "onecycle":
            # Steps PER BATCH, not per epoch — handled separately in the loop.
            scheduler = OneCycleLR(optim, max_lr=lr, epochs=epochs,
                                   steps_per_epoch=max(1, len(train_dl)))
            onecycle = True
        else:
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

        # ── Class weighting (helps imbalanced datasets) ───────────────────
        weight_tensor = None
        if body.class_weights:
            counts = torch.zeros(n_classes)
            for _, lbl in train_ds.samples:
                counts[lbl] += 1
            counts = counts.clamp(min=1)
            weight_tensor = (counts.sum() / (counts * n_classes)).to(device)
            push(f"Class weighting on — counts {[int(c) for c in counts]}")

        criterion = nn.CrossEntropyLoss(
            weight=weight_tensor,
            label_smoothing=max(0.0, min(0.5, body.label_smoothing)))

        # ── Mixed precision (only meaningful on CUDA) ─────────────────────
        use_amp = bool(body.amp) and use_cuda
        scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)
        clip    = max(0.0, float(body.grad_clip))

        run_dir   = os.path.join(RUNS_DIR, f"custom_train_{run_id}")
        os.makedirs(run_dir, exist_ok=True)
        best_acc  = 0.0
        best_path = os.path.join(run_dir, "best.pth")
        patience_counter = 0

        push(f"Training — {epochs} ep  {body.optimizer}  lr={lr}  "
             f"sched={body.lr_scheduler}  wd={body.weight_decay}  "
             f"smooth={body.label_smoothing:.2f}"
             + (f"  warmup={wu} ep" if wu > 0 and not onecycle else "")
             + ("  amp" if use_amp else "")
             + (f"  clip={clip}" if clip > 0 else "")
             + ("  class-weighted" if weight_tensor is not None else "")
             + (f"  early-stop patience={body.patience}" if body.patience > 0 else ""))

        n_batches = max(1, len(train_dl))
        last_batch_push = 0.0
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for bi, (imgs, labels) in enumerate(train_dl, 1):
                # Within-epoch progress so the UI isn't frozen on long epochs
                now = time.time()
                if now - last_batch_push >= 2.0:
                    last_batch_push = now
                    push(f"__BATCH__:{epoch}/{epochs}:{min(1.0, bi / n_batches):.3f}")
                imgs, labels = imgs.to(device), labels.to(device)
                optim.zero_grad()

                with torch.cuda.amp.autocast(enabled=use_amp):
                    # Mixup
                    if body.mixup > 0 and torch.rand(1).item() < body.mixup:
                        lam = float(torch.distributions.Beta(
                            torch.tensor(0.4), torch.tensor(0.4)).sample())
                        idx  = torch.randperm(imgs.size(0), device=device)
                        imgs = lam * imgs + (1 - lam) * imgs[idx]
                        loss = lam * criterion(model(imgs), labels) + \
                               (1 - lam) * criterion(model(imgs), labels[idx])
                    else:
                        loss = criterion(model(imgs), labels)

                scaler.scale(loss).backward()
                if clip > 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optim)
                scaler.update()

                # OneCycle steps every batch
                if onecycle and scheduler is not None:
                    scheduler.step()
                train_loss += loss.item()

            if scheduler is not None and not onecycle:
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
            "class_names":      classes,   # alphabetical order used by ImageFolder
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

    classes_with_data = sum(
        1 for cls in project.classes
        if os.path.exists(os.path.join(CLS_DATA_DIR, str(project_id), cls)) and
           any(os.path.splitext(f)[1].lower() in IMG_EXTS
               for f in os.listdir(os.path.join(CLS_DATA_DIR, str(project_id), cls)))
    )
    if classes_with_data < 2:
        raise HTTPException(400, "Upload images for at least 2 classes in the Classification Dataset section before training")

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

    saved_results = json.loads(run.results_json) if run.results_json else {}
    saved_classes = saved_results.get("class_names") or sorted(project.classes)
    n_classes = len(saved_classes)
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


# ── Grad-CAM (show what the model looks at) ─────────────────────────────────────

@router.post("/runs/{run_id}/gradcam")
async def custom_gradcam(project_id: int, run_id: int,
                         file: UploadFile = File(...),
                         session: Session = Depends(get_session)):
    """
    Grad-CAM: run the trained custom CNN on an image and return a heatmap that
    highlights the regions the model used to make its decision.

    How it works (the teaching value):
      1. Forward pass; record the feature maps of the last conv layer.
      2. Backward pass from the predicted class score; record the gradients.
      3. Weight each feature map by the average of its gradients, sum, ReLU.
      4. That map = where the model "looked". Upscale and overlay on the photo.
    """
    import io, base64
    import torch
    import torch.nn.functional as F
    import numpy as np
    import torchvision.transforms as T
    from PIL import Image as PILImage

    run = session.get(CustomTrainingRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(404, "Run not found")
    if not run.model_path or not os.path.exists(run.model_path):
        raise HTTPException(404, "Train this run first, then try Grad-CAM.")

    project = session.get(Project, project_id)
    cfg     = session.get(CustomModelConfig, run.config_id)
    if not cfg:
        raise HTTPException(404, "Config not found")

    saved   = json.loads(run.results_json) if run.results_json else {}
    classes = saved.get("class_names") or sorted(project.classes)
    layers  = json.loads(cfg.layers_json)
    input_h, input_w = cfg.input_h, cfg.input_w

    # Build + load the model
    model = _build_torch_model(layers, input_h, input_w, len(classes))
    model.load_state_dict(torch.load(run.model_path, map_location="cpu", weights_only=False))
    model.eval()
    # Linear-probe (frozen-backbone) runs save params with requires_grad=False, so
    # no gradients would reach the conv layer. Re-enable grad on this throwaway
    # inference copy so Grad-CAM has gradients to work with.
    for p in model.parameters():
        p.requires_grad_(True)

    # Target = the LAST convolutional layer (standard Grad-CAM choice)
    target = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            target = m
    if target is None:
        raise HTTPException(400,
            "This network has no convolutional layer, so there is nothing spatial "
            "to visualise. Add at least one Conv layer.")

    # Read + preprocess the image
    raw = await file.read()
    try:
        pil = PILImage.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"Cannot read image: {exc}")
    tf = T.Compose([
        T.Resize((input_h, input_w)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    x = tf(pil).unsqueeze(0)

    # Capture the target layer's activation and its gradient. A tensor hook on
    # the activation is more reliable than register_full_backward_hook, which
    # silently fails to fire on some module layouts (e.g. MobileNet's last conv).
    store = {}
    def fwd(_m, _i, o):
        store["act"] = o
        if o.requires_grad:
            o.register_hook(lambda g: store.__setitem__("grad", g))
    h1 = target.register_forward_hook(fwd)

    try:
        out   = model(x)                         # [1, n_classes]
        probs = F.softmax(out, dim=1)[0]
        cls   = int(out.argmax(dim=1))
        model.zero_grad()
        out[0, cls].backward()
    finally:
        h1.remove()

    if "act" not in store or "grad" not in store:
        raise HTTPException(400,
            "Could not capture gradients for this model — Grad-CAM needs a "
            "convolutional feature map with gradient flow.")

    acts  = store["act"][0].detach()             # [C, h, w]
    grads = store["grad"][0].detach()            # [C, h, w]
    weights = grads.mean(dim=(1, 2))             # [C]
    cam = F.relu((weights[:, None, None] * acts).sum(0))   # [h, w]
    cam = cam.numpy().astype("float32")
    cam -= cam.min()
    cam /= (cam.max() + 1e-8)

    # Colourise + overlay on the original image
    import cv2
    orig = np.array(pil)                          # H×W×3 RGB
    H, W = orig.shape[:2]
    cam_rs  = cv2.resize(cam, (W, H))
    heat    = cv2.applyColorMap((cam_rs * 255).astype(np.uint8), cv2.COLORMAP_JET)  # BGR
    heat_rgb = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = (0.45 * heat_rgb + 0.55 * orig).clip(0, 255).astype(np.uint8)

    def _b64(rgb):
        ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buf.tobytes()).decode()

    preds = sorted(
        [{"class_name": classes[i] if i < len(classes) else f"cls{i}",
          "probability": round(float(probs[i]), 4)} for i in range(len(probs))],
        key=lambda d: d["probability"], reverse=True,
    )
    return {
        "overlay_b64":  _b64(overlay),
        "heatmap_b64":  _b64(heat_rgb),
        "original_b64": _b64(orig),
        "class_name":   classes[cls] if cls < len(classes) else f"cls{cls}",
        "probability":  round(float(probs[cls]), 4),
        "predictions":  preds[:5],
    }


