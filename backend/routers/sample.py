"""
Sample dataset generator.

Lets anyone try NoCode CV end to end in a few clicks, without uploading
anything first. It builds a small synthetic image dataset (shapes) directly
on disk in the same place the normal upload flow stores classification
images, so every downstream feature (analytics, training, custom Conv
Builder) works on it with no special cases.

Two entry points:
  POST /api/sample/project        -> create a ready-to-train demo project
  POST /api/sample/fill/{id}      -> fill an existing project with samples
"""
from __future__ import annotations

import json
import math
import os
import random
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from database import get_session
from models import Project, CustomModelConfig

router = APIRouter(prefix="/sample", tags=["sample"])

# Same folder the classification upload endpoint writes to.
CLS_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "cls_uploads")
os.makedirs(CLS_DATA_DIR, exist_ok=True)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# The two classes we generate. Keep it to two so it trains fast on any GPU.
SHAPES = ["circle", "square"]

# 96x96 keeps training snappy and the shapes crisp.
CANVAS = 96

# A few saturated colours so the shape is always a clear subject.
_SHAPE_COLORS = [
    (214, 48, 49), (41, 98, 255), (38, 166, 91),
    (233, 138, 21), (142, 68, 173),
]


def _draw_sample(shape: str, size: int = CANVAS):
    """Draw one clean shape on a plain background. Returns a PIL image.

    Kept deliberately simple (single large shape, high contrast, only a small
    tilt) so a freshly built CNN learns it in a handful of epochs. The point of
    the sample is to show the whole flow working, not to be a hard benchmark.
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size), (245, 245, 245))
    draw = ImageDraw.Draw(img)

    fg = random.choice(_SHAPE_COLORS)
    r = size // 3                                     # big, consistent size
    # Keep the shape near the middle with only a small jitter. A simple CNN
    # (few pooling layers, no global pool) is not translation invariant, so
    # large random offsets would make the sample needlessly hard to learn.
    j = size // 8
    cx = size // 2 + random.randint(-j, j)
    cy = size // 2 + random.randint(-j, j)

    if shape == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fg)
    else:
        # square with only a slight tilt (about +-15 degrees)
        angle = random.uniform(-0.26, 0.26)
        pts = []
        for dx, dy in [(-r, -r), (r, -r), (r, r), (-r, r)]:
            rx = dx * math.cos(angle) - dy * math.sin(angle)
            ry = dx * math.sin(angle) + dy * math.cos(angle)
            pts.append((cx + rx, cy + ry))
        draw.polygon(pts, fill=fg)

    return img


def _write_samples(project_id: int, classes: list[str], per_class: int) -> dict:
    """Generate images for each class into cls_uploads/<project>/<class>/."""
    counts: dict[str, int] = {}
    for cls in classes:
        cls_dir = os.path.join(CLS_DATA_DIR, str(project_id), cls)
        os.makedirs(cls_dir, exist_ok=True)
        # clear old samples so re-running does not pile up duplicates
        for f in os.listdir(cls_dir):
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                try:
                    os.remove(os.path.join(cls_dir, f))
                except OSError:
                    pass
        shape = "circle" if "circle" in cls.lower() else "square"
        for _ in range(per_class):
            img = _draw_sample(shape)
            fname = f"{uuid.uuid4().hex}.jpg"
            img.save(os.path.join(cls_dir, fname), "JPEG", quality=90)
        counts[cls] = per_class
    return counts


def _default_layers(num_classes: int) -> list[dict]:
    """A small but capable CNN for the Conv Builder. Three conv/pool blocks give
    it enough capacity and spatial invariance to learn the sample shapes well,
    while still training in seconds. The backend appends the final classifier
    head automatically, so the trailing linear acts as a hidden layer."""
    return [
        {"id": "c1", "type": "conv2d",   "params": {"filters": 16, "kernel_size": 3, "stride": 1, "padding": 1}},
        {"id": "r1", "type": "relu",     "params": {}},
        {"id": "p1", "type": "maxpool2d", "params": {"kernel_size": 2, "stride": 2}},
        {"id": "c2", "type": "conv2d",   "params": {"filters": 32, "kernel_size": 3, "stride": 1, "padding": 1}},
        {"id": "r2", "type": "relu",     "params": {}},
        {"id": "p2", "type": "maxpool2d", "params": {"kernel_size": 2, "stride": 2}},
        {"id": "c3", "type": "conv2d",   "params": {"filters": 64, "kernel_size": 3, "stride": 1, "padding": 1}},
        {"id": "r3", "type": "relu",     "params": {}},
        {"id": "p3", "type": "maxpool2d", "params": {"kernel_size": 2, "stride": 2}},
        {"id": "f1", "type": "flatten",  "params": {}},
        {"id": "l1", "type": "linear",   "params": {"out_features": 128}},
    ]


@router.post("/project")
def create_sample_project(per_class: int = 120, session: Session = Depends(get_session)):
    """Create (or reset) a demo project that is ready to train immediately.

    Returns the project id and the custom-model config id so the UI can jump
    straight to the training page.
    """
    name = "Sample: Circles vs Squares"

    # reuse the demo project if it already exists
    project = session.exec(select(Project).where(Project.name == name)).first()
    if project is None:
        project = Project(name=name,
                          description="Auto-generated sample dataset. Try training instantly, no upload needed.")
        project.classes = SHAPES
        session.add(project)
        session.commit()
        session.refresh(project)
    else:
        project.classes = SHAPES
        session.add(project)
        session.commit()

    counts = _write_samples(project.id, SHAPES, per_class)

    # one ready-to-go model config per demo project
    cfg = session.exec(
        select(CustomModelConfig).where(CustomModelConfig.project_id == project.id)
    ).first()
    if cfg is None:
        cfg = CustomModelConfig(
            project_id=project.id,
            name="Sample CNN",
            layers_json=json.dumps(_default_layers(len(SHAPES))),
            input_h=64,
            input_w=64,
            created_at=datetime.now().isoformat(),
        )
        session.add(cfg)
        session.commit()
        session.refresh(cfg)
    else:
        # refresh the demo config so a re-run picks up the current default net
        cfg.layers_json = json.dumps(_default_layers(len(SHAPES)))
        cfg.input_h = 64
        cfg.input_w = 64
        session.add(cfg)
        session.commit()

    return {
        "project_id": project.id,
        "config_id": cfg.id,
        "classes": SHAPES,
        "counts": counts,
    }


@router.post("/fill/{project_id}")
def fill_project(project_id: int, per_class: int = 120, session: Session = Depends(get_session)):
    """Fill an existing project with sample images.

    Uses the project's own classes when it has at least two, otherwise sets it
    up with the default shapes. Handy for the 'Load sample images' button.
    """
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    classes = [c for c in project.classes if c]
    if len(classes) < 2:
        classes = SHAPES
        project.classes = classes
        session.add(project)
        session.commit()

    counts = _write_samples(project.id, classes, per_class)
    return {"project_id": project.id, "classes": classes, "counts": counts}
