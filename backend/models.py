from __future__ import annotations
# PEP 563: store all annotations as strings at class-definition time so that
# Python 3.14 (PEP 649 lazy __annotations__) doesn't break Pydantic's field
# discovery inside SQLModel's metaclass __new__.
#
# SQLAlchemy compat: with string annotations, SQLModel passes the raw string
# (e.g. "List['Image']") to SQLAlchemy's class-registry resolver which cannot
# resolve generic wrapper strings.  Fix: supply sa_relationship explicitly so
# SQLModel skips annotation-based relationship setup and SQLAlchemy resolves
# the plain class-name string ("Image", "Project", …) from its own registry.

from typing import Optional, List
from sqlalchemy.orm import relationship
from sqlmodel import SQLModel, Field, Relationship
import json


class Project(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    description: str = ""
    classes_json: str = "[]"
    images: List["Image"] = Relationship(
        sa_relationship=relationship("Image", back_populates="project")
    )

    @property
    def classes(self) -> List[str]:
        return json.loads(self.classes_json)

    @classes.setter
    def classes(self, value: List[str]):
        self.classes_json = json.dumps(value)


class Image(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id")
    filename: str
    original_name: str
    # Validation / stats fields (populated on upload)
    md5_hash: str = Field(default="")
    width: int = Field(default=0)
    height: int = Field(default=0)
    channels: int = Field(default=3)
    color_space: str = Field(default="RGB")   # "RGB" | "Grayscale"
    is_corrupt: bool = Field(default=False)
    file_size: int = Field(default=0)
    project: Optional["Project"] = Relationship(
        sa_relationship=relationship("Project", back_populates="images", uselist=False)
    )
    annotations: List["Annotation"] = Relationship(
        sa_relationship=relationship("Annotation", back_populates="image")
    )


class Annotation(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    image_id: int = Field(foreign_key="image.id")
    class_id: int
    shape_type: str = Field(default="bbox")   # bbox | polygon | point
    x_center: float = Field(default=0.0)
    y_center: float = Field(default=0.0)
    width: float = Field(default=0.0)
    height: float = Field(default=0.0)
    points_json: str = Field(default="[]")
    image: Optional["Image"] = Relationship(
        sa_relationship=relationship("Image", back_populates="annotations", uselist=False)
    )


class TrainingRun(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id")
    status: str = "pending"
    epochs: int = 50
    imgsz: int = 640
    batch: int = 16
    model_base: str = "yolov8n.pt"
    model_path: str = ""
    results_json: str = "{}"
    created_at: str = ""
    # Extended fields
    run_dir: str = ""
    onnx_path: str = ""
    aug_config_json: str = "{}"


class ExternalModel(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    model_path: str
    created_at: str = ""


class ClassificationRun(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id")
    status: str = "pending"     # pending | running | done | failed
    epochs: int = 10
    imgsz: int = 224
    batch: int = 32
    base_model: str = "resnet18"   # resnet18|resnet50|mobilenet_v3_small|efficientnet_b0
    lr: float = 0.001
    freeze_backbone: bool = True
    model_path: str = ""
    run_dir: str = ""
    results_json: str = "{}"       # top1_acc, val_loss, per_class metrics
    created_at: str = ""


# ── Conv Builder models ────────────────────────────────────────────────────────

class CustomModelConfig(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id")
    name: str = Field(default="My Model")
    layers_json: str = Field(default="[]")
    input_h: int = Field(default=64)
    input_w: int = Field(default=64)
    created_at: str = Field(default="")


class CustomTrainingRun(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    config_id: int = Field(default=0)
    project_id: int = Field(foreign_key="project.id")
    status: str = Field(default="pending")
    epochs: int = Field(default=20)
    batch: int = Field(default=32)
    lr: float = Field(default=0.001)
    model_path: str = Field(default="")
    run_dir: str = Field(default="")
    results_json: str = Field(default="{}")
    created_at: str = Field(default="")


# Rebuild Pydantic models so forward-reference annotations are resolved now
# that every class in this module is fully defined.
Project.model_rebuild()
Image.model_rebuild()
Annotation.model_rebuild()
