from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from typing import List
from pydantic import BaseModel
import json

from database import get_session
from models import Annotation, Image

router = APIRouter(prefix="/projects/{project_id}/images/{image_id}/annotations", tags=["annotations"])


class AnnIn(BaseModel):
    class_id: int
    shape_type: str = "bbox"          # bbox | polygon | point
    x_center: float = 0.0
    y_center: float = 0.0
    width: float = 0.0
    height: float = 0.0
    points: List[List[float]] = []    # [[x1,y1],[x2,y2],...] for polygon/point


class AnnOut(AnnIn):
    id: int

    class Config:
        from_attributes = True


@router.get("", response_model=List[AnnOut])
def get_annotations(project_id: int, image_id: int, session: Session = Depends(get_session)):
    img = session.get(Image, image_id)
    if not img or img.project_id != project_id:
        raise HTTPException(404, "Image not found")
    anns = session.exec(select(Annotation).where(Annotation.image_id == image_id)).all()
    return [
        AnnOut(
            id=a.id, class_id=a.class_id, shape_type=a.shape_type,
            x_center=a.x_center, y_center=a.y_center, width=a.width, height=a.height,
            points=json.loads(a.points_json or "[]"),
        )
        for a in anns
    ]


@router.post("", response_model=List[AnnOut])
def save_annotations(project_id: int, image_id: int, shapes: List[AnnIn], session: Session = Depends(get_session)):
    img = session.get(Image, image_id)
    if not img or img.project_id != project_id:
        raise HTTPException(404, "Image not found")

    for a in session.exec(select(Annotation).where(Annotation.image_id == image_id)).all():
        session.delete(a)

    saved = []
    for s in shapes:
        ann = Annotation(
            image_id=image_id, class_id=s.class_id, shape_type=s.shape_type,
            x_center=s.x_center, y_center=s.y_center, width=s.width, height=s.height,
            points_json=json.dumps(s.points),
        )
        session.add(ann)
        saved.append(ann)
    session.commit()
    for ann in saved:
        session.refresh(ann)
    return [AnnOut(
        id=ann.id, class_id=ann.class_id, shape_type=ann.shape_type,
        x_center=ann.x_center, y_center=ann.y_center, width=ann.width, height=ann.height,
        points=json.loads(ann.points_json),
    ) for ann in saved]
