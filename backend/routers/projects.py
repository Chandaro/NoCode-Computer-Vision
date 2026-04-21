from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from typing import List
from pydantic import BaseModel
import json, os, shutil

from database import get_session
from models import Project, Image, Annotation

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")

router = APIRouter(prefix="/projects", tags=["projects"])


class ProjectCreate(BaseModel):
    name: str
    description: str = ""
    classes: List[str] = []


class ProjectOut(BaseModel):
    id: int
    name: str
    description: str
    classes: List[str]

    class Config:
        from_attributes = True


@router.get("", response_model=List[ProjectOut])
def list_projects(session: Session = Depends(get_session)):
    projects = session.exec(select(Project)).all()
    return [ProjectOut(id=p.id, name=p.name, description=p.description, classes=p.classes) for p in projects]


@router.post("", response_model=ProjectOut)
def create_project(data: ProjectCreate, session: Session = Depends(get_session)):
    project = Project(name=data.name, description=data.description, classes_json=json.dumps(data.classes))
    session.add(project)
    session.commit()
    session.refresh(project)
    return ProjectOut(id=project.id, name=project.name, description=project.description, classes=project.classes)


@router.get("/{project_id}", response_model=ProjectOut)
def get_project(project_id: int, session: Session = Depends(get_session)):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    return ProjectOut(id=project.id, name=project.name, description=project.description, classes=project.classes)


@router.put("/{project_id}", response_model=ProjectOut)
def update_project(project_id: int, data: ProjectCreate, session: Session = Depends(get_session)):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    project.name = data.name
    project.description = data.description
    project.classes_json = json.dumps(data.classes)
    session.add(project)
    session.commit()
    session.refresh(project)
    return ProjectOut(id=project.id, name=project.name, description=project.description, classes=project.classes)


@router.delete("/{project_id}")
def delete_project(project_id: int, session: Session = Depends(get_session)):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    images = session.exec(select(Image).where(Image.project_id == project_id)).all()
    filenames = [img.filename for img in images]
    for img in images:
        for ann in session.exec(select(Annotation).where(Annotation.image_id == img.id)).all():
            session.delete(ann)
        session.delete(img)
    session.delete(project)
    session.commit()
    for filename in filenames:
        path = os.path.join(UPLOAD_DIR, filename)
        if os.path.exists(path):
            os.remove(path)
    return {"ok": True}
