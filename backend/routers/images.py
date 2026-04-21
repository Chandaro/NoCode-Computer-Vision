from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from sqlmodel import Session, select
from typing import List
from pydantic import BaseModel
import shutil, uuid, os, hashlib
from PIL import Image as PILImage

from database import get_session
from models import Image, Project, Annotation

router = APIRouter(prefix="/projects/{project_id}/images", tags=["images"])

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


class ImageOut(BaseModel):
    id: int
    filename: str
    original_name: str
    annotated: bool
    width: int = 0
    height: int = 0
    color_space: str = "RGB"
    is_corrupt: bool = False
    file_size: int = 0

    class Config:
        from_attributes = True


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _inspect_image(path: str):
    """Return (width, height, channels, color_space, is_corrupt)."""
    try:
        pil = PILImage.open(path)
        pil.verify()          # raises on corrupt
        pil = PILImage.open(path)   # must reopen after verify
        w, h = pil.size
        mode = pil.mode
        channels   = 1 if mode in ("L", "LA") else 3
        color_space = "Grayscale" if channels == 1 else "RGB"
        return w, h, channels, color_space, False
    except Exception:
        return 0, 0, 0, "unknown", True


@router.get("", response_model=List[ImageOut])
def list_images(project_id: int, session: Session = Depends(get_session)):
    images = session.exec(select(Image).where(Image.project_id == project_id)).all()
    result = []
    for img in images:
        anns = session.exec(select(Annotation).where(Annotation.image_id == img.id)).all()
        result.append(ImageOut(
            id=img.id, filename=img.filename, original_name=img.original_name,
            annotated=len(anns) > 0, width=img.width, height=img.height,
            color_space=img.color_space, is_corrupt=img.is_corrupt, file_size=img.file_size,
        ))
    return result


@router.post("", response_model=List[ImageOut])
async def upload_images(
    project_id: int,
    files: List[UploadFile] = File(...),
    session: Session = Depends(get_session),
):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    saved = []
    skipped_duplicates = 0

    for file in files:
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            continue

        unique_name = f"{uuid.uuid4().hex}{ext}"
        dest = os.path.join(UPLOAD_DIR, unique_name)

        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Duplicate check
        md5 = _md5(dest)
        existing = session.exec(
            select(Image).where(Image.project_id == project_id, Image.md5_hash == md5)
        ).first()
        if existing:
            os.remove(dest)
            skipped_duplicates += 1
            continue

        file_size = os.path.getsize(dest)
        w, h, channels, color_space, is_corrupt = _inspect_image(dest)

        img_record = Image(
            project_id=project_id, filename=unique_name, original_name=file.filename or unique_name,
            md5_hash=md5, width=w, height=h, channels=channels,
            color_space=color_space, is_corrupt=is_corrupt, file_size=file_size,
        )
        session.add(img_record)
        session.commit()
        session.refresh(img_record)
        saved.append(ImageOut(
            id=img_record.id, filename=img_record.filename, original_name=img_record.original_name,
            annotated=False, width=w, height=h, color_space=color_space,
            is_corrupt=is_corrupt, file_size=file_size,
        ))

    return saved


@router.get("/{image_id}/file")
def get_image_file(project_id: int, image_id: int, session: Session = Depends(get_session)):
    img = session.get(Image, image_id)
    if not img or img.project_id != project_id:
        raise HTTPException(404, "Image not found")
    path = os.path.join(UPLOAD_DIR, img.filename)
    if not os.path.exists(path):
        raise HTTPException(404, "File not found")
    return FileResponse(path)


@router.delete("/{image_id}")
def delete_image(project_id: int, image_id: int, session: Session = Depends(get_session)):
    img = session.get(Image, image_id)
    if not img or img.project_id != project_id:
        raise HTTPException(404, "Image not found")
    path = os.path.join(UPLOAD_DIR, img.filename)
    anns = session.exec(select(Annotation).where(Annotation.image_id == image_id)).all()
    for ann in anns:
        session.delete(ann)
    session.delete(img)
    session.commit()
    if os.path.exists(path):
        os.remove(path)
    return {"ok": True}


class BulkDeleteBody(BaseModel):
    ids: List[int]


@router.delete("")
def bulk_delete_images(
    project_id: int,
    body: BulkDeleteBody,
    session: Session = Depends(get_session),
):
    deleted = 0
    for image_id in body.ids:
        img = session.get(Image, image_id)
        if not img or img.project_id != project_id:
            continue
        path = os.path.join(UPLOAD_DIR, img.filename)
        for ann in session.exec(select(Annotation).where(Annotation.image_id == image_id)).all():
            session.delete(ann)
        session.delete(img)
        if os.path.exists(path):
            os.remove(path)
        deleted += 1
    session.commit()
    return {"deleted": deleted}
