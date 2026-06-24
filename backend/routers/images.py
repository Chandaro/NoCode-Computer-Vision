from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from sqlmodel import Session, select
from typing import List
from pydantic import BaseModel
import shutil, uuid, os, hashlib, json, math
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


def _yaml_class_names(raw: str) -> list[str] | None:
    """Extract class names from a Roboflow/Ultralytics data.yaml `names:` field.
    Handles both list form (names: ['a','b']) and dict form (names: {0: a, 1: b})."""
    try:
        import yaml
        data = yaml.safe_load(raw) or {}
        names = data.get("names")
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
        if isinstance(names, list):
            return [str(n) for n in names]
    except Exception:
        pass
    return None


@router.post("/derive-classes")
def derive_classes(project_id: int, session: Session = Depends(get_session)):
    """Populate class labels from existing annotations when none are defined.

    Scans all annotations for the highest class id and fills any missing names
    with placeholders (class0, class1, …), preserving names already set."""
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    from sqlalchemy import func
    # Join annotations to images by project and take the highest class id in one
    # query (avoids SQLite's ~999-variable IN limit on large datasets).
    max_cls_id = session.exec(
        select(func.max(Annotation.class_id))
        .join(Image, Annotation.image_id == Image.id)
        .where(Image.project_id == project_id)
    ).one()
    if max_cls_id is None:
        raise HTTPException(400, "No annotations found to derive classes from")

    existing = list(project.classes)
    names = [existing[i] if i < len(existing) else f"class{i}" for i in range(max_cls_id + 1)]
    project.classes_json = json.dumps(names)
    session.add(project)
    session.commit()
    return {"classes": names, "count": len(names)}


@router.post("/extract-frames")
async def extract_frames(
    project_id: int,
    file: UploadFile = File(...),
    every_sec: float = Form(1.0),
    max_frames: int = Form(300),
    session: Session = Depends(get_session),
):
    """Extract frames from an uploaded video into the project's image dataset.

    Grabs one frame every `every_sec` seconds (up to `max_frames`), skips exact
    duplicates (static footage), and adds each as a normal project image."""
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    import cv2
    suffix = os.path.splitext(file.filename or "video.mp4")[1].lower() or ".mp4"
    tmp = os.path.join(UPLOAD_DIR, f"_vidtmp_{uuid.uuid4().hex}{suffix}")
    with open(tmp, "wb") as f:
        shutil.copyfileobj(file.file, f)

    imported = 0
    skipped = 0
    scanned = 0
    base = os.path.splitext(os.path.basename(file.filename or "video"))[0]
    try:
        cap = cv2.VideoCapture(tmp)
        if not cap.isOpened():
            raise HTTPException(400, "Could not read this video file")
        fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(round(fps * max(0.1, every_sec))))   # frames between captures
        cap_lim = max(1, min(int(max_frames), 5000))

        while imported < cap_lim:
            ok, frame = cap.read()
            if not ok:
                break
            if scanned % step == 0:
                unique = f"{uuid.uuid4().hex}.jpg"
                dest = os.path.join(UPLOAD_DIR, unique)
                cv2.imwrite(dest, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                md5 = _md5(dest)
                existing = session.exec(
                    select(Image).where(Image.project_id == project_id, Image.md5_hash == md5)
                ).first()
                if existing:
                    os.remove(dest)
                    skipped += 1
                else:
                    w, h, channels, color_space, is_corrupt = _inspect_image(dest)
                    session.add(Image(
                        project_id=project_id, filename=unique,
                        original_name=f"{base}_f{scanned:06d}.jpg",
                        md5_hash=md5, width=w, height=h, channels=channels,
                        color_space=color_space, is_corrupt=is_corrupt,
                        file_size=os.path.getsize(dest)))
                    session.commit()
                    imported += 1
            scanned += 1
        cap.release()
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    return {"imported": imported, "skipped_duplicates": skipped, "frames_scanned": scanned}


@router.post("/import-yolo")
async def import_yolo_dataset(
    project_id: int,
    files: List[UploadFile] = File(...),
    session: Session = Depends(get_session),
):
    """
    Import a YOLO-format dataset in one step.

    Upload images and their matching .txt label files together.
    Each label line: class_id x_center y_center width height  (normalized 0–1)
    Optionally include classes.txt to auto-populate the project's class list.
    """
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    image_files: dict[str, UploadFile] = {}
    label_files: dict[str, str] = {}
    classes_list: list[str] | None = None
    yaml_raw: str | None = None
    names_raw: str | None = None

    for file in files:
        # Strip subdirectory prefix (e.g. "images/dog.jpg" → "dog.jpg")
        name = os.path.basename(file.filename or "")
        stem = os.path.splitext(name)[0].lower()   # lowercase for consistent matching
        ext  = os.path.splitext(name)[1].lower()

        if ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            image_files[stem] = file
        elif ext in (".yaml", ".yml"):
            yaml_raw = (await file.read()).decode("utf-8", errors="replace")
        elif ext == ".names" or name.lower() == "obj.names":
            names_raw = (await file.read()).decode("utf-8", errors="replace")
        elif ext == ".txt":
            raw = (await file.read()).decode("utf-8", errors="replace")
            if stem == "classes":
                classes_list = [l.strip() for l in raw.splitlines() if l.strip()]
            else:
                label_files[stem] = raw

    # Resolve class names: classes.txt > data.yaml (Roboflow) > obj.names
    if classes_list is None and yaml_raw:
        classes_list = _yaml_class_names(yaml_raw)
    if classes_list is None and names_raw:
        classes_list = [l.strip() for l in names_raw.splitlines() if l.strip()]

    if classes_list:
        project.classes_json = json.dumps(classes_list)
        session.add(project)
        session.commit()

    imported = 0
    annotated_count = 0
    skipped = 0
    max_cls_id = -1

    for stem, img_file in image_files.items():
        ext = os.path.splitext(img_file.filename or "")[1].lower()
        unique_name = f"{uuid.uuid4().hex}{ext}"
        dest = os.path.join(UPLOAD_DIR, unique_name)

        img_file.file.seek(0)
        with open(dest, "wb") as f:
            shutil.copyfileobj(img_file.file, f)

        md5 = _md5(dest)
        existing = session.exec(
            select(Image).where(Image.project_id == project_id, Image.md5_hash == md5)
        ).first()
        if existing:
            os.remove(dest)
            skipped += 1
            continue

        file_size = os.path.getsize(dest)
        w, h, channels, color_space, is_corrupt = _inspect_image(dest)

        img_record = Image(
            project_id=project_id,
            filename=unique_name,
            original_name=img_file.filename or unique_name,
            md5_hash=md5, width=w, height=h,
            channels=channels, color_space=color_space,
            is_corrupt=is_corrupt, file_size=file_size,
        )
        session.add(img_record)
        session.commit()
        session.refresh(img_record)
        imported += 1

        # Stem is already lowercased — direct lookup
        label_content = label_files.get(stem)
        if label_content:
            ann_count = 0
            for line in label_content.splitlines():
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                try:
                    cls_id = int(parts[0])
                    if cls_id > max_cls_id:
                        max_cls_id = cls_id
                    coords = [float(v) for v in parts[1:]]

                    if len(coords) == 4:
                        # Standard bbox: cx cy w h
                        ann = Annotation(
                            image_id=img_record.id,
                            class_id=cls_id,
                            shape_type="bbox",
                            x_center=coords[0],
                            y_center=coords[1],
                            width=coords[2],
                            height=coords[3],
                        )
                    elif len(coords) >= 6 and len(coords) % 2 == 0:
                        # Polygon/segmentation: x1 y1 x2 y2 ...
                        pts = [[coords[i], coords[i + 1]] for i in range(0, len(coords), 2)]
                        # Compute bbox from polygon for x_center/y_center/width/height
                        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
                        ann = Annotation(
                            image_id=img_record.id,
                            class_id=cls_id,
                            shape_type="polygon",
                            x_center=(x1 + x2) / 2,
                            y_center=(y1 + y2) / 2,
                            width=x2 - x1,
                            height=y2 - y1,
                            points_json=json.dumps(pts),
                        )
                    else:
                        continue

                    session.add(ann)
                    ann_count += 1
                except (ValueError, ZeroDivisionError):
                    continue
            if ann_count:
                annotated_count += 1
            session.commit()

    # Fallback: no names file but annotations exist → derive placeholder names
    # from the highest class id seen, so the labels are defined and trainable.
    derived = False
    if not project.classes and max_cls_id >= 0:
        project.classes_json = json.dumps([f"class{i}" for i in range(max_cls_id + 1)])
        session.add(project)
        session.commit()
        derived = True

    return {
        "imported": imported,
        "annotated": annotated_count,
        "skipped_duplicates": skipped,
        "classes_updated": bool(classes_list) or derived,
    }


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
