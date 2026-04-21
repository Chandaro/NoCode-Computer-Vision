from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlmodel import Session, select
import os, shutil
from datetime import datetime

from database import get_session
from models import ExternalModel

router = APIRouter(prefix="/models/external", tags=["external-models"])

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "external_models")
os.makedirs(MODELS_DIR, exist_ok=True)


@router.get("")
def list_models(session: Session = Depends(get_session)):
    return session.exec(select(ExternalModel)).all()


@router.post("")
async def upload_model(
    name: str = Form(...),
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    if not (file.filename or "").endswith(".pt"):
        raise HTTPException(400, "Only .pt model files are supported")

    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in (file.filename or "model.pt"))
    dest = os.path.join(MODELS_DIR, safe_name)

    # Avoid overwriting — append suffix if name taken
    base, ext = os.path.splitext(dest)
    counter = 1
    while os.path.exists(dest):
        dest = f"{base}_{counter}{ext}"
        counter += 1

    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    record = ExternalModel(
        name=name or safe_name,
        model_path=os.path.abspath(dest),
        created_at=datetime.now().isoformat(timespec="seconds"),
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return record


@router.delete("/{model_id}")
def delete_model(model_id: int, session: Session = Depends(get_session)):
    record = session.get(ExternalModel, model_id)
    if not record:
        raise HTTPException(404, "Model not found")
    if os.path.exists(record.model_path):
        os.remove(record.model_path)
    session.delete(record)
    session.commit()
    return {"ok": True}
