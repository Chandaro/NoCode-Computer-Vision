from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlmodel import Session, select
import os, shutil, json
from datetime import datetime

from database import get_session
from models import ExternalModel

router = APIRouter(prefix="/models/external", tags=["external-models"])

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "external_models")
os.makedirs(MODELS_DIR, exist_ok=True)


def describe_model(path: str) -> dict:
    """Return {kind, task} for a stored model path.

    kind: 'yolo' (a .pt file) or 'hf' (a HuggingFace model directory).
    task: 'detection' | 'classification' | 'unknown' (HF only, from config.json).
    """
    if path and os.path.isdir(path):
        task = "unknown"
        cfg_path = os.path.join(path, "config.json")
        if os.path.isfile(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    archs = " ".join(json.load(f).get("architectures", []) or [])
                low = archs.lower()
                if "objectdetection" in low:
                    task = "detection"
                elif "imageclassification" in low:
                    task = "classification"
            except Exception:
                pass
        return {"kind": "hf", "task": task}
    return {"kind": "yolo", "task": "detection"}


@router.get("")
def list_models(session: Session = Depends(get_session)):
    out = []
    for m in session.exec(select(ExternalModel)).all():
        d = m.model_dump()
        d.update(describe_model(m.model_path))
        out.append(d)
    return out


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


@router.post("/hf")
async def upload_hf_model(
    name: str = Form(...),
    files: list[UploadFile] = File(...),
    session: Session = Depends(get_session),
):
    """Import a HuggingFace model folder (config.json + model.safetensors +
    preprocessor_config.json, etc.). Saved as a directory; the model task
    (detection vs classification) is detected from config.json."""
    basenames = [os.path.basename((f.filename or "").replace("\\", "/")) for f in files]
    if "config.json" not in basenames:
        raise HTTPException(400, "A HuggingFace model folder must include config.json")
    has_weights = any(b.endswith((".safetensors", ".bin")) for b in basenames)
    if not has_weights:
        raise HTTPException(400, "No model weights found (.safetensors or .bin)")

    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in (name or "hf_model")) or "hf_model"
    dest_dir = os.path.join(MODELS_DIR, safe_name)
    base = dest_dir
    counter = 1
    while os.path.exists(dest_dir):
        dest_dir = f"{base}_{counter}"
        counter += 1
    os.makedirs(dest_dir, exist_ok=True)

    for f, bn in zip(files, basenames):
        if not bn:
            continue
        with open(os.path.join(dest_dir, bn), "wb") as out:
            shutil.copyfileobj(f.file, out)

    info = describe_model(dest_dir)
    if info["task"] == "unknown":
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise HTTPException(400, "Unsupported model: config.json is not an image "
                                 "classification or object-detection model")

    record = ExternalModel(
        name=name or safe_name,
        model_path=os.path.abspath(dest_dir),
        created_at=datetime.now().isoformat(timespec="seconds"),
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    out = record.model_dump()
    out.update(info)
    return out


@router.delete("/{model_id}")
def delete_model(model_id: int, session: Session = Depends(get_session)):
    record = session.get(ExternalModel, model_id)
    if not record:
        raise HTTPException(404, "Model not found")
    if record.model_path and os.path.isdir(record.model_path):
        shutil.rmtree(record.model_path, ignore_errors=True)
    elif record.model_path and os.path.exists(record.model_path):
        os.remove(record.model_path)
    session.delete(record)
    session.commit()
    return {"ok": True}
