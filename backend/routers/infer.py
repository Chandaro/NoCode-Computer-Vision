from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Form
from sqlmodel import Session
import tempfile, os, shutil
from pathlib import Path

from database import get_session
from models import TrainingRun, ClassificationRun, Project
from models import Image as ImageModel

router = APIRouter(tags=["inference"])

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")


# ─── Detection inference ──────────────────────────────────────────────────────
@router.post("/projects/{project_id}/training/runs/{run_id}/infer")
async def detection_infer(
    project_id: int,
    run_id: int,
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
    session: Session = Depends(get_session),
):
    run = session.get(TrainingRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(404, "Run not found")
    if run.status != "done":
        raise HTTPException(400, "Run has not completed")
    if not run.model_path or not os.path.exists(run.model_path):
        raise HTTPException(404, "Model weights not found")

    project = session.get(Project, project_id)
    class_names = project.classes if project else []

    suffix = Path(file.filename or "img.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        import torch
        from ultralytics import YOLO
        device = "0" if torch.cuda.is_available() else "cpu"
        model = YOLO(run.model_path)
        try:
            results = model.predict(tmp_path, conf=conf, iou=iou, verbose=False, device=device)
        except RuntimeError:
            results = model.predict(tmp_path, conf=conf, iou=iou, verbose=False, device="cpu")
        r = results[0]
        ih, iw = r.orig_shape

        detections = []
        for box in r.boxes:
            x1n, y1n, x2n, y2n = box.xyxyn[0].tolist()
            cls_id = int(box.cls[0])
            detections.append({
                "x": round(x1n, 4), "y": round(y1n, 4),
                "w": round(x2n - x1n, 4), "h": round(y2n - y1n, 4),
                "conf": round(float(box.conf[0]), 4),
                "class_id": cls_id,
                "class_name": class_names[cls_id] if cls_id < len(class_names) else f"cls{cls_id}",
            })
        return {"detections": detections, "image_w": iw, "image_h": ih, "count": len(detections)}
    finally:
        os.unlink(tmp_path)


# ─── Classification inference ─────────────────────────────────────────────────
@router.post("/projects/{project_id}/classification/runs/{run_id}/infer")
async def classification_infer(
    project_id: int,
    run_id: int,
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    run = session.get(ClassificationRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(404, "Run not found")
    if run.status != "done":
        raise HTTPException(400, "Run has not completed")
    if not run.model_path or not os.path.exists(run.model_path):
        raise HTTPException(404, "Model weights not found")

    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    classes = project.classes

    suffix = Path(file.filename or "img.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        import torch
        import torchvision.transforms as T
        import torchvision.models as M
        import torch.nn as nn
        from PIL import Image as PILImage

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_fn_map = {
            "resnet18":           M.resnet18,
            "resnet50":           M.resnet50,
            "mobilenet_v3_small": M.mobilenet_v3_small,
            "efficientnet_b0":    M.efficientnet_b0,
        }
        model = model_fn_map.get(run.base_model, M.resnet18)(weights=None)
        n_classes = len(classes)
        if hasattr(model, "fc"):
            model.fc = nn.Linear(model.fc.in_features, n_classes)
        elif hasattr(model, "classifier"):
            in_feat = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_feat, n_classes)

        model.load_state_dict(torch.load(run.model_path, map_location=device, weights_only=False))
        model = model.to(device).eval()

        tf = T.Compose([
            T.Resize((run.imgsz, run.imgsz)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        img = PILImage.open(tmp_path).convert("RGB")
        tensor = tf(img).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0].tolist()

        preds = sorted(
            [{"class_id": i,
              "class_name": classes[i] if i < len(classes) else f"cls{i}",
              "probability": round(p, 4)}
             for i, p in enumerate(probs)],
            key=lambda x: x["probability"], reverse=True,
        )
        return {"predictions": preds, "top1": preds[0] if preds else None, "top5": preds[:5]}
    finally:
        os.unlink(tmp_path)


# ─── Auto-annotate ────────────────────────────────────────────────────────────
@router.post("/projects/{project_id}/images/{image_id}/auto-annotate")
async def auto_annotate(
    project_id: int,
    image_id: int,
    run_id: int,
    conf: float = 0.25,
    session: Session = Depends(get_session),
):
    """Run detection model on a stored image; returns suggested annotations (not saved)."""
    img_rec = session.get(ImageModel, image_id)
    if not img_rec or img_rec.project_id != project_id:
        raise HTTPException(404, "Image not found")

    run = session.get(TrainingRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(404, "Run not found")
    if run.status != "done":
        raise HTTPException(400, "Run not complete")
    if not run.model_path or not os.path.exists(run.model_path):
        raise HTTPException(404, "Model not found")

    img_path = os.path.join(UPLOAD_DIR, img_rec.filename)
    if not os.path.exists(img_path):
        raise HTTPException(404, "Image file not found")

    import torch
    from ultralytics import YOLO
    device = "0" if torch.cuda.is_available() else "cpu"
    model = YOLO(run.model_path)
    try:
        results = model.predict(img_path, conf=conf, verbose=False, device=device)
    except RuntimeError:
        results = model.predict(img_path, conf=conf, verbose=False, device="cpu")
    r = results[0]

    annotations = []
    for box in r.boxes:
        x1n, y1n, x2n, y2n = box.xyxyn[0].tolist()
        cls_id = int(box.cls[0])
        annotations.append({
            "class_id": cls_id,
            "shape_type": "bbox",
            "x_center": round((x1n + x2n) / 2, 6),
            "y_center": round((y1n + y2n) / 2, 6),
            "width":    round(x2n - x1n, 6),
            "height":   round(y2n - y1n, 6),
            "points":   [],
            "conf":     round(float(box.conf[0]), 4),
        })
    return {"annotations": annotations, "count": len(annotations)}
