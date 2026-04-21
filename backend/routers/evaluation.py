from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlmodel import Session
import os, json, csv

from database import get_session
from models import TrainingRun, Project

router = APIRouter(prefix="/projects/{project_id}/training/runs/{run_id}/eval", tags=["evaluation"])

# Plots that YOLO writes to the run_dir
KNOWN_PLOTS = [
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "results.png",
    "BoxF1_curve.png",
    "BoxPR_curve.png",
    "BoxP_curve.png",
    "BoxR_curve.png",
    "labels.jpg",
    "labels_correlogram.jpg",
    "val_batch0_labels.jpg",
    "val_batch0_pred.jpg",
    "val_batch1_labels.jpg",
    "val_batch1_pred.jpg",
    "val_batch2_labels.jpg",
    "val_batch2_pred.jpg",
]


def _get_run(project_id: int, run_id: int, session: Session) -> TrainingRun:
    run = session.get(TrainingRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(404, "Run not found")
    if run.status != "done":
        raise HTTPException(400, "Run has not completed yet")
    return run


def _resolve_run_dir(run: TrainingRun) -> str:
    """Return the directory that contains plot PNGs."""
    if run.run_dir and os.path.isdir(run.run_dir):
        return run.run_dir
    # Fallback: derive from run id
    runs_dir = os.path.join(os.path.dirname(__file__), "..", "runs")
    candidate = os.path.join(runs_dir, f"train_{run.id}", "weights")
    return candidate if os.path.isdir(candidate) else ""


def _parse_results_csv(run_dir: str):
    """Parse the last row of results.csv into a dict of metric→value."""
    csv_path = os.path.join(run_dir, "results.csv")
    if not os.path.exists(csv_path):
        return {}
    try:
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return {}
        last = rows[-1]
        return {k.strip(): float(v.strip()) for k, v in last.items() if v.strip()}
    except Exception:
        return {}


@router.get("")
def get_eval_metadata(project_id: int, run_id: int, session: Session = Depends(get_session)):
    run     = _get_run(project_id, run_id, session)
    project = session.get(Project, project_id)
    run_dir = _resolve_run_dir(run)

    available_plots = []
    if run_dir:
        for fname in KNOWN_PLOTS:
            if os.path.exists(os.path.join(run_dir, fname)):
                available_plots.append(fname)
        # Also catch any additional val_batch PNGs dynamically
        for fname in os.listdir(run_dir):
            if fname.startswith("val_batch") and fname not in available_plots:
                available_plots.append(fname)

    csv_metrics = _parse_results_csv(run_dir) if run_dir else {}

    results = json.loads(run.results_json)
    per_class = results.get("per_class", {})

    return {
        "run_id":         run_id,
        "model_base":     run.model_base,
        "epochs":         run.epochs,
        "available_plots": available_plots,
        "csv_metrics":    csv_metrics,
        "overall":        {k: v for k, v in results.items() if k != "per_class"},
        "per_class":      per_class,
        "classes":        project.classes if project else [],
    }


@router.get("/plots/{filename}")
def get_plot(project_id: int, run_id: int, filename: str,
             session: Session = Depends(get_session)):
    run     = _get_run(project_id, run_id, session)
    run_dir = _resolve_run_dir(run)
    if not run_dir:
        raise HTTPException(404, "Run directory not found")

    # Safety: prevent path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(400, "Invalid filename")

    path = os.path.join(run_dir, filename)
    if not os.path.exists(path):
        raise HTTPException(404, f"Plot {filename} not found")

    media = "image/png" if filename.endswith(".png") else "image/jpeg"
    return FileResponse(path, media_type=media)
