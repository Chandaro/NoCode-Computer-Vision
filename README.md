# NoCode CV Trainer

**Train computer vision models without writing a single line of code.**

NoCode CV Trainer is a desktop application for Windows that lets you upload images, draw annotations, and train state-of-the-art object detection and image classification models — all through a clean point-and-click interface.

---

## What You Can Do

| Feature | Description |
|---|---|
| **Annotation Studio** | Draw bounding boxes, polygons, and point markers on your images |
| **Object Detection** | Train YOLOv8 models to detect and locate objects in images |
| **Image Classification** | Train ResNet, MobileNet, or EfficientNet classifiers |
| **Live Inference** | Run your trained model on new images instantly |
| **Analytics** | Visualize label distributions and dataset statistics |
| **Evaluation** | Review model performance with precision/recall metrics |
| **Export** | Export your dataset in YOLO or COCO format |

---

## Runs On Your Machine — No Cloud Required

> **Your hardware does all the work.** NoCode CV Trainer runs entirely on your own computer — no internet connection needed during training, no data uploaded anywhere, no subscription, no API calls to external services.
>
> Training deep learning models is computationally intensive. Here's what to expect:
>
> | Hardware | Training Speed | Recommendation |
> |---|---|---|
> | **NVIDIA GPU (CUDA)** | Fast — minutes per epoch | Ideal for serious training |
> | **CPU only** | Slow — can take hours per epoch | Fine for small datasets / testing |
> | **AMD / Intel GPU** | Not accelerated | Falls back to CPU |
>
> The more images and epochs you train, the more CPU/GPU and RAM your machine will use. This is normal — close other heavy applications while training for best performance.
>
> On first use, YOLOv8 will download its base model weights (~130 MB) from the internet once. After that, everything runs offline.

---

## Requirements

| Requirement | Status | Notes |
|---|---|---|
| **Python 3.9 or newer** | Required | [python.org/downloads](https://www.python.org/downloads/) |
| **Node.js** | Optional | Not needed — frontend is pre-built |
| **NVIDIA GPU + CUDA** | Optional | CPU works fine; GPU trains much faster |

> **Just need Python.** Everything else is handled automatically by the installer.

---

## Quick Start (Non-Technical)

1. **Double-click** `Install NoCode CV.bat`
   - This creates a virtual environment and installs all dependencies
   - Takes 2–5 minutes on first run
   - You only need to do this once

2. **Double-click** `NoCode CV.bat` to launch the app

3. Open your browser and go to **http://localhost:8000**

That's it. No terminal, no commands, no configuration.

---

## Developer Setup (Manual)

If you prefer to set things up yourself:

**1. Clone the repository**
```bash
git clone https://github.com/Chandaro/NoCode_CV.git
cd NoCode_CV
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install Python dependencies**
```bash
pip install -r backend/requirements.txt
```

**4. (Optional) Rebuild the frontend**

The pre-built frontend is already included in `frontend/dist/`. Only do this if you modify the UI:
```bash
cd frontend
npm install
npm run build
cd ..
```

**5. Start the server**
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** in your browser.

---

## How It Works

```
frontend/dist/         ← Pre-built React UI (served by FastAPI)
backend/
  main.py              ← FastAPI app, serves UI + API
  routers/
    images.py          ← Upload & manage images
    training.py        ← YOLOv8 training with live log streaming (SSE)
    classification.py  ← PyTorch classification training
    infer.py           ← Run inference on trained models
    analytics.py       ← Dataset statistics
    evaluation.py      ← Model metrics
    export.py          ← YOLO / COCO export
  database.py          ← SQLite via SQLModel
```

The frontend is a React + TypeScript SPA. The backend is a FastAPI server that also serves the built frontend as static files — so there's only one process to run and one URL to visit.

---

## Tech Stack

**Backend**
- [FastAPI](https://fastapi.tiangolo.com/) — REST API + static file serving
- [SQLModel](https://sqlmodel.tiangolo.com/) + SQLite — data persistence
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — object detection training & inference
- [PyTorch](https://pytorch.org/) — classification training (ResNet18/50, MobileNetV3, EfficientNet-B0)
- [Uvicorn](https://www.uvicorn.org/) — ASGI server

**Frontend**
- [React](https://react.dev/) + [TypeScript](https://www.typescriptlang.org/)
- [Vite](https://vitejs.dev/) — build tool
- HTML5 Canvas — annotation drawing engine

---

## Project Structure

```
NoCode_CV/
├── Install NoCode CV.bat   ← Run once to install
├── NoCode CV.bat           ← Run to launch
├── installer.py            ← GUI installer (tkinter)
├── launcher.py             ← App launcher (opens browser)
├── backend/                ← FastAPI server + ML logic
├── frontend/               ← React source + pre-built dist
└── README.md
```

---

## License

MIT — free to use, modify, and distribute.
