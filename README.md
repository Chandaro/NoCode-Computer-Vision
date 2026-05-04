# NoCode CV Trainer

A self-hosted desktop application for Windows that lets you annotate images, train computer vision models, and evaluate results without writing code. Everything runs locally — no data leaves the machine.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)

---

## Features

**Annotation**  
Draw bounding boxes, polygons, and point markers on images. Supports undo/redo, zoom, and label management.

**YOLOv8 Object Detection**  
Train YOLOv8 models with configurable hyperparameters and augmentation settings. Training output streams to the UI in real time.

**Image Classification**  
Fine-tune ResNet, MobileNetV3, or EfficientNet-B0 on custom image classes using PyTorch transfer learning.

**Custom CNN Builder**  
Define a neural network architecture layer by layer in an interactive canvas. Presets are included for common configurations.

**Evaluation**  
Review precision, recall, and mAP metrics with per-class breakdowns. Run inference on new images immediately after training.

**Dataset Export**  
Export labelled datasets in YOLO or COCO format as a zip archive, compatible with any downstream pipeline.

---

## Installation

Run `Install NoCode CV.bat` to open the setup wizard. It handles the following automatically:

- Detects Python on your system (supports `python`, `python3`, and the `py` launcher)
- Creates an isolated virtual environment
- Detects your GPU and installs the correct PyTorch build (CUDA 11.8, 12.x, or CPU fallback)
- Installs all backend dependencies

After installation, launch the app with `NoCode CV.bat` and open `http://localhost:8000` in a browser. The first install takes roughly five minutes. Subsequent launches are immediate.

---

## Manual Setup

**Clone the repository**

```bash
git clone https://github.com/Chandaro/NoCode-Computer-Vision.git
cd NoCode-Computer-Vision
```

**Create a virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**Install PyTorch**

Choose the build that matches your hardware:

```bash
# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8  (GTX 10xx / RTX 20xx)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.x  (RTX 30/40 series)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.5+ (RTX 40/50 series)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**Install backend dependencies**

```bash
pip install -r backend/requirements.txt \
    --prefer-binary \
    --extra-index-url https://download.pytorch.org/whl/cpu
```

**Rebuild the frontend** (optional — a pre-built bundle is already included in `frontend/dist/`)

```bash
cd frontend
npm install && npm run build
cd ..
```

**Start the server**

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`.

---

## Requirements

| | Minimum | Recommended |
|---|---|---|
| Python | 3.9 | 3.11 |
| RAM | 8 GB | 16 GB |
| Disk | 4 GB free | — |
| GPU | — | NVIDIA with 6 GB VRAM |

AMD and Intel GPUs are not supported for CUDA acceleration and fall back to CPU training.

YOLOv8 downloads pretrained base weights (~130 MB) on first use. All subsequent runs are fully offline.

---

## Project Structure

```
NoCode-Computer-Vision/
├── Install NoCode CV.bat     setup wizard, run once
├── NoCode CV.bat             application launcher
├── installer.py              Tkinter GUI installer
├── launcher.py               starts the backend and opens the browser
│
├── backend/
│   ├── main.py               FastAPI entry point
│   ├── database.py           SQLite schema via SQLModel
│   ├── requirements.txt      Python dependencies
│   └── routers/
│       ├── images.py         image upload and storage
│       ├── training.py       YOLOv8 training, SSE log streaming
│       ├── classification.py PyTorch classification training
│       ├── custom.py         custom CNN builder and training
│       ├── infer.py          inference on trained models
│       ├── analytics.py      dataset statistics
│       ├── evaluation.py     metrics and evaluation
│       └── export.py         YOLO and COCO export
│
└── frontend/
    ├── dist/                 pre-built bundle, served by FastAPI
    └── src/
        ├── pages/            React page components
        └── components/       shared UI components
```

The frontend is a React + TypeScript SPA served as static files directly by FastAPI — one process, one port.

---

## Dependencies

**Backend**

| Package | Version | Purpose |
|---|---|---|
| FastAPI | 0.115 | HTTP API and static file serving |
| Uvicorn | 0.32 | ASGI server |
| SQLModel | 0.0.22 | ORM and SQLite persistence |
| Ultralytics | 8.4 | YOLOv8 training and inference |
| PyTorch | 2.x | Classification model training |
| Pillow | 11.x | Image I/O and preprocessing |

**Frontend**

| Package | Version | Purpose |
|---|---|---|
| React | 18 | UI framework |
| TypeScript | 5.x | Type checking |
| Vite | 6.x | Build tooling |
| Three.js | 0.x | 3D CNN architecture visualisation |
| HTML5 Canvas | — | Annotation drawing engine |

---

## Contributing

Fork the repository, create a branch off `main`, and open a pull request. Keep each PR to a single change or fix.

```bash
git checkout -b your-branch-name
git push origin your-branch-name
```

---

## License

[MIT](LICENSE)
