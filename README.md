<div align="center">

<br/>

<img src="https://img.shields.io/badge/NoCode%20CV-Trainer-5865f2?style=for-the-badge&labelColor=0d0d0f" alt="NoCode CV Trainer" />

<br/><br/>

**Train production-grade computer vision models — no code required.**

Annotate images, design custom CNN architectures, train YOLOv8 & classification models,<br/>
evaluate performance, and export datasets — all from a single desktop application.

<br/>

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00b4d8?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![React](https://img.shields.io/badge/React-18-61dafb?style=flat-square&logo=react&logoColor=black)](https://react.dev/)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

<br/>

</div>

---

## Overview

NoCode CV Trainer is a **self-hosted, offline-first** desktop application for Windows that gives non-technical teams the same computer vision capabilities as professional ML engineers — without writing a single line of code.

Everything runs **locally on your machine**. No subscriptions. No data leaves your network. No cloud dependencies.

---

## Features

<table>
<tr>
<td width="50%">

### 🖼 Annotation Studio
Draw bounding boxes, polygons, and point markers on your images with a full-featured annotation canvas — complete with undo/redo, zoom, and label management.

</td>
<td width="50%">

### 🧠 Custom CNN Builder
Design and visualise your own neural network architecture layer by layer in an interactive 3D/2D canvas. Load architecture presets or build from scratch.

</td>
</tr>
<tr>
<td width="50%">

### ⚡ YOLOv8 Object Detection
Train state-of-the-art YOLOv8 models with real-time log streaming, live loss curves, and full augmentation control — directly on your GPU.

</td>
<td width="50%">

### 🏷 Image Classification
Fine-tune ResNet, MobileNetV3, and EfficientNet-B0 on your own image classes using PyTorch transfer learning. Results in minutes, not hours.

</td>
</tr>
<tr>
<td width="50%">

### 🔬 Evaluation & Inference
Review model performance with precision, recall, mAP metrics, and per-class breakdowns. Run live inference on new images immediately after training.

</td>
<td width="50%">

### 📦 Dataset Export
Export your labelled dataset in **YOLO** or **COCO** format as a ready-to-use zip archive — compatible with any downstream ML pipeline.

</td>
</tr>
</table>

---

## Getting Started

### Option A — One-Click Install *(Recommended)*

> **No terminal required.** Everything is handled automatically.

```
1.  Double-click   →   Install NoCode CV.bat
2.  Double-click   →   NoCode CV.bat
3.  Open browser   →   http://localhost:8000
```

The installer will:
- Detect your Python version automatically (`python`, `python3`, `py` launcher — all supported)
- Create an isolated virtual environment
- Install the correct PyTorch build for your GPU (CUDA 11.8 / 12.x / CPU — auto-detected)
- Install all backend libraries
- Launch a guided setup wizard

**First install takes ~5 minutes.** Subsequent launches are instant.

---

### Option B — Manual / Developer Setup

**1. Clone**
```bash
git clone https://github.com/Chandaro/NoCode-Computer-Vision.git
cd NoCode-Computer-Vision
```

**2. Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install PyTorch** *(choose one)*
```bash
# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.x (RTX 30/40 series)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.5+ (RTX 40/50 series — Blackwell / Ada)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**4. Install backend dependencies**
```bash
pip install -r backend/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

**5. *(Optional)* Rebuild the frontend**

The pre-built frontend is already included in `frontend/dist/`. Only needed if you modify the UI:
```bash
cd frontend
npm install && npm run build
cd ..
```

**6. Start the server**
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open **[http://localhost:8000](http://localhost:8000)** in your browser.

---

## Hardware Requirements

| Hardware | Training Speed | Notes |
|---|---|---|
| **NVIDIA GPU + CUDA 12.x** | ⚡ Fast — minutes per epoch | Recommended for serious training |
| **NVIDIA GPU + CUDA 11.8** | ⚡ Fast — minutes per epoch | Older GPUs (GTX 10xx / RTX 20xx) |
| **CPU only** | 🐢 Slow — hours per epoch | Fine for small datasets & testing |
| **AMD / Intel GPU** | 🐢 Falls back to CPU | No CUDA acceleration |

> **Minimum:** Python 3.9+, 8 GB RAM, 4 GB free disk space  
> **Recommended:** Python 3.11+, 16 GB RAM, NVIDIA GPU with 6 GB VRAM

On first use, YOLOv8 downloads its base weights (~130 MB) once. All subsequent training runs offline.

---

## Architecture

```
NoCode-Computer-Vision/
│
├── Install NoCode CV.bat      ← Guided GUI installer (run once)
├── NoCode CV.bat              ← App launcher (run to start)
├── installer.py               ← Tkinter setup wizard
├── launcher.py                ← Starts backend + opens browser
│
├── backend/
│   ├── main.py                ← FastAPI app entry point
│   ├── database.py            ← SQLite via SQLModel
│   ├── requirements.txt       ← Python dependencies
│   └── routers/
│       ├── images.py          ← Image upload & management
│       ├── training.py        ← YOLOv8 training + SSE log streaming
│       ├── classification.py  ← PyTorch classification training
│       ├── custom.py          ← Custom CNN builder & training
│       ├── infer.py           ← Inference on trained models
│       ├── analytics.py       ← Dataset statistics
│       ├── evaluation.py      ← Model metrics & evaluation
│       └── export.py          ← YOLO / COCO dataset export
│
└── frontend/
    ├── dist/                  ← Pre-built production bundle (served by FastAPI)
    └── src/
        ├── pages/             ← React page components
        │   ├── CustomModel.tsx  ← CNN architecture builder
        │   └── ...
        └── components/        ← Shared UI components
```

> The frontend is a **React + TypeScript SPA** served directly by FastAPI as static files — one process, one URL, zero configuration.

---

## Tech Stack

### Backend
| Library | Version | Role |
|---|---|---|
| [FastAPI](https://fastapi.tiangolo.com/) | 0.115 | REST API + static file serving |
| [Uvicorn](https://www.uvicorn.org/) | 0.32 | ASGI server |
| [SQLModel](https://sqlmodel.tiangolo.com/) | 0.0.22 | ORM + SQLite persistence |
| [Ultralytics](https://github.com/ultralytics/ultralytics) | 8.4 | YOLOv8 detection training & inference |
| [PyTorch](https://pytorch.org/) | 2.x | Classification training backbone |
| [Pillow](https://python-pillow.org/) | 11.x | Image processing |

### Frontend
| Library | Version | Role |
|---|---|---|
| [React](https://react.dev/) | 18 | UI framework |
| [TypeScript](https://www.typescriptlang.org/) | 5.x | Type safety |
| [Vite](https://vitejs.dev/) | 6.x | Build tool |
| [Three.js](https://threejs.org/) | 0.x | 3D CNN architecture visualisation |
| HTML5 Canvas | — | Annotation drawing engine |

---

## Workflow

```
Upload Images  →  Annotate  →  Train  →  Evaluate  →  Export
     │               │            │           │            │
  Drag & drop    BBox / Poly   YOLOv8 or   mAP / P/R   YOLO zip
  or bulk add    Point tools   ResNet etc  per class    COCO zip
```

---

## Contributing

Contributions are welcome. To get started:

```bash
# Fork the repo, then clone your fork
git clone https://github.com/YOUR_USERNAME/NoCode-Computer-Vision.git

# Create a feature branch
git checkout -b feature/your-feature-name

# Make changes, then push and open a PR
git push origin feature/your-feature-name
```

Please keep PRs focused — one feature or fix per PR.

---

## License

[MIT](LICENSE) — free to use, modify, and distribute.

---

<div align="center">

Built with FastAPI · PyTorch · React · Three.js

</div>
