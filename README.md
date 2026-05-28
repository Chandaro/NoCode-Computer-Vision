<div align="center">

<img src="frontend/public/favicon.svg" width="64" height="64" alt="NoCode CV Logo" />

# NoCode CV

**Train computer vision models without writing a single line of code.**  
Annotation · Detection · Classification · Segmentation · Pose · Webcam — all running 100% locally.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![YOLO](https://img.shields.io/badge/Ultralytics-YOLO11-FF6F00?style=flat-square)](https://ultralytics.com)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS-8b5cf6?style=flat-square)]()

<br />

> A self-hosted desktop web app. One `.bat` / `.sh` installer — no Docker, no cloud account, no data ever leaving your machine.

<br />

![Demo](docs/screenshots/demo.gif)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Screenshots](#screenshots)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Feature Reference](#feature-reference)
  - [Projects](#projects)
  - [Image Upload & Validation](#image-upload--validation)
  - [Annotation Studio](#annotation-studio)
  - [Dataset Analytics](#dataset-analytics)
  - [Object Detection Training](#object-detection-training)
  - [Image Classification](#image-classification)
  - [Custom CNN Builder](#custom-cnn-builder)
  - [Evaluation & Inference](#evaluation--inference)
  - [Video Inference + Tracking](#video-inference--tracking)
  - [Instance Segmentation](#instance-segmentation)
  - [Pose Estimation](#pose-estimation)
  - [Live Webcam](#live-webcam)
- [Dataset Import Formats](#dataset-import-formats)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

NoCode CV is a browser-based machine learning workstation that runs entirely on your own hardware. You open a URL in your browser — just like any web app — but everything: the model training, the inference, the database, the file storage, runs on a local Python server. No GPU cloud required. No API keys. No subscriptions.

It is designed for:
- **Researchers** who want to prototype CV pipelines quickly without boilerplate
- **Learners** who want to understand detection, segmentation, and classification hands-on
- **Small teams** who need a shared annotation + training tool without a SaaS budget

---

## Screenshots

| | |
|---|---|
| ![Projects](docs/screenshots/projects.png) | ![Annotate](docs/screenshots/annotate.png) |
| **Projects Dashboard** — manage all your CV projects | **Annotation Studio** — bbox, polygon, point tools |
| ![Train](docs/screenshots/train.png) | ![Analytics](docs/screenshots/analytics.png) |
| **Detection Training** — live log, augmentation preview | **Dataset Analytics** — class distribution, heatmaps |
| ![Segmentation](docs/screenshots/segmentation.png) | ![Pose](docs/screenshots/pose.png) |
| **Instance Segmentation** — pixel-level masks | **Pose Estimation** — 17-keypoint skeleton |
| ![Video](docs/screenshots/video_inference.png) | ![Webcam](docs/screenshots/webcam.png) |
| **Video Inference** — YouTube / file + tracking | **Live Webcam** — real-time detection overlay |

> 📁 Screenshots live in `docs/screenshots/`. Add your own after running the app.

---

## Features

| Feature | Description |
|---|---|
| 🖼 **Annotation Studio** | Bounding box, polygon & point tools. Auto-annotate with any trained `.pt` model |
| ⚡ **Object Detection** | YOLO11 / v10 / v9 / v8 · live training logs · augmentation preview · resume training |
| 🏷 **Classification** | ResNet / MobileNet / EfficientNet · top-5 predictions · per-class accuracy |
| 🧱 **Custom CNN Builder** | Design your own neural network architecture layer-by-layer and train it |
| 📊 **Dataset Analytics** | Class distribution, aspect ratio, RGB stats, annotation heatmaps |
| 📦 **Dataset Import / Export** | Upload images · import YOLO / COCO · export zip in one click |
| 🔬 **Evaluation** | mAP50, F1, PR curve, confusion matrix · image & URL batch testing |
| 📹 **Video Inference** | File upload or YouTube / TikTok URL · object tracking · download result |
| 🎯 **Pose Estimation** | 17-keypoint skeleton on image / video / webcam · YOLO11-pose |
| 🧩 **Instance Segmentation** | Pixel masks on image / video / webcam · pre-trained or custom `.pt` |
| 📷 **Live Webcam** | Real-time detection + segmentation overlay · FPS counter · per-class stats |

---

## Quick Start

```bash
# Windows
Install NoCode CV.bat

# macOS / Linux
chmod +x "Install NoCode CV.sh" && ./"Install NoCode CV.sh"
```

That's it. The installer:
1. Creates a Python virtual environment
2. Installs all backend dependencies (`ultralytics`, `fastapi`, `torch`, etc.)
3. Installs all frontend dependencies and builds the React app
4. Creates a launcher shortcut on your desktop

**After installation, launch the app:**

```bash
# Windows
NoCode CV.bat

# macOS / Linux
./NoCode\ CV.sh
```

Then open **http://localhost:8000** in your browser.

---

## Installation

### Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.9 | 3.11 |
| RAM | 8 GB | 16 GB |
| GPU | — (CPU works) | NVIDIA CUDA |
| Disk | 5 GB free | 20 GB free |
| OS | Windows 10 / macOS 12 | Windows 11 / macOS 14 |

### Manual Installation

```bash
# 1. Clone the repository
git clone https://github.com/Chandaro/NoCode-Computer-Vision.git
cd NoCode-Computer-Vision

# 2. Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install backend dependencies
pip install -r backend/requirements.txt

# 4. Install and build frontend
cd frontend
npm install
npm run build
cd ..

# 5. Start the server
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000**.

### GPU Support (NVIDIA)

The app auto-detects CUDA. If you have an NVIDIA GPU, install the CUDA-enabled torch build first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | React 18 · TypeScript · Vite · Tailwind CSS |
| **Backend** | Python · FastAPI · SQLModel · SQLite |
| **CV / ML** | Ultralytics YOLO · PyTorch · torchvision · OpenCV |
| **Video** | yt-dlp (1000+ sites) · OpenCV VideoWriter |
| **Icons** | Lucide React |
| **Charts** | Recharts |

---

## Project Structure

```
NoCode-Computer-Vision/
├── backend/
│   ├── main.py              # FastAPI app entry point
│   ├── models.py            # SQLModel database schemas
│   ├── database.py          # DB session setup
│   └── routers/
│       ├── projects.py      # Project & image CRUD
│       ├── annotate.py      # Annotation endpoints
│       ├── train.py         # Training orchestration
│       ├── infer.py         # Inference (image, video, URL)
│       ├── classify.py      # Classification training
│       ├── segment.py       # Segmentation inference
│       ├── pose.py          # Pose estimation
│       └── webcam.py        # Webcam streaming
├── frontend/
│   ├── src/
│   │   ├── pages/           # One file per page/feature
│   │   ├── components/
│   │   │   └── ui.tsx       # Shared design system components
│   │   ├── api.ts           # Axios API client
│   │   ├── App.tsx          # Router
│   │   └── index.css        # Design tokens + global styles
│   └── public/
│       └── favicon.svg
├── installer.py             # Cross-platform installer script
├── launcher.py              # App launcher
├── Install NoCode CV.bat    # Windows installer
├── Install NoCode CV.sh     # macOS/Linux installer
└── docs/
    └── screenshots/         # Add your screenshots here
```

---

## Feature Reference

### Projects

Every project is an isolated workspace containing images, annotations, class labels, training runs, and exported models. Create one project per detection task.

- Create, rename, and delete projects
- Define class labels at any time without losing existing annotations
- All data is stored locally in `nocode_cv.db` (SQLite)

---

### Image Upload & Validation

- Upload JPEG, PNG, BMP — single file or hundreds at once
- Auto-validates each image: dimensions, color space (RGB / grayscale), MD5 hash (deduplication), corruption check
- Batch progress bar for large uploads

---

### Annotation Studio

![Annotation tools](docs/screenshots/annotate_tools.png)

| Tool | Use case |
|---|---|
| **Bounding box** | Object detection — draw a rectangle around each object |
| **Polygon** | Precise outlines — click points around irregular shapes |
| **Point** | Keypoints, landmarks, or event locations |

- Pan and zoom canvas
- Per-class color coding
- Undo / redo
- Auto-annotate: run any trained `.pt` model on all unannotated images in one click
- Export annotations as YOLO `.txt` or COCO JSON

---

### Dataset Analytics

Visualize your dataset before training to catch imbalances early:

- Class distribution bar chart
- Annotations-per-image histogram
- Image size scatter plot (width × height)
- Aspect ratio buckets
- RGB channel mean / std
- Annotation heatmap overlay

---

### Object Detection Training

Trains YOLO models (YOLO11, v10, v9, v8) via fine-tuning on your annotated images.

**Available base models:**

| Family | Sizes |
|---|---|
| YOLO11 | n · s · m · l · x |
| YOLOv10 | n · s · m · l · x |
| YOLOv9 | t · s · m · c · e |
| YOLOv8 | n · s · m · l · x |

**Key training controls:**

- Epochs, image size, batch size, validation split
- Optimizer (auto / SGD / Adam / AdamW / RMSProp)
- Learning rate (lr0, lrf), momentum, weight decay
- 14 augmentation knobs (flip, rotation, mosaic, mixup, copy-paste, etc.)
- Early stopping patience
- Live streaming log with color-coded output
- Stop / resume any run

---

### Image Classification

Transfer learning on ResNet-18, ResNet-50, MobileNetV3-Small, or EfficientNet-B0.

- Top-1 / Top-5 accuracy
- Per-class accuracy breakdown
- Confusion matrix
- Freeze backbone toggle (faster training on small datasets)
- Inline inference — upload an image, get top-5 predictions with confidence scores

---

### Custom CNN Builder

Design a neural network architecture from scratch by stacking layers visually:

- Conv2D, MaxPool, BatchNorm, Dropout, Linear, ReLU, Flatten
- Set kernel size, stride, padding, channels per layer
- Live architecture preview
- Train on your project's classified images

---

### Evaluation & Inference

After any detection training run:

- **mAP50 / mAP50-95** — standard COCO detection metrics
- **Precision / Recall / F1** curves
- **Confusion matrix**
- **Test Set Evaluation** — upload a batch of images, get per-image detections + aggregate stats
- Single image inference via file upload or any image URL

---

### Video Inference + Tracking

- Upload a video file (MP4, AVI, MOV, MKV) or paste a URL
- YouTube, TikTok, Twitter, Vimeo and 1000+ sites supported via yt-dlp
- Optional object tracking — assigns persistent IDs across frames (ByteTrack)
- Download the annotated output video
- Confidence threshold slider

---

### Instance Segmentation

Run pixel-level segmentation masks on:
- Images (upload or URL)
- Video files / URLs
- Live webcam feed

Uses YOLO11-seg models. Works with custom-trained `.pt` files.

---

### Pose Estimation

Detects 17 human body keypoints and draws a skeleton overlay.

- Image, video, or webcam input
- YOLO11-pose models
- Adjustable confidence threshold

---

### Live Webcam

Real-time inference from your camera with:
- Detection or segmentation overlay
- FPS counter
- Per-class instance count
- Confidence threshold control

---

## Dataset Import Formats

### YOLO Folder

Select a folder that contains `images/` and `labels/` sub-directories with matching filenames:

```
my-dataset/
├── images/
│   ├── photo_001.jpg
│   └── photo_002.jpg
└── labels/
    ├── photo_001.txt   ← same filename, .txt extension
    └── photo_002.txt
```

### YOLO Files (flat)

Select image files and their matching `.txt` label files together in one file picker. Every `.jpg` must have a `.txt` with the same base name.

### YOLO Label Format

Each `.txt` file has one line per object:

```
class_id  cx  cy  width  height
```

- All values are **0–1 normalised** relative to image dimensions
- `class_id` is the zero-based index into your project's class list
- `cx cy` is the bounding box center
- `width height` is the bounding box size

```
0 0.512 0.480 0.300 0.420
1 0.210 0.340 0.150 0.200
```

Compatible with exports from **Roboflow**, **CVAT**, **LabelImg**, and **Label Studio**.

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Commit your changes: `git commit -m "feat: add your feature"`
4. Push: `git push origin feat/your-feature`
5. Open a pull request

Please follow the existing code style (TypeScript strict, Python type hints where practical).

---

## License

MIT © [Tekashimo](https://github.com/Chandaro)

---

<div align="center">
  <sub>Built with FastAPI · React · Ultralytics YOLO · PyTorch</sub>
</div>
