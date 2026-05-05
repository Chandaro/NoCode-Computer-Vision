# NoCode CV Trainer

A self-hosted desktop application for Windows that lets you annotate images, train computer vision models, and evaluate results without writing code. Everything runs locally — no data leaves the machine.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)

---

## Features

---

### Projects

Every piece of work in NoCode CV lives inside a project. Think of a project as a folder that holds everything related to one specific task: your images, your class labels, your annotation data, your training runs, and your exported models. If you want to build two different detectors — one for smoke and one for product defects — you create two separate projects and work on them independently.

What you can do:

- Create a project with a name and an optional description.
- Add class labels to the project. A class label is the name of an object category you want the model to detect or classify, for example "car", "pedestrian", or "crack". You can add or change labels at any time without losing your existing work.
- Open any project from the projects page to access all of its images, annotations, training runs, and results.
- Delete a project in one click. This removes all images, annotations, and training history associated with it.

---

### Image Upload and Validation

Before you can train a model, you need to give it examples to learn from. The Images page lets you upload your image dataset and immediately tells you the quality of what you uploaded.

What you can do:

- Upload one image or hundreds at once using the file picker or drag-and-drop.
- Accept JPEG, PNG, and most common image file formats.
- View a list of all uploaded images with their filename, dimensions, file size, and color space.

What happens automatically on upload:

- Each image is inspected for its **width and height** in pixels.
- Its **color space** is detected (RGB for color images, Grayscale for black-and-white).
- An **MD5 hash** (a unique fingerprint) is calculated so the app can detect if you accidentally upload the same file twice.
- The image is checked for **corruption** — files that are broken or unreadable are flagged so you can replace them before wasting time annotating them.

---

### Annotation

Annotation is the process of labeling your images by drawing shapes around the objects you want the model to learn. Without annotations, the model has no way to know what counts as a "car" versus a "background" in your images. NoCode CV includes a full annotation canvas that runs directly in the browser — no external tool required.

**Shape tools**

Three drawing tools are available depending on what kind of label your task needs:

- **Bounding box** — draw a rectangle by clicking and dragging. This is the most common annotation type and is required for YOLOv8 object detection. Use it when the object has a roughly rectangular shape or when exact borders are not important.
- **Polygon** — click a series of points around the outline of an object. When you return close to the first point, the shape closes. Use polygons when the object has an irregular shape and you want to mark it precisely (for example, the exact border of a leaf, a crack in a surface, or an irregularly shaped part).
- **Point** — click once to drop a single dot. Use points to mark specific locations such as keypoints, landmarks, or events in an image.

**Canvas navigation**

- Scroll to zoom in and out on the image. Zoom in when you need to draw precise boxes on small objects.
- Hold and drag with no tool active to pan around a zoomed image.
- Navigate to the previous or next image using the arrow buttons or keyboard shortcuts without leaving the canvas.

**Working with annotations**

- Select an existing annotation to move it, resize it (bounding box), or drag individual vertices (polygon).
- Switch the active class label from the list on the right side — new shapes are tagged with the currently selected class and drawn in its assigned color.
- Each class is shown in a distinct color so you can immediately see which annotations belong to which category.
- Delete any selected annotation with the delete key or trash button.
- Undo or redo changes within the current session using the undo/redo buttons or keyboard shortcuts.

**Saving**

- Click Save to write all annotations for the current image to the database.
- When you navigate away, unsaved changes for the current image are held in memory until the page unloads.

---

### Dataset Analytics

Once your images are uploaded and annotated, the Analytics page gives you a statistical summary of your dataset. Reviewing this before training helps you catch problems that would otherwise result in a poor model.

What is reported:

- **Class distribution** — a bar chart showing how many annotation instances exist per class across the entire dataset. If one class has 800 annotations and another has 20, the model will strongly favor the larger class. You should aim for a reasonably balanced count across classes before training.
- **Shape type breakdown** — how many of your total annotations are bounding boxes, polygons, or points.
- **Annotations per image** — a histogram bucketing images by how many annotations they contain: 0, 1–5, 6–10, 11–20, and 21 or more. Images with zero annotations are unannotated and contribute nothing to training — the histogram helps you see how much of your dataset is actually labeled.
- **Image size scatter plot** — a chart of width versus height for every image. If your images have very different resolutions, the model will see them all at the same size after resizing, which may distort some of them.
- **Aspect ratio distribution** — groups images into buckets (tall portrait, square, wide landscape, ultra-wide) so you can see whether your dataset has a consistent shape.
- **Color space breakdown** — counts how many images are RGB versus grayscale. A mix of both may affect training if the model expects a fixed number of input channels.
- **Channel mean and standard deviation** — computes the average pixel value and spread for each of the red, green, and blue channels across the dataset. These values are used for normalization in the custom CNN builder.

---

### YOLOv8 Object Detection Training

Object detection is the task of finding objects in an image and drawing a box around each one. YOLOv8 (You Only Look Once, version 8) is one of the fastest and most accurate object detection architectures available. Given an image, it outputs a list of bounding boxes, each with a class label (what the object is) and a confidence score (how certain the model is).

Training a YOLOv8 model means starting from weights that were already trained on a large general dataset (COCO, with 80 common object categories), then continuing to train on your specific images and classes. The model forgets nothing about what it already knows — it simply learns to also recognize your new categories. This process is called fine-tuning and requires far less data than training from scratch.

**Choosing a model size**

Five pretrained model sizes are available. Smaller models train faster and use less GPU memory but are less accurate. Larger models need more resources but produce better results on complex tasks.

| Model | Parameters | Best use |
|---|---|---|
| yolov8n.pt (nano) | ~3M | Quick experiments, edge devices with limited compute |
| yolov8s.pt (small) | ~11M | Devices with moderate compute, mobile deployment |
| yolov8m.pt (medium) | ~26M | General-purpose tasks with a decent GPU |
| yolov8l.pt (large) | ~44M | High-accuracy tasks where speed is less critical |
| yolov8x.pt (extra-large) | ~68M | Maximum accuracy, requires a powerful GPU |

If you are just starting out or testing a new dataset, begin with yolov8n.pt. Upgrade to a larger model once you know your data is good.

**Training parameters**

- **Epochs** — one epoch means the model has seen every image in your training set once. More epochs give the model more time to learn, but too many cause overfitting — the model memorizes your training images rather than learning to generalize. Start with 50 and adjust based on the validation metrics.
- **Image size** — all images are resized to this square resolution (in pixels) before being passed to the model. 640 is the standard for YOLOv8. Larger values like 1280 preserve more detail for small objects but require significantly more GPU memory and time.
- **Batch size** — how many images the model processes in one forward and backward pass before updating its weights. Larger batches produce more stable gradient updates but require more GPU memory. If training crashes with an out-of-memory error, halve the batch size.
- **Validation split** — the fraction of your images withheld from training and used purely for evaluation. At the end of each epoch, the model is tested on these images and its precision, recall, and mAP are recorded. This lets you monitor whether the model is improving or overfitting.

**Augmentation**

Augmentation means the training pipeline randomly transforms each image before feeding it to the model. The model never sees the exact same image twice, which forces it to become more robust. These are the available augmentation options:

| Augmentation | Effect |
|---|---|
| Horizontal flip | Randomly mirrors images left-to-right. Useful when objects can face either direction. |
| Vertical flip | Randomly mirrors images top-to-bottom. Useful for aerial or overhead imagery. |
| Rotation | Randomly rotates images by up to ±N degrees. Helps the model handle tilted cameras. |
| Translation | Randomly shifts images horizontally and vertically by a fraction of the image size. |
| Scale | Randomly zooms images in or out. Makes the model robust to objects at different distances. |
| HSV hue | Randomly shifts the hue of image colors. Helps with color variation in lighting conditions. |
| HSV saturation | Randomly changes color saturation. Models trained without this may fail on faded or oversaturated images. |
| HSV brightness | Randomly changes image brightness. Helps with dim or overexposed conditions. |
| Mosaic | Cuts four training images into quarters and assembles them into one composite image. Forces the model to detect objects at different scales and positions. Enabled by default. |
| Mixup | Blends two images and their labels with a random opacity. Creates soft boundary examples that can improve generalization. |

**Live training log**

Once training starts, every line of output from the YOLO process is streamed to the browser in real time. You can watch the loss decrease and the mAP increase epoch by epoch without opening a terminal.

**Resume training**

If a training run was interrupted, or you want to continue from a previous checkpoint to train for more epochs, you can select an existing run on the training page and resume from its last saved weights rather than starting over.

---

### Image Classification Training

Image classification is a simpler task than object detection. Instead of finding and locating multiple objects, the model looks at the whole image and assigns it a single label. For example: "this image shows a cat", or "this plant leaf has early blight".

Classification is appropriate when:
- Every image in your dataset clearly belongs to exactly one category.
- You do not need to know where in the image the object is.
- Your categories are mutually exclusive (an image cannot belong to two classes at once).

NoCode CV trains classification models using transfer learning. A model that was already trained on 1.2 million images (ImageNet) is used as a starting point. Only the final decision-making layers are retrained on your data. This works well even with a few hundred images per class.

**Base models**

| Model | Notes |
|---|---|
| ResNet-18 | Small and fast. A reliable baseline for most classification tasks. |
| ResNet-50 | More capacity than ResNet-18. Better when you have more data and need higher accuracy. |
| MobileNetV3-Small | Designed for mobile and embedded devices. Very small model size, quick to train and deploy. |
| EfficientNet-B0 | Achieves good accuracy with fewer parameters. A good choice when memory is limited. |

**Training parameters**

- **Epochs** — classification models typically converge in 10–30 epochs. Start at 10 and increase if accuracy is still improving.
- **Image size** — all images are resized to this square resolution. 224×224 is the standard because it matches the resolution the base models were originally trained on.
- **Batch size** — number of images per training step. 32 is a safe default.
- **Learning rate** — the step size used when updating model weights. 0.001 is a standard starting value for fine-tuning. If the model diverges (loss goes up instead of down), try a smaller value like 0.0001.
- **Freeze backbone** — when enabled, the convolutional layers that extract features from images are kept frozen at their pretrained values. Only the final classification layer is trained. This is faster and works better when your dataset is small. When disabled, all layers are updated, which can improve accuracy on larger datasets but risks overwriting the useful pretrained features (catastrophic forgetting).

---

### Custom CNN Builder

The Custom CNN Builder is for users who want to design and experiment with their own neural network architecture instead of using a pretrained one. You add layers one by one using a visual interface, set each layer's parameters, and then train the resulting network on your project's classified images.

This feature is educational as well as practical — it lets you see how adding or removing layers changes the network, and how those choices affect training performance.

**What a convolutional neural network does**

A CNN reads an image as a grid of pixel values and passes it through a sequence of mathematical operations (layers). Early layers detect low-level patterns like edges. Middle layers combine those into shapes. Later layers combine shapes into objects. The final layer converts everything into a score per class.

**Available layers**

| Layer | What it does |
|---|---|
| Conv2D | The core building block. Applies a set of learned filters to the image to extract features. You choose the number of filters and their size. More filters = more capacity to learn patterns. |
| BatchNorm2D | Normalizes the output of the previous layer so values stay in a stable range. Makes training faster and more reliable. Usually placed after Conv2D. |
| ReLU | Activation function. Replaces every negative value with zero. Without activations, a stack of Conv2D layers is just one big matrix multiplication. |
| GELU | A smoother activation function than ReLU. Used in more modern architectures. |
| Sigmoid | Squashes values to the range 0–1. Used in the final layer for binary classification (two classes). |
| MaxPool2D | Reduces the spatial size of the feature map by keeping only the maximum value in each local window. Reduces memory and computation, and makes the network less sensitive to exact object position. |
| AvgPool2D | Same as MaxPool2D but uses the average instead of the maximum. |
| Dropout | During training, randomly sets a fraction of activations to zero. This prevents the network from relying too heavily on any single feature, which reduces overfitting. |
| Flatten | Converts the 2D feature map into a 1D vector. Required before any Linear layer. |
| Linear | A fully connected layer that multiplies the 1D vector by a weight matrix. Usually placed at the end to map features to class scores. |

**3D visualization**

Each layer in your architecture is rendered as a 3D block in an interactive scene. The block dimensions scale with the layer's parameters (number of filters, kernel size), giving you a visual sense of how the network's internal representation grows and shrinks as data flows through it.

**Presets**

Pre-built architectures are available as starting points: a minimal classifier, a deeper classifier, and others. Select one, inspect it, then modify it to suit your needs.

**Input dimensions**

Set the expected image size (width and height in pixels). As you add layers, the builder computes the output dimensions at each step and warns you if a layer configuration would produce an invalid (zero or negative) spatial size.

---

### Evaluation

After a YOLOv8 training run completes, the Evaluation page shows the full set of performance metrics and diagnostic charts generated during training. This page helps you understand how well the model works and where it struggles.

**Core metrics**

Understanding these four numbers tells you most of what you need to know about a detection model's performance:

- **Precision** — out of every detection the model made, what fraction was correct. If the model predicted 100 boxes and 80 of them actually contained the target object, precision is 0.80. High precision means the model rarely raises false alarms.
- **Recall** — out of every real object in the validation images, what fraction did the model find. If there were 100 real objects and the model detected 70 of them, recall is 0.70. High recall means the model rarely misses things.
- **mAP50** — mean Average Precision at an IoU (intersection over union) threshold of 0.50. IoU measures how much the predicted box overlaps the true box. At mAP50, a prediction counts as correct if the boxes overlap by at least 50%. This is the standard metric for comparing detection models. A perfect score is 1.0.
- **mAP50-95** — the same metric averaged across IoU thresholds from 0.50 to 0.95 in steps of 0.05. This is a stricter measure because it requires more precise box placement. It is harder to score well on and better reflects real-world localization quality.

**Per-class metrics**

The same precision, recall, and mAP numbers are shown for each class individually. This tells you whether the model is failing on a specific category — usually because that class has fewer training examples, noisier annotations, or visual ambiguity with another class.

**Diagnostic plots**

YOLOv8 saves a set of charts during training. These are displayed directly in the browser:

| Plot | What to look for |
|---|---|
| results.png | Training loss should decrease over epochs. Validation mAP should increase. If validation mAP flattens while training loss keeps dropping, the model is overfitting. |
| Confusion matrix | Each row is a true class; each column is a predicted class. The diagonal shows correct predictions. Off-diagonal entries show which classes the model confuses with each other. |
| Confusion matrix (normalized) | Same as above, but expressed as percentages of each true class. Easier to read when class sizes differ. |
| PR curve | Plots precision against recall at different confidence thresholds. A curve that stays near the top-right corner indicates a good model. |
| F1 curve | Plots F1 score (harmonic mean of precision and recall) against confidence threshold. The peak of this curve is the best operating point. |
| BoxP curve / BoxR curve | Precision and recall separately, plotted against confidence threshold. Useful when you want to understand the precision-recall tradeoff at a specific confidence setting. |
| Label distribution | Shows the spatial distribution of annotation centers across all training images. A uniform spread is healthy. Concentrated blobs mean the model may only learn to detect objects in certain positions. |
| Validation batch previews | Side-by-side comparison of the ground-truth boxes (what you labeled) and the model's predicted boxes on validation images. This is the quickest way to visually judge whether the model is making sensible predictions. |

---

### Inference (Running Predictions)

Inference means giving a trained model a brand-new image — one it has never seen — and letting it make a prediction. This is how you verify the model works in practice, and how you would use it in a real application.

**Object detection inference**

Upload any image and select a completed YOLOv8 training run. The model runs detection on the image and returns all detected objects. The result is displayed as an overlay on the image, with bounding boxes, class labels, and confidence scores.

Two parameters control what gets shown:

- **Confidence threshold** — only detections above this confidence score are shown. Setting it to 0.25 means the model must be at least 25% confident before reporting a detection. Lower values catch more objects but include more false positives. Raise the threshold if too many incorrect boxes appear.
- **IoU threshold** — when the model predicts multiple overlapping boxes for the same object, non-maximum suppression removes the duplicates. The IoU threshold controls how much overlap is allowed before boxes are merged. Lower values keep more distinct boxes; higher values merge boxes more aggressively.

**Classification inference**

Upload any image and select a completed classification training run. The model returns:

- The top predicted class (the single most likely category).
- A ranked list of all classes with their probability scores. This tells you not just what the model thinks the image shows, but also how confident it is and what the second-most-likely option was.

**External model inference**

You can upload a pre-trained YOLOv8 `.pt` model file from outside the app — for example, a model you downloaded, received from a colleague, or trained on another machine. Register it under External Models, then run detection inference on any image using that model. This lets you test any compatible model without tying it to a project training run.

---

### Model Export

Once a YOLOv8 detection model is trained, you may want to deploy it somewhere other than this app — in a Python script, a phone app, a web API, or on specialized hardware. The Export page converts the trained model to different deployment formats.

- **ONNX** — Open Neural Network Exchange. An open, hardware-agnostic format supported by almost every inference runtime: ONNX Runtime, OpenCV DNN, TensorRT, CoreML, and more. This is the best choice if you want maximum compatibility and plan to run the model in a Python script or a cross-platform application.
- **TFLite** — TensorFlow Lite. A compact format designed for mobile and embedded devices: Android apps, Raspberry Pi, Edge TPU (Google Coral), and microcontrollers. Produces smaller files with lower memory requirements than full TensorFlow.
- **TensorRT** — NVIDIA's optimized inference runtime. Compiles the model into a `.engine` file that is tuned specifically for the GPU it was exported on. Runs significantly faster than ONNX on NVIDIA hardware. Requires a machine with a compatible NVIDIA GPU and TensorRT installed.

Export runs in the background after you click the button. When it finishes, a download link appears for the converted file.

---

### Dataset Export

If you want to use your annotated data outside of NoCode CV — to train a model in a different framework, share it with a team, or contribute to a public benchmark — you can export the full dataset as a ZIP archive.

- **YOLO format** — exports images and label files in the standard folder structure expected by Ultralytics and other YOLO training pipelines: `images/` and `labels/` directories, with each label file containing one line per annotation in normalized `[class_id x_center y_center width height]` format. A `data.yaml` file describes the class names and counts. Bounding boxes, polygons, and points are all exported in the appropriate YOLO text format.
- **COCO format** — exports a single `annotations.json` file in the COCO JSON schema alongside the image files. This format is compatible with Detectron2, MMDetection, YOLOX, and most academic benchmarks.

---

### External Model Management

You can register pre-trained YOLOv8 `.pt` model files from outside the app. Once registered, these models appear in the inference page and can be used to run detections on any image. This is useful when you receive a trained model from someone else, download one from the internet, or want to compare multiple models on the same image without re-training.

Registered external models are stored on disk inside the app's `external_models/` folder and tracked in the database. They can be deleted from the interface when no longer needed.

---

## Installation

**Windows**

Run `Install NoCode CV.bat` to open the setup wizard.

**macOS / Linux**

```bash
bash "Install NoCode CV.sh"
```

Both launchers open the same Tkinter setup wizard and handle the following automatically:

- Detects Python 3.9+ on your system
- Creates an isolated virtual environment (or reuses your existing Python if you choose)
- Detects your GPU and installs the correct PyTorch build (CUDA 11.8, 12.x, or CPU fallback)
- Installs all backend dependencies
- Writes a platform-appropriate launcher script (`NoCode CV.bat` on Windows, `NoCode CV.sh` on macOS/Linux)

After installation:

- **Windows** — double-click `NoCode CV.bat` or the desktop shortcut
- **macOS / Linux** — run `bash "NoCode CV.sh"` in a terminal

Then open `http://localhost:8000` in a browser. The first install takes roughly five minutes due to the PyTorch download. Subsequent launches are immediate.

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
