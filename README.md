# NoCode CV Trainer

A self-hosted desktop application for Windows and macOS that lets you annotate images, train computer vision models, and evaluate results without writing code. Everything runs locally — no data leaves the machine.

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

### YOLO Object Detection Training

Object detection is the task of finding objects in an image and drawing a box around each one. NoCode CV supports the full YOLO family — YOLO11, YOLOv10, YOLOv9, and YOLOv8 — all of which are fast, accurate single-stage detectors. Given an image, a YOLO model outputs a list of bounding boxes, each with a class label (what the object is) and a confidence score (how certain the model is).

Training starts from weights that were already trained on a large general dataset (COCO, with 80 common object categories), then continues on your specific images and classes. The model forgets nothing about what it already knows — it simply learns to also recognize your new categories. This process is called fine-tuning and requires far less data than training from scratch.

**Importing an existing YOLO dataset**

If you already have a labeled dataset in standard YOLO format (an `images/` folder and a `labels/` folder), you can import it directly:

- Click **Import YOLO Folder** and select the folder that contains `images/` and `labels/` sub-folders. The app matches each image to its label file by filename and imports all pairs automatically.
- Click **Import YOLO Files** to select individual image and label files from any location.

Large datasets (thousands of images) are uploaded in batches and a progress indicator shows how many pairs have been processed. Only images that have a matching label file are imported as annotated; unpaired images are uploaded without annotations.

**Choosing a model architecture**

NoCode CV ships with pretrained weights for four YOLO generations. Newer generations generally produce better accuracy at the same model size. All models are downloaded automatically on first use and cached locally.

| Family | Nano | Small | Medium | Large | Extra-large |
|---|---|---|---|---|---|
| YOLO11 | yolo11n.pt | yolo11s.pt | yolo11m.pt | yolo11l.pt | yolo11x.pt |
| YOLOv10 | yolov10n.pt | yolov10s.pt | yolov10m.pt | yolov10l.pt | yolov10x.pt |
| YOLOv9 | yolov9t.pt | yolov9s.pt | yolov9m.pt | yolov9c.pt | yolov9e.pt |
| YOLOv8 | yolov8n.pt | yolov8s.pt | yolov8m.pt | yolov8l.pt | yolov8x.pt |

Smaller models (nano, small) train faster and use less GPU memory but are less accurate. Larger models need more resources but produce better results on complex tasks. If you are just starting out or testing a new dataset, begin with **yolo11n.pt**. Upgrade to a larger model once you know your data is good.

**Training parameters**

- **Epochs** — one epoch means the model has seen every image in your training set once. More epochs give the model more time to learn, but too many cause overfitting — the model memorizes your training images rather than learning to generalize. Start with 50 and adjust based on the validation metrics.
- **Image size** — all images are resized to this square resolution (in pixels) before being passed to the model. 640 is the standard. Larger values like 1280 preserve more detail for small objects but require significantly more GPU memory and time.
- **Batch size** — how many images the model processes in one forward and backward pass before updating its weights. Larger batches produce more stable gradient updates but require more GPU memory. If training crashes with an out-of-memory error, halve the batch size.
- **Validation split** — the fraction of your images withheld from training and used purely for evaluation. At the end of each epoch, the model is tested on these images and its precision, recall, and mAP are recorded. This lets you monitor whether the model is improving or overfitting. Images are assigned to the validation set randomly so the split is representative of the full dataset.

**Optimizer & training hyperparameters**

These settings control how the model updates its weights during training. The defaults work well for most datasets; change them only if you have a reason to.

| Setting | Default | What it does |
|---|---|---|
| Optimizer | auto | Weight-update algorithm. `auto` lets YOLO choose the best optimizer for the selected architecture. Other options: `SGD`, `Adam`, `AdamW`, `NAdam`, `RAdam`, `RMSProp`. |
| Initial learning rate (lr0) | 0.01 | Step size at the start of training. Too large and training is unstable; too small and it converges slowly. |
| Final learning rate (lrf) | 0.01 | Fraction of lr0 to decay to by the last epoch. The scheduler interpolates between lr0 and lr0×lrf over training. |
| Momentum | 0.937 | Controls how much of the previous gradient is carried into the next step (SGD/SGD-based optimizers). Higher values smooth noisy gradients. |
| Weight decay | 0.0005 | L2 regularization coefficient. Penalizes large weights and reduces overfitting. |
| Warmup epochs | 3.0 | Number of epochs at the start of training where the learning rate ramps up gradually from near-zero to lr0. Prevents unstable updates in the first steps. |
| Early stop patience | 50 | If validation mAP does not improve for this many epochs, training stops automatically. Prevents wasting time on runs that have already converged. Set to 0 to disable. |

**Augmentation**

Augmentation means the training pipeline randomly transforms each image before feeding it to the model. The model never sees the exact same image twice, which forces it to become more robust. These are the available augmentation options:

| Augmentation | Default | Effect |
|---|---|---|
| Horizontal flip | 0.5 | Randomly mirrors images left-to-right. Useful when objects can face either direction. |
| Vertical flip | 0.0 | Randomly mirrors images top-to-bottom. Useful for aerial or overhead imagery. |
| Rotation | 0.0° | Randomly rotates images by up to ±N degrees. Helps the model handle tilted cameras. |
| Translation | 0.1 | Randomly shifts images horizontally and vertically by a fraction of the image size. |
| Scale | 0.5 | Randomly zooms images in or out. Makes the model robust to objects at different distances. |
| Shear | 0.0° | Randomly shears images along the horizontal axis by up to ±N degrees. Helps when objects appear at oblique angles. |
| Perspective | 0.0 | Applies a random perspective warp (0.0–0.001). Simulates the effect of a camera viewing the scene from a different angle. |
| HSV hue | 0.015 | Randomly shifts the hue of image colors. Helps with color variation in lighting conditions. |
| HSV saturation | 0.7 | Randomly changes color saturation. Models trained without this may fail on faded or oversaturated images. |
| HSV brightness | 0.4 | Randomly changes image brightness. Helps with dim or overexposed conditions. |
| Mosaic | 1.0 | Cuts four training images into quarters and assembles them into one composite image. Forces the model to detect objects at different scales and positions. Set to 0 to disable. |
| Mixup | 0.0 | Blends two images and their labels with a random opacity. Creates soft boundary examples that can improve generalization. |
| Copy-paste | 0.0 | Copies annotated objects from one image and pastes them into another. Effectively multiplies the number of object instances the model sees, which is especially useful for rare classes. |
| Random erasing | 0.4 | Randomly blacks out a rectangular region of each image. Forces the model to identify objects from partial views and reduces sensitivity to occlusion. |

**Live training log**

Once training starts, every line of output from the YOLO process is streamed to the browser in real time. You can watch the loss decrease and the mAP increase epoch by epoch without opening a terminal. A summary bar shows the current epoch, mAP50, precision, and recall at a glance.

**Stop and resume training**

- To cancel a run that is in progress, click the **Stop** button next to it in the run history. The run is marked as stopped and its last saved weights are preserved.
- To continue from a previous checkpoint and train for more epochs, select an existing completed or stopped run and click **Resume**. Training picks up from the last saved weights rather than starting over.
- Completed runs that are no longer needed can be removed with the **Delete** button.

---

### Image Classification Training

Image classification is a simpler task than object detection. Instead of finding and locating multiple objects, the model looks at the whole image and assigns it a single label. For example: "this image shows a cat", or "this plant leaf has early blight".

Classification is appropriate when:
- Every image in your dataset clearly belongs to exactly one category.
- You do not need to know where in the image the object is.
- Your categories are mutually exclusive (an image cannot belong to two classes at once).

NoCode CV trains classification models using transfer learning. A model that was already trained on 1.2 million images (ImageNet) is used as a starting point. Only the final decision-making layers are retrained on your data. This works well even with a few hundred images per class.

**How labels are assigned**

The class label for each image is taken from the **first annotation drawn on that image** in the Annotate page. Each class needs at least 2 annotated images, and the project must have at least 2 classes to start training. Unannotated images are silently skipped.

**Base models**

| Model | Notes |
|---|---|
| ResNet-18 | Small and fast. A reliable baseline for most classification tasks. |
| ResNet-50 | More capacity than ResNet-18. Better when you have more data and need higher accuracy. |
| MobileNetV3-Small | Designed for mobile and embedded devices. Very small model size, quick to train and deploy. |
| EfficientNet-B0 | Achieves good accuracy with fewer parameters. A good choice when memory is limited. |

**Training parameters**

- **Epochs** — classification models typically converge in 10–30 epochs. Start at 10 and increase if accuracy is still improving.
- **Image size** — all images are resized to this square resolution before training. Available sizes: 128, 224 (default), 256, 384. 224 matches the resolution the base models were originally trained on.
- **Batch size** — number of images per training step. 32 is a safe default.
- **Learning rate** — the step size used when updating model weights. Options: 0.01, 0.001 (default), 0.0001, 0.00001. Use 0.001 for fine-tuning. Drop to 0.0001 if the model diverges (loss goes up instead of down).
- **Freeze backbone** — when enabled, the convolutional layers that extract features from images are kept frozen at their pretrained values. Only the final classification layer is trained. This is faster and works better when your dataset is small. When disabled, all layers are updated, which can improve accuracy on larger datasets but risks overwriting the useful pretrained features (catastrophic forgetting).

**Built-in augmentation**

During training, images are automatically augmented with random horizontal flips and random color jitter (brightness, contrast, saturation) to help the model generalize. Validation images are not augmented.

**Results and metrics**

After training completes, the run history shows:
- **Top-1 accuracy** — the fraction of validation images where the model's single best guess was correct.
- **Top-5 accuracy** — the fraction where the correct class appeared anywhere in the model's top 5 predictions.
- **Per-class accuracy** — accuracy broken down by each individual class. Reveals if the model is failing on a specific category.
- **Confusion matrix** — a color-coded grid showing which classes the model confuses with each other. Click CM on any run to expand it.

**Built-in inference**

Once a run is completed, an inference panel appears directly on the Classification page. Upload any image, select a run, and click Classify. The result shows the top predicted class with its confidence score, followed by the next four most likely classes. No need to go to a separate inference page.

---

### Custom CNN Builder

The Custom CNN Builder is for users who want to design and experiment with their own neural network architecture instead of using a pretrained one. You add layers one by one using a visual interface, set each layer's parameters, and then train the resulting network on your project's classified images.

This feature is educational as well as practical — it lets you see how adding or removing layers changes the network, and how those choices affect training performance. Unlike the transfer-learning classification trainer, the custom CNN starts from randomly initialized weights (trained from scratch).

**What a convolutional neural network does**

A CNN reads an image as a grid of pixel values and passes it through a sequence of mathematical operations (layers). Early layers detect low-level patterns like edges. Middle layers combine those into shapes. Later layers combine shapes into objects. The final layer converts everything into a score per class.

**Available layers**

| Layer | What it does |
|---|---|
| Conv2D | The core building block. Applies a set of learned filters to the image to extract features. You choose the number of filters, kernel size, stride, and padding. More filters = more capacity to learn patterns. |
| BatchNorm2D | Normalizes each batch of activations to zero mean and unit variance. Prevents exploding/vanishing gradients and speeds up training. Usually placed after Conv2D. |
| ReLU | Activation function. Replaces every negative value with zero. Without activations, a stack of Conv2D layers collapses into one big matrix multiplication. |
| GELU | A smoother activation function than ReLU. Used in more modern architectures like Transformers. |
| Sigmoid | Squashes values to the range 0–1. Useful in binary classification output heads. |
| MaxPool2D | Reduces the spatial size of the feature map by keeping only the maximum value in each local window. Reduces memory and computation, and makes features robust to small shifts. |
| AvgPool2D | Same as MaxPool2D but uses the average instead of the maximum. Produces smoother downsampling. |
| Dropout | During training, randomly zeroes a fraction of activations. Prevents the network from relying too heavily on any single feature, reducing overfitting. Set the probability (p) from 0 to 1. |
| Flatten | Converts the 2D feature map into a 1D vector. Required before any Linear layer. Added automatically if you add a Linear without one. |
| Linear (FC) | A fully connected layer. Usually placed at the end to map learned features to class scores. |

**Architecture presets**

Five ready-made architectures are available as starting points:

| Preset | Description |
|---|---|
| Minimal | Single conv block → flatten → linear. Best for quick experiments and tiny datasets. |
| LeNet | Two conv blocks. A classic baseline, reliable for most tasks. |
| VGG-mini | Three deep conv blocks. Good accuracy on complex or large images. |
| BN Net | Two conv blocks with BatchNorm after each. Trains faster and more stably than the plain versions. |
| RegNet | Two conv blocks with BatchNorm and Dropout. Strong regularization — good when your dataset is small and overfitting is a concern. |

Select a preset, inspect the layers, then add, remove, or modify to suit your needs.

**Input dimensions**

Set the expected image width and height in pixels. All uploaded images are resized to this size before training. As you add layers, the builder computes the output shape at each step in real time and shows it next to each layer row. If a layer configuration would produce an invalid (zero or negative) spatial size, the builder flags it immediately so you can fix it before training.

A live **parameter count** is displayed as you build, giving you a rough sense of how large the model is before committing to training.

**Training parameters**

- **Epochs** — how many full passes through the training set. Custom CNNs often need more epochs than pretrained models (start with 20–50).
- **Batch size** — number of images per training step. 32 is a safe default; reduce if you run out of memory.
- **Learning rate** — step size for weight updates. 0.001 is a standard starting point for Adam (the optimizer used automatically).

**Results**

After training completes, the run history shows the best validation accuracy achieved. The model is saved as a `.pth` file and can be downloaded. Per-class breakdown and confusion matrix are not shown for custom runs (only top-1 accuracy).

---

### Evaluation

After a YOLO training run completes, the Evaluation page shows the full set of performance metrics and diagnostic charts generated during training. This page helps you understand how well the model works and where it struggles.

**Core metrics**

Understanding these four numbers tells you most of what you need to know about a detection model's performance:

- **Precision** — out of every detection the model made, what fraction was correct. If the model predicted 100 boxes and 80 of them actually contained the target object, precision is 0.80. High precision means the model rarely raises false alarms.
- **Recall** — out of every real object in the validation images, what fraction did the model find. If there were 100 real objects and the model detected 70 of them, recall is 0.70. High recall means the model rarely misses things.
- **mAP50** — mean Average Precision at an IoU (intersection over union) threshold of 0.50. IoU measures how much the predicted box overlaps the true box. At mAP50, a prediction counts as correct if the boxes overlap by at least 50%. This is the standard metric for comparing detection models. A perfect score is 1.0.
- **mAP50-95** — the same metric averaged across IoU thresholds from 0.50 to 0.95 in steps of 0.05. This is a stricter measure because it requires more precise box placement. It is harder to score well on and better reflects real-world localization quality.

**Per-class metrics**

The same precision, recall, and mAP numbers are shown for each class individually. This tells you whether the model is failing on a specific category — usually because that class has fewer training examples, noisier annotations, or visual ambiguity with another class.

**Diagnostic plots**

YOLO saves a set of charts during training. These are displayed directly in the browser:

| Plot | What to look for |
|---|---|
| results.png | Training loss should decrease over epochs. Validation mAP should increase. If validation mAP flattens while training loss keeps dropping, the model is overfitting. Early stopping will halt training automatically once improvement stalls. |
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

Upload any image and select a completed YOLO training run. The model runs detection on the image and returns all detected objects. The result is displayed as an overlay on the image, with bounding boxes, class labels, and confidence scores.

Two parameters control what gets shown:

- **Confidence threshold** — only detections above this confidence score are shown. Setting it to 0.25 means the model must be at least 25% confident before reporting a detection. Lower values catch more objects but include more false positives. Raise the threshold if too many incorrect boxes appear.
- **IoU threshold** — when the model predicts multiple overlapping boxes for the same object, non-maximum suppression removes the duplicates. The IoU threshold controls how much overlap is allowed before boxes are merged. Lower values keep more distinct boxes; higher values merge boxes more aggressively.

**Classification inference**

Upload any image and select a completed classification training run. The model returns:

- The top predicted class (the single most likely category).
- A ranked list of all classes with their probability scores. This tells you not just what the model thinks the image shows, but also how confident it is and what the second-most-likely option was.

**External model inference**

You can upload a pre-trained YOLO `.pt` model file from outside the app — for example, a model you downloaded, received from a colleague, or trained on another machine. Register it under External Models, then run detection inference on any image using that model. This lets you test any compatible model without tying it to a project training run.

---

### Model Export

Once a YOLO detection model is trained, you may want to deploy it somewhere other than this app — in a Python script, a phone app, a web API, or on specialized hardware. The Export page converts the trained model to different deployment formats.

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

You can register pre-trained YOLO `.pt` model files from outside the app. Once registered, these models appear in the inference page and can be used to run detections on any image. This is useful when you receive a trained model from someone else, download one from the internet, or want to compare multiple models on the same image without re-training.

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

## Requirements

| | Minimum | Recommended |
|---|---|---|
| Python | 3.9 | 3.11 |
| RAM | 8 GB | 16 GB |
| Disk | 4 GB free | — |
| GPU | — | NVIDIA with 6 GB VRAM |

AMD and Intel GPUs are not supported for CUDA acceleration and fall back to CPU training.

YOLO pretrained base weights (~6–130 MB depending on model size) are downloaded automatically on first use. All subsequent runs are fully offline.

---

## License

[MIT](LICENSE)
