# Adversarial Perception Testing Framework

A ROS2-based framework for systematically testing how object detection and tracking systems fail under adversarial conditions — and building defenses against those failures.

```
                     clean frame                    foggy frame
                    ┌───────────┐                  ┌───────────┐
                    │  ┌──┐     │                  │           │
                    │  │##│ car │      fog          │  ?    ?   │
                    │  └──┘     │  ──────────►     │           │
                    │     ┌──┐  │   attack         │     ?     │
                    │     │##│  │                   │           │
                    └───────────┘                  └───────────┘
                    MOTA: 0.72                      MOTA: 0.31
```

**The question this answers:** *How fragile is a modern perception pipeline, and what does it take to harden it?*

## What It Does

Takes video (or drone footage from VisDrone), runs YOLOv8 object detection + ByteTrack multi-object tracking, then systematically degrades the input with simulated attacks and measures the impact on tracking accuracy using standard MOT metrics (MOTA, MOTP, IDF1).

```
Video ──► [Attack Node] ──► Detection Node ──► Tracking Node ──► [Defense Node] ──► Evaluation
           fog/rain/         YOLOv8              ByteTrack        temporal          MOTA/MOTP
           patches                                                consistency       IDF1
```

## Current Status

| Component | Status | Description |
|-----------|--------|-------------|
| Detection (YOLOv8) | Done | Supports video files + image sequences (VisDrone) |
| Tracking (ByteTrack) | Done | Custom implementation with Kalman filtering |
| Environmental attacks | Done | Fog, rain, blur, low light, lens dirt, contrast |
| Visual adversarial attacks | Done | Patches, stripes, checkerboard, occlusion |
| Defense mechanisms | Done | Temporal consistency, anomaly detection |
| Web dashboard | Done | Live MJPEG streaming at localhost:8080 |
| Benchmark script | Done | Standalone comparison across all attack configs |
| Evaluation node | Done | MOTA/MOTP/IDF1 against ground truth |
| Fine-tuning pipeline | Done | VisDrone → YOLO format conversion + training script |
| Gradient-based attacks | Done | FGSM & PGD (custom implementation, no torchattacks) |
| Adversarial training | Planned | Fine-tune on adversarial examples for robustness |

## Quick Start (Docker)

### Prerequisites
- Docker Desktop

### Setup

```bash
git clone <this-repo>
cd adversarial-tracking

# Build the container (ROS2 Humble + YOLOv8 + all dependencies)
docker compose build

# Start the container
docker compose up -d
docker exec -it tracking_adversarial bash
```

### Download Test Data

Download a sequence from [VisDrone](https://github.com/VisDrone/VisDrone-Dataset) and place it in `tracking_ws/data/`:
```
tracking_ws/data/
├── videos/
│   └── uav0000086_00000_v/    # folder of numbered JPGs
│       ├── 0000001.jpg
│       ├── 0000002.jpg
│       └── ...
└── annotations/
    └── uav0000086_00000_v.txt  # VisDrone MOT ground truth
```

### Run

```bash
# Inside the Docker container:
source /tracking_ws/install/setup.bash

# 1. Baseline — clean detection + tracking
ros2 launch tracking_adversarial baseline.launch.py \
    video_source:=/tracking_ws/data/videos/uav0000086_00000_v

# 2. Under attack — add fog
ros2 launch tracking_adversarial adversarial.launch.py \
    video_source:=/tracking_ws/data/videos/uav0000086_00000_v \
    attack_type:=fog attack_intensity:=0.7

# 3. With defenses
ros2 launch tracking_adversarial defended.launch.py \
    video_source:=/tracking_ws/data/videos/uav0000086_00000_v \
    attack_type:=fog attack_intensity:=0.7
```

Open **http://localhost:8080** in your browser to see the live dashboard.

### Benchmark

Run all attack configurations and get a comparison table (no ROS2 needed):

```bash
python3 /tracking_ws/scripts/benchmark.py \
    --sequence /tracking_ws/data/videos/uav0000086_00000_v \
    --annotations /tracking_ws/data/annotations/uav0000086_00000_v.txt \
    --model yolov8n.pt
```

## Architecture

Every component is an independent ROS2 node communicating via topics:

```
┌──────────────────┐
│   Video Source    │  Reads frames from video file or image sequence
│  (Detection Node) │  Publishes to /camera/image_raw_source
└────────┬─────────┘
         │
    ┌────▼─────────────┐
    │   Attack Node     │  Applies fog/rain/patches to each frame
    │   (optional)      │  Publishes to /camera/degraded
    └────┬─────────────┘
         │
    ┌────▼─────────────┐
    │  Detection Node   │  YOLOv8 inference on each frame
    │  (YOLOv8)         │  Publishes Detection2DArray to /detections
    └────┬─────────────┘
         │
    ┌────▼─────────────┐
    │  Tracking Node    │  ByteTrack: matches detections across frames
    │  (ByteTrack)      │  Publishes tracked objects to /tracks
    └────┬─────────────┘
         │
    ┌────▼─────────────┐
    │  Defense Node     │  Temporal consistency + anomaly detection
    │  (optional)       │  Publishes corrected tracks to /tracks/defended
    └────┬─────────────┘
         │
    ┌────▼─────────────┐          ┌──────────────────┐
    │  Evaluation Node  │          │  Web Visualizer   │
    │  MOTA/MOTP/IDF1   │          │  localhost:8080   │
    └──────────────────┘          └──────────────────┘
```

## Attacks

| Type | Attack | What It Does |
|------|--------|-------------|
| Environmental | `fog` | Atmospheric scattering model — blends toward white with depth gradient |
| Environmental | `rain` | Random white streaks + blue tint darkening |
| Environmental | `blur` | Directional motion blur convolution kernel |
| Environmental | `low_light` | Gamma darkening + sensor noise + blue shift |
| Environmental | `lens_dirt` | Circular blurred/tinted regions |
| Environmental | `contrast` | Pulls pixel values toward mean gray |
| Adversarial | `patch` | High-frequency noise pattern on detected objects |
| Adversarial | `stripe` | Alternating black/white stripes on objects |
| Adversarial | `checkerboard` | Grid pattern overlaid on objects |
| Adversarial | `occlusion` | Covers a portion of the object with a solid block |
| Gradient | `fgsm_light` | FGSM with epsilon=4/255 — imperceptible perturbation |
| Gradient | `fgsm_heavy` | FGSM with epsilon=16/255 — barely visible perturbation |
| Gradient | `pgd_light` | PGD with epsilon=4/255, 10 steps — stronger than FGSM |
| Gradient | `pgd_heavy` | PGD with epsilon=16/255, 20 steps — near-optimal attack |

Environmental and adversarial attacks accept an `intensity` parameter from 0.0 to 1.0. Gradient attacks use `epsilon` (perturbation budget) instead — configured via preset names.

## Metrics

- **MOTA** (Multi-Object Tracking Accuracy): `1 - (FN + FP + ID_switches) / GT`
- **MOTP** (Multi-Object Tracking Precision): average IoU of matched track-to-ground-truth pairs
- **IDF1**: identity preservation — does each object keep a consistent ID over time

## Project Structure

```
adversarial-tracking/
├── README.md
├── Dockerfile
├── docker-compose.yml
├── scripts/
│   ├── benchmark.py              # Standalone benchmarking (no ROS2 needed)
│   ├── convert_visdrone_to_yolo.py  # Dataset conversion for fine-tuning
│   ├── train_visdrone.py         # YOLOv8 fine-tuning on VisDrone
│   ├── modal_train.py            # Cloud GPU training via Modal (A100)
│   ├── train_kaggle.ipynb        # Kaggle notebook for training
│   └── train_colab.ipynb         # Colab notebook for training
├── tracking_ws/
│   ├── data/
│   │   ├── videos/               # Input sequences
│   │   ├── annotations/          # Ground truth
│   │   └── results/              # Output metrics
│   ├── docs/
│   │   └── project_plan.md       # Detailed walkthrough + learning guide
│   └── src/tracking_adversarial/
│       ├── tracking_adversarial/
│       │   ├── detection_node.py    # YOLOv8 detection
│       │   ├── tracking_node.py     # ByteTrack tracking
│       │   ├── attack_node.py       # Attack application
│       │   ├── defense_node.py      # Defense mechanisms
│       │   ├── evaluation_node.py   # Metrics computation
│       │   ├── web_visualizer.py    # MJPEG web dashboard
│       │   └── utils/
│       │       ├── degradation.py       # Environmental attack implementations
│       │       ├── adversarial.py       # Adversarial attack implementations
│       │       ├── gradient_attacks.py  # FGSM & PGD gradient-based attacks
│       │       ├── defense_utils.py     # Defense algorithm implementations
│       │       └── metrics.py           # MOT metrics (MOTA/MOTP/IDF1)
│       ├── launch/
│       │   ├── baseline.launch.py
│       │   ├── adversarial.launch.py
│       │   ├── defended.launch.py
│       │   └── evaluation.launch.py
│       └── config/
│           └── params.yaml
```

## Tech Stack

- **ROS2 Humble** — node orchestration, message passing, parameter management
- **YOLOv8** (Ultralytics) — real-time object detection, pretrained on COCO
- **ByteTrack** — multi-object tracking with two-stage association
- **OpenCV** — image processing, attack implementations
- **Docker** — containerized ROS2 environment

## Fine-Tuning Pipeline

The pretrained YOLOv8 model was trained on COCO (ground-level photos). It struggles on drone footage because the perspective is completely different. Fine-tuning adapts it to the VisDrone aerial domain.

### Step 1: Dataset Conversion (VisDrone → YOLO)

VisDrone MOT annotations store one line per object per frame:
```
frame_id, track_id, x, y, w, h, confidence, category, truncation, occlusion

Example: 1, 3, 516, 284, 70, 45, 1, 4, 0, 0
         │  │  │    │    │   │     │
         │  │  │    │    │   │     └── category 4 = car
         │  │  │    │    │   └── box height (pixels)
         │  │  │    │    └── box width (pixels)
         │  │  │    └── y top-left corner (pixels)
         │  │  └── x top-left corner (pixels)
         │  └── which tracked object
         └── which frame
```

YOLO needs a completely different format — one `.txt` file per image:
```
class_id  center_x  center_y  width  height    (all normalized 0-1)

Example: 2 0.394643 0.284259 0.050000 0.041667
```

The conversion does two things:

**1. Remap categories.** VisDrone has 10 object categories. We compress to 8 classes:
```
VisDrone 1 (pedestrian) + 2 (people)          → class 0 (pedestrian)
VisDrone 3 (bicycle)                          → class 1 (bicycle)
VisDrone 4 (car)                              → class 2 (car)
VisDrone 5 (van)                              → class 3 (van)
VisDrone 6 (truck)                            → class 4 (truck)
VisDrone 7 (tricycle) + 8 (awning-tricycle)   → class 5 (tricycle)
VisDrone 9 (bus)                              → class 6 (bus)
VisDrone 10 (motor)                           → class 7 (motor)
```

**2. Convert coordinates.** VisDrone gives pixel positions of the top-left corner + size. YOLO wants the center point + size, all divided by image dimensions so values are between 0 and 1:
```
Given: image 1400×1080, box at (516, 284) with size 70×45

center_x = (516 + 70/2) / 1400 = 0.394
center_y = (284 + 45/2) / 1080 = 0.284
width    = 70 / 1400            = 0.050
height   = 45 / 1080            = 0.042
```

The output is a standard YOLO dataset structure:
```
yolo_dataset/
├── dataset.yaml              ← points YOLO to images + class names
├── images/train/             ← 24,201 frames (56 sequences)
├── images/val/               ← 2,846 frames (7 sequences)
├── labels/train/             ← one .txt per image (same filename)
└── labels/val/
```

### Step 2: Training

Uses the Ultralytics library (high-level wrapper around PyTorch) to fine-tune `yolov8n.pt` on the converted dataset. Key settings:
- Lower learning rate (0.001 vs default 0.01) to preserve pretrained features
- Warmup epochs to gradually increase LR at start
- Augmentation (rotation, flip, color shifts, mosaic) to prevent overfitting
- Checkpoints saved every 5 epochs for crash recovery

```bash
# Convert
python3 scripts/convert_visdrone_to_yolo.py \
    --train-dir data/VisDrone2019-MOT-train \
    --val-dir data/VisDrone2019-MOT-val \
    --output data/yolo_dataset

# Train (GPU recommended)
python3 scripts/train_visdrone.py \
    --data data/yolo_dataset/dataset.yaml \
    --epochs 50 --device 0

# Or use Modal for cloud A100 GPU:
modal run scripts/modal_train.py
```

## Roadmap

See [docs/project_plan.md](tracking_ws/docs/project_plan.md) for the full learning roadmap, from current state through gradient-based adversarial attacks and adversarial training.
