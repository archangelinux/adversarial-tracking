# Adversarial Perception Testing Framework

A ROS2-based framework for evaluating how adversarial conditions degrade object detection and multi-object tracking in drone surveillance, and measuring the effectiveness of defenses.

```
                     clean frame                    foggy frame
                    +───────────+                  +───────────+
                    │  +──+     │                  │           │
                    │  │##│ car │      fog         │  ?    ?   │
                    │  +──+     │  ──────────>     │           │
                    │     +──+  │   attack         │     ?     │
                    │     │##│  │                  │           │
                    +───────────+                  +───────────+
```

**Core question:** *How fragile is a modern perception pipeline under adversarial conditions, and how much does domain-specific fine-tuning help?*

## Architecture

The system is a pipeline of independent ROS2 nodes communicating via pub/sub topics. Each node handles one responsibility, making it straightforward to swap components (e.g. replace YOLOv8 with another detector) or add/remove attack and defense stages without modifying other nodes.

```
+-────────────────-+
│   Video Source    │  Reads VisDrone image sequences
│  (Detection Node) │  Publishes raw frames
+────────┬─────────+
         │
    +────v─────────────+
    │   Attack Node     │  Applies environmental or adversarial perturbations
    │   (optional)      │  14 attack types with configurable intensity
    +────┬─────────────+
         │
    +────v─────────────+
    │  Detection Node   │  YOLOv8 inference → bounding boxes + class + confidence
    │  (YOLOv8)         │
    +────┬─────────────+
         │
    +────v─────────────+
    │  Tracking Node    │  ByteTrack: two-stage association with Kalman filtering
    │  (ByteTrack)      │  Assigns persistent IDs across frames
    +────┬─────────────+
         │
    +────v─────────────+
    │  Defense Node     │  Temporal consistency checking + anomaly detection
    │  (optional)       │  Corrects impossible position jumps
    +────┬─────────────+
         │
    +────v─────────────+          +──────────────────+
    │  Evaluation Node  │          │  Web Visualizer   │
    │  MOTA / MOTP      │          │  localhost:8080   │
    +──────────────────+          +──────────────────+
```

## Tracking: ByteTrack

Detection alone only gives per-frame bounding boxes with no identity. ByteTrack links detections across frames into persistent tracks so each object maintains a consistent ID over time (e.g. "car #7 in frame 1 is the same car #7 in frame 200").

**Two-stage association** is ByteTrack's core design choice:

1. **High-confidence match** — detections above the confidence threshold (0.5) are matched to existing tracks using IoU (Intersection over Union — how much two bounding boxes overlap). Optimal matching is solved with the Hungarian algorithm.
2. **Low-confidence match** — detections between 0.1 and 0.5 confidence are matched to tracks still unmatched after stage 1. This recovers partially occluded or blurred objects that other trackers would discard entirely.

**Kalman filtering** smooths each track's trajectory. Each track maintains a state vector `[cx, cy, w, h, vx, vy, vw, vh]` (center position, size, and their velocities). Every frame, the filter predicts where the object should be based on its velocity, then corrects that prediction with the actual matched detection. During brief occlusions (object disappears for a few frames), the filter continues predicting position from velocity until the object reappears or the track times out after 30 frames.

**What's slow in benchmarking** is not ByteTrack — its association step is just NumPy matrix operations and takes microseconds per frame. The bottleneck is YOLOv8 inference: one full neural network forward pass per frame, at 2,846 frames per run.

## Attack Taxonomy

The framework implements 14 attacks across three categories, each modeling a different threat scenario:

| Category | Attack | Mechanism | Threat Model |
|----------|--------|-----------|--------------|
| Environmental | `fog` | Atmospheric scattering with depth gradient | Naturally occurring |
| Environmental | `rain` | Streak overlays + blue tint darkening | Naturally occurring |
| Environmental | `blur` | Directional convolution kernel | Camera shake / motion |
| Environmental | `low_light` | Gamma darkening + sensor noise | Nighttime operation |
| Environmental | `lens_dirt` | Circular blurred regions | Hardware degradation |
| Environmental | `contrast` | Pixel values pulled toward mean gray | Sensor limitations |
| Adversarial | `patch` | High-frequency noise pattern on objects | Physical adversarial patches |
| Adversarial | `stripe` | Alternating B/W stripes on objects | Adversarial clothing/wraps |
| Adversarial | `checkerboard` | Grid pattern overlay on objects | Adversarial surface patterns |
| Adversarial | `occlusion` | Solid block covering part of object | Physical occlusion |
| Gradient | `fgsm_light` | FGSM, epsilon=4/255 | White-box, imperceptible |
| Gradient | `fgsm_heavy` | FGSM, epsilon=16/255 | White-box, barely visible |
| Gradient | `pgd_light` | PGD, epsilon=4/255, 10 steps | White-box, stronger than FGSM |
| Gradient | `pgd_heavy` | PGD, epsilon=16/255, 20 steps | White-box, near-optimal |

**Environmental vs adversarial vs gradient:** Environmental attacks degrade the whole image in ways a human would also struggle with. Adversarial pattern attacks target specific detected objects, modeling physical-world attacks like adversarial patches on clothing. Gradient attacks (FGSM/PGD) compute pixel perturbations using the model's own gradients — imperceptible to humans but devastating to the detector. The gradient attacks are implemented from scratch against YOLOv8's detection head (not using torchattacks, which targets classifiers).

## Defense Mechanisms

Two heuristic defenses operate on tracker output:

- **Temporal consistency** — maintains a position history per track and predicts expected positions from weighted velocity averaging. When a detection jumps further than a configurable threshold, the predicted position is substituted. This catches cases where attacks cause brief detection failures or position shifts.

- **Anomaly detection** — computes running z-scores on velocity and acceleration for each track. Tracks exceeding 3 standard deviations are flagged as anomalous, signaling potential adversarial interference to downstream systems.

## Evaluation Metrics

- **MOTA** (Multi-Object Tracking Accuracy): `1 - (FN + FP + IDSW) / GT`
  - **FN** (False Negatives) — ground truth objects the tracker missed
  - **FP** (False Positives) — detections with no corresponding ground truth object
  - **IDSW** (ID Switches) — a tracked object changes identity between frames
  - **GT** (Ground Truth) — total annotated objects across all frames
  - MOTA is the primary metric. It can go negative when total errors exceed the ground truth count.

- **MOTP** (Multi-Object Tracking Precision): `sum(IoU) / TP`
  - **IoU** (Intersection over Union) — overlap ratio between predicted and ground truth boxes
  - **TP** (True Positives) — correctly matched detections
  - Measures localization quality independent of detection rate.

For multi-sequence evaluation, metrics are aggregated by summing raw counts (TP, FP, FN, IDSW) across all sequences before computing MOTA/MOTP. This avoids giving equal weight to short and long sequences.

## Benchmark Results

Evaluated on the full VisDrone2019-MOT validation set (7 sequences, 2,846 frames, 108,645 ground truth annotations):

| Model | MOTA | MOTP | TP | FP | FN | ID Switches |
|-------|------|------|------|------|--------|-------------|
| YOLOv8n pretrained (COCO) | 0.107 | 0.799 | 15,243 | 1,201 | 93,402 | 2,420 |
| YOLOv8n fine-tuned (VisDrone) | 0.207 | 0.762 | 39,071 | 7,902 | 69,574 | 8,630 |

**Why MOTA is low:** VisDrone is one of the hardest MOT benchmarks. Objects are tiny (often 10-30 pixels), densely packed (averaging ~38 per frame), and viewed from drone altitude — a perspective COCO never trained on. Published state-of-the-art results on VisDrone-MOT with full-size models achieve MOTA 0.3-0.5. YOLOv8n (nano) is the smallest and fastest variant, trading accuracy for speed. These numbers are expected for this model size on this dataset.

**Key observations:**
- Fine-tuning nearly **doubled MOTA** (0.107 → 0.207) and increased true positives by **156%** (15K → 39K), confirming that domain adaptation is essential for aerial footage.
- The pretrained model's high MOTP (0.799) but low MOTA (0.107) reveals it detects few objects but localizes them well — it only finds the easy, large targets. Fine-tuning trades slightly lower MOTP for dramatically better recall.
- The increase in ID switches (2.4K → 8.6K) is expected: more detections means more opportunities for the tracker to confuse identities, especially in VisDrone's dense, small-object scenes. This is a tracker tuning issue, not a model quality problem.

## Fine-Tuning Pipeline

The pretrained YOLOv8n was trained on COCO (330K ground-level photos). Drone footage has fundamentally different characteristics: top-down perspective, small and densely packed objects, and different scale distributions. Fine-tuning adapts the model to this domain.

### Dataset conversion (VisDrone → YOLO format)

VisDrone MOT stores per-object-per-frame annotations (`frame_id, track_id, x, y, w, h, confidence, category, ...`). YOLO expects one `.txt` file per image with normalized center coordinates (`class_id center_x center_y width height`). The conversion script handles coordinate transformation and category remapping (VisDrone's 10 categories → 8 YOLO classes, merging pedestrian/people and tricycle/awning-tricycle).

### Training approach

- **Low learning rate (0.001)** instead of freezing layers — with 24K training images there's no overfitting risk, and allowing all layers to update slightly lets early features adapt to aerial-specific edge patterns while late layers retrain fully. This is effectively a soft version of selective layer freezing.
- **Standard augmentation** (mosaic, rotation, color shifts) to improve generalization.
- **Checkpoint recovery** for resuming interrupted training runs.

## Quick Start

### Prerequisites
- Docker Desktop

### Setup

```bash
git clone <this-repo>
cd adversarial-tracking
docker compose build
docker compose up -d
docker exec -it tracking_adversarial bash
```

### Running the ROS2 Pipeline

```bash
source /tracking_ws/install/setup.bash

# Baseline — clean detection + tracking
ros2 launch tracking_adversarial baseline.launch.py \
    video_source:=/tracking_ws/data/videos/uav0000086_00000_v

# Under attack — add fog at 70% intensity
ros2 launch tracking_adversarial adversarial.launch.py \
    video_source:=/tracking_ws/data/videos/uav0000086_00000_v \
    attack_type:=fog attack_intensity:=0.7

# With defenses enabled
ros2 launch tracking_adversarial defended.launch.py \
    video_source:=/tracking_ws/data/videos/uav0000086_00000_v \
    attack_type:=fog attack_intensity:=0.7
```

Open **http://localhost:8080** for the live web dashboard.

### Standalone Benchmark (no ROS2)

```bash
# Single sequence
python3 scripts/benchmark.py \
    --sequence /tracking_ws/data/videos/uav0000086_00000_v \
    --annotations /tracking_ws/data/annotations/uav0000086_00000_v.txt \
    --model yolov8n.pt

# All sequences in a VisDrone split
python3 scripts/benchmark.py \
    --sequence-dir /tracking_ws/data/VisDrone2019-MOT-val \
    --model yolov8n.pt \
    --baseline-only
```

### Fine-Tuning

```bash
# Convert VisDrone annotations to YOLO format
python3 scripts/convert_visdrone_to_yolo.py \
    --train-dir data/VisDrone2019-MOT-train \
    --val-dir data/VisDrone2019-MOT-val \
    --output data/yolo_dataset

# Train (GPU recommended — or use Modal for cloud A100)
python3 scripts/train_visdrone.py \
    --data data/yolo_dataset/dataset.yaml \
    --epochs 50 --device 0

# Cloud training via Modal
modal run scripts/modal_train.py
```

## Project Structure

```
adversarial-tracking/
├── Dockerfile                          # ROS2 Humble + YOLOv8 + dependencies
├── docker-compose.yml
├── scripts/
│   ├── benchmark.py                    # Standalone benchmarking across all attacks
│   ├── convert_visdrone_to_yolo.py     # VisDrone MOT → YOLO format conversion
│   ├── train_visdrone.py               # YOLOv8 fine-tuning on VisDrone
│   └── modal_train.py                  # Cloud GPU training via Modal (A100)
└── tracking_ws/
    └── src/tracking_adversarial/
        ├── tracking_adversarial/
        │   ├── detection_node.py       # YOLOv8 detection + video source
        │   ├── tracking_node.py        # ByteTrack multi-object tracker
        │   ├── attack_node.py          # Attack application (env + adversarial)
        │   ├── defense_node.py         # Temporal consistency + anomaly detection
        │   ├── evaluation_node.py      # MOTA/MOTP computation vs ground truth
        │   ├── visualization_node.py   # RViz2 marker publisher
        │   ├── web_visualizer.py       # MJPEG web dashboard (port 8080)
        │   └── utils/
        │       ├── degradation.py      # Environmental attacks (fog, rain, blur, ...)
        │       ├── adversarial.py      # Adversarial pattern attacks (patch, stripe, ...)
        │       ├── gradient_attacks.py # FGSM + PGD (custom, not torchattacks)
        │       ├── defense_utils.py    # Defense algorithms
        │       └── metrics.py          # MOT metric accumulation
        ├── launch/                     # baseline / adversarial / defended / evaluation
        └── config/params.yaml          # All tunable parameters
```

## Tech Stack

- **ROS2 Humble** — node orchestration, pub/sub messaging, runtime parameter management
- **YOLOv8** (Ultralytics) — real-time object detection, pretrained on COCO
- **ByteTrack** — multi-object tracking with two-stage association (high + low confidence)
- **PyTorch** — gradient-based attack computation (FGSM/PGD)
- **OpenCV** — image processing for environmental and adversarial attacks
- **Docker** — containerized ROS2 environment with all dependencies
