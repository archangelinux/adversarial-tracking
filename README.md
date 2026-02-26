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

**Core question:** *How fragile is a modern perception pipeline under adversarial conditions, how much does domain-specific fine-tuning help, and does a newer architecture (YOLO26) offer better adversarial robustness than YOLOv8?*

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

**Environmental vs adversarial vs gradient:** Environmental attacks degrade the whole image in ways a human would also struggle with. Adversarial pattern attacks target specific detected objects, modeling physical-world attacks like adversarial patches on clothing. Gradient attacks (FGSM/PGD) compute pixel perturbations using the model's own gradients — imperceptible to humans but devastating to the detector.

### The NMS bypass problem

Gradient attacks require computing gradients through the entire model — a smooth, continuous path from output back to input pixels. Standard YOLO inference ends with **Non-Maximum Suppression (NMS)**, a post-processing step that filters thousands of overlapping candidate boxes down to clean detections: keep the highest-confidence box, discard all boxes that overlap it above an IoU threshold, repeat. NMS is a hard keep/discard selection, not a differentiable operation, so gradients can't flow through it.

Libraries like `torchattacks` target classifiers (one output label per image), where this problem doesn't exist. For detection models, the gradient attacks are implemented from scratch: the `YOLOAttackWrapper` in `gradient_attacks.py` bypasses NMS and attacks the **raw detection grid** — the tensor of all candidate predictions before filtering. The loss function operates on these raw predictions, so gradients flow cleanly from the detection head back to the input pixels. Perturbations that corrupt the raw predictions cause the final post-NMS detections to fail.

**YOLO26 compatibility:** YOLO26 introduces an NMS-free end-to-end design with a one-to-one detection head. However, this head explicitly detaches gradients internally (since it's designed for inference, not training through). The wrapper disables `end2end` mode to fall back to the one-to-many head, which produces raw predictions in the same format as YOLOv8 with gradients intact.

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

Evaluated on the full VisDrone2019-MOT validation set (7 sequences, 2,846 frames, 108,645 ground truth annotations). Four models tested: YOLOv8n and YOLO26n, each pretrained (COCO) and fine-tuned (VisDrone).

### Baseline Performance (Clean, No Attacks)

| Model | MOTA | MOTP | TP | FP | FN | ID Switches |
|-------|------|------|--------|--------|--------|-------------|
| YOLOv8n pretrained | 0.107 | 0.799 | 15,243 | 1,201 | 93,402 | 2,420 |
| YOLOv8n fine-tuned | 0.334 | 0.762 | 49,677 | 8,985 | 58,968 | 8,372 |
| YOLO26n pretrained | 0.087 | 0.813 | 12,656 | 882 | 95,989 | 1,769 |
| YOLO26n fine-tuned | 0.195 | 0.768 | 36,453 | 8,436 | 72,192 | 6,848 |

### Performance Under Attack (MOTA)

Non-gradient attacks run locally on CPU; gradient attacks (FGSM/PGD) run on A100 GPU via Modal.

| Attack | v8 pretrained | v8 fine-tuned | v26 pretrained | v26 fine-tuned |
|--------|--------------|--------------|----------------|----------------|
| Baseline (clean) | 0.107 | 0.334 | 0.087 | 0.195 |
| Fog (light) | — | 0.335 | 0.076 | 0.191 |
| Fog (heavy) | — | 0.258 | 0.054 | 0.148 |
| Rain (light) | — | 0.303 | 0.081 | 0.188 |
| Rain (heavy) | — | 0.243 | 0.061 | 0.166 |
| Blur (light) | — | 0.147 | 0.058 | 0.143 |
| Blur (heavy) | — | 0.011 | 0.045 | 0.084 |
| Low light | — | 0.186 | 0.050 | 0.117 |
| Low light (extreme) | — | 0.000 | 0.001 | 0.002 |
| Contrast loss | — | 0.343 | 0.079 | 0.193 |
| Adv. patch | — | 0.134 | -0.003 | -0.006 |
| Adv. stripe | — | 0.003 | -0.004 | -0.029 |
| Checkerboard | — | 0.011 | -0.003 | -0.008 |
| Occlusion | — | 0.092 | 0.016 | 0.041 |
| FGSM (light) | 0.101 | 0.205 | 0.085 | 0.201 |
| FGSM (heavy) | 0.066 | 0.153 | 0.061 | 0.152 |
| PGD (light) | 0.103 | 0.209 | 0.086 | 0.205 |
| PGD (heavy) | 0.091 | 0.192 | 0.078 | 0.190 |

### MOTA Drop from Baseline (%)

| Attack | v8 fine-tuned | v26 fine-tuned |
|--------|--------------|----------------|
| Fog (heavy) | -22.8% | -24.1% |
| Rain (heavy) | -27.4% | -15.0% |
| Blur (light) | -56.0% | -26.5% |
| Blur (heavy) | -96.8% | -56.7% |
| Low light | -44.3% | -39.9% |
| Low light (extreme) | -100% | -99.0% |
| Adv. stripe | -99.1% | -114.9% |
| Checkerboard | -96.8% | -104.3% |
| Occlusion | -72.5% | -78.7% |
| FGSM (light) | -38.7% | -3.0%\* |
| FGSM (heavy) | -54.4% | -21.8% |
| PGD (light) | -37.6% | -5.1%\* |
| PGD (heavy) | -42.6% | -2.2% |

\*YOLO26 fine-tuned shows marginal improvement under light gradient attacks — likely within methodological noise (baseline measured on CPU, gradient attacks on GPU with different inference head configuration).

### Key Findings

1. **Fine-tuning is essential.** Both models roughly doubled or tripled baseline MOTA after fine-tuning on VisDrone. Domain adaptation matters more than architecture choice for aerial tracking.

2. **YOLOv8n outperforms YOLO26n at nano scale.** Despite being a newer architecture with attention mechanisms, YOLO26n's design overhead hurts at the smallest model size (0.334 vs 0.195 fine-tuned baseline). The architectural innovations likely need larger model variants (s/m/l) to pay off.

3. **YOLO26 fine-tuned is significantly more robust to gradient attacks.** PGD heavy drops YOLOv8 fine-tuned by 42.6% but YOLO26 fine-tuned by only 2.2%. This is the most notable result — the attention-based architecture appears to diffuse adversarial gradients, making white-box attacks less effective even though baseline performance is lower.

4. **Adversarial pattern attacks are devastating regardless of architecture.** Stripe, checkerboard, and patch attacks drive MOTA to zero or negative across all models. These attacks directly corrupt object appearance and overwhelm any detector.

5. **Environmental attacks show a robustness-accuracy tradeoff.** YOLO26 fine-tuned shows smaller relative drops under rain and blur despite lower absolute MOTA, suggesting the architecture degrades more gracefully under input corruption.

6. **Low light (extreme) is a universal failure mode.** All models collapse to MOTA ~0.000 at 80% intensity — this represents a fundamental limit of visual perception, not a model-specific weakness.

**Why MOTA is low overall:** VisDrone is one of the hardest MOT benchmarks. Objects are tiny (often 10-30 pixels), densely packed (~38 per frame), and viewed from drone altitude. Published state-of-the-art results with full-size models and 1280px input achieve MOTA 0.3-0.5. This study uses nano (n) variants at 640px resolution for computational feasibility. The focus is on **relative degradation** under attack rather than absolute tracking performance.

## Fine-Tuning Pipeline

Both YOLOv8n and YOLO26n are pretrained on COCO (330K ground-level photos). Drone footage has fundamentally different characteristics: top-down perspective, small and densely packed objects, and different scale distributions. Fine-tuning adapts each model to the VisDrone domain.

### Dataset conversion (VisDrone → YOLO format)

VisDrone MOT stores per-object-per-frame annotations (`frame_id, track_id, x, y, w, h, confidence, category, ...`). YOLO expects one `.txt` file per image with normalized center coordinates (`class_id center_x center_y width height`). The conversion script handles coordinate transformation and category remapping (VisDrone's 10 categories → 8 YOLO classes, merging pedestrian/people and tricycle/awning-tricycle).

### Training approach

Both models were trained on Modal (A100 GPU) with identical hyperparameters for fair comparison:

- **50 epochs** with early stopping (patience 100)
- **Low learning rate (0.001)** instead of freezing layers — with 24K training images there's no overfitting risk, and allowing all layers to update slightly lets early features adapt to aerial-specific edge patterns while late layers retrain fully.
- **Standard augmentation** (mosaic, mixup, rotation, color shifts) to improve generalization.
- **640px input resolution** for computational feasibility (1280px would improve small object detection but increase training and benchmark time significantly).
- **Checkpoint recovery** for resuming interrupted training runs.

## Design Decisions

### Why Ultralytics is only used for detection

Ultralytics YOLO provides built-in modes for tracking (`model.track()` with ByteTrack), validation (mAP), and benchmarking (inference speed across export formats). This project intentionally uses only **Train** and **Predict** modes, building everything else custom:

- **Tracking** — Ultralytics' built-in `model.track()` is a black box. The ROS2 architecture requires a separate tracking node so attack and defense stages can be inserted between detection and tracking. Implementing ByteTrack from scratch also gives direct access to Kalman filter states, association thresholds, and track lifecycle — all of which the defense node inspects to detect anomalies.

- **Evaluation** — Ultralytics' `val` mode computes detection metrics (mAP). This project needs **tracking** metrics (MOTA/MOTP), which measure identity consistency across frames — something Ultralytics doesn't provide. The custom `MOTMetrics` class computes these from raw TP/FP/FN/IDSW counts.

- **Gradient attacks** — Ultralytics doesn't expose the raw detection grid needed for backpropagation. The custom FGSM/PGD implementation wraps YOLOv8's detection head directly, bypassing NMS (which isn't differentiable) to compute gradients against the raw prediction tensor. Libraries like `torchattacks` target classifiers, not detection models.

- **Benchmarking** — Ultralytics' benchmark mode measures inference speed across export formats (ONNX, TensorRT). This project's benchmark measures tracking accuracy degradation under 14 different adversarial attacks — a fundamentally different evaluation.

### Why ROS2

Each pipeline stage (source → attack → detection → tracking → defense → evaluation) runs as an independent node communicating via pub/sub topics. This makes it possible to add or remove attack and defense stages at launch time without modifying any node code. The same detection node works in baseline, adversarial, and defended configurations — only the launch file changes.

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
│   ├── modal_train.py                  # Cloud GPU training via Modal (A100)
│   ├── modal_train_yolo26.py           # YOLO26 fine-tuning on Modal (A100)
│   ├── modal_benchmark_gradient.py     # Gradient attack benchmarks on Modal (A100)
│   └── run_all_benchmarks.sh           # Run all missing benchmarks (Modal + local)
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
- **YOLOv8 + YOLO26** (Ultralytics) — real-time object detection, pretrained on COCO, fine-tuned on VisDrone
- **ByteTrack** — multi-object tracking with two-stage association (high + low confidence)
- **PyTorch** — gradient-based attack computation (FGSM/PGD)
- **OpenCV** — image processing for environmental and adversarial attacks
- **Docker** — containerized ROS2 environment with all dependencies
- **Modal** — cloud GPU (A100) for fine-tuning and gradient attack benchmarks
