# tracking_adversarial (ROS2 Package)

ROS2 package containing all nodes for the adversarial perception testing framework.

## Nodes

| Node | Executable | Subscribes | Publishes |
|------|-----------|------------|-----------|
| DetectionNode | `detection_node` | `/camera/image_raw`, `/camera/degraded` | `/detections`, `/camera/image_raw_source` |
| TrackingNode | `tracking_node` | `/detections` | `/tracks` |
| AttackNode | `attack_node` | `/camera/image_raw_source`, `/detections` | `/camera/degraded` |
| DefenseNode | `defense_node` | `/tracks` | `/tracks/defended` |
| EvaluationNode | `evaluation_node` | `/tracks` | `/evaluation/metrics` |
| WebVisualizer | `web_visualizer` | `/camera/image_raw_source`, `/camera/degraded`, `/detections`, `/tracks`, `/tracks/defended` | HTTP MJPEG at :8080 |

## Topic Flow

```
/camera/image_raw_source  ──► attack_node ──► /camera/degraded
                                                    │
/camera/image_raw ◄──── (or clean) ────────────────┘
        │                                           │
        ▼                                           ▼
  detection_node ◄─────────────────────── detection_node
        │                                  (auto-switches to degraded)
        ▼
   /detections ──► tracking_node ──► /tracks ──► defense_node ──► /tracks/defended
                                        │                              │
                                        ▼                              ▼
                                   web_visualizer               web_visualizer
                                   evaluation_node
```

## Launch Files

- `baseline.launch.py` — detection + tracking + web visualizer (no attacks)
- `adversarial.launch.py` — adds attack node
- `defended.launch.py` — adds attack + defense nodes
- `evaluation.launch.py` — configurable with ground truth comparison

## Parameters

All parameters are in `config/params.yaml`. Key parameters can also be set via launch arguments:

```bash
ros2 launch tracking_adversarial adversarial.launch.py \
    video_source:=/path/to/sequence \
    model:=yolov8n.pt \
    device:=cpu \
    attack_type:=fog \
    attack_intensity:=0.7
```

Runtime parameter changes:
```bash
ros2 param set /attack_node attack.type rain
ros2 param set /attack_node attack.intensity 0.5
ros2 param set /attack_node attack.enabled false
```

## Build

```bash
cd /tracking_ws
colcon build --packages-select tracking_adversarial
source install/setup.bash
```
