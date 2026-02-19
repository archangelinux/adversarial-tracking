"""Standalone benchmark script — no ROS2 needed.

Reads VisDrone sequences, runs YOLOv8 + ByteTrack with various attack
configurations, compares against ground truth, and prints a comparison table.

Usage (inside Docker):
    python3 /tracking_ws/scripts/benchmark.py \
        --sequence /tracking_ws/data/videos/uav0000086_00000_v \
        --annotations /tracking_ws/data/annotations/uav0000086_00000_v.txt \
        --model yolov8n.pt

This is where the real research happens — you get numbers showing exactly
how much each attack degrades tracking performance.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

# Add the package source so we can import attack utilities directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tracking_ws' / 'src' / 'tracking_adversarial'))
from tracking_adversarial.utils.degradation import apply_degradation
from tracking_adversarial.utils.adversarial import apply_adversarial_attack


# VisDrone category mapping → COCO class IDs
# VisDrone: 1=pedestrian, 2=people, 3=bicycle, 4=car, 5=van, 6=truck,
#           7=tricycle, 8=awning-tricycle, 9=bus, 10=motor
# We only care about categories that roughly map to COCO classes YOLO knows.
VISDRONE_TO_COCO = {
    1: 0,   # pedestrian → person
    2: 0,   # people → person
    3: 1,   # bicycle → bicycle
    4: 2,   # car → car
    5: 7,   # van → truck (closest COCO match)
    6: 7,   # truck → truck
    9: 5,   # bus → bus
    10: 3,  # motor → motorcycle
}


# ── Minimal ByteTrack (same logic as the ROS2 node, self-contained) ──────────

class KalmanBoxTracker:
    _count = 0

    def __init__(self, bbox, score, cls_id):
        self.id = KalmanBoxTracker._count
        KalmanBoxTracker._count += 1
        self.bbox = np.array(bbox, dtype=float)
        self.score = score
        self.cls_id = cls_id
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.velocity = np.zeros(4)

    def predict(self):
        self.bbox += self.velocity
        self.age += 1
        self.time_since_update += 1
        return self.bbox

    def update(self, bbox, score, cls_id):
        old = self.bbox.copy()
        self.bbox = np.array(bbox, dtype=float)
        self.velocity = 0.7 * self.velocity + 0.3 * (self.bbox - old)
        self.score = score
        self.cls_id = cls_id
        self.hits += 1
        self.time_since_update = 0


def iou_matrix(boxes_a, boxes_b):
    a = np.array(boxes_a)
    b = np.array(boxes_b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


class SimpleByteTrack:
    def __init__(self, high_thresh=0.5, low_thresh=0.1, match_thresh=0.8, max_age=30):
        self.trackers = []
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.max_age = max_age
        KalmanBoxTracker._count = 0

    def update(self, detections):
        """detections: list of (x1, y1, x2, y2, score, cls_id)"""
        for t in self.trackers:
            t.predict()

        high = [d for d in detections if d[4] >= self.high_thresh]
        low = [d for d in detections if self.low_thresh <= d[4] < self.high_thresh]

        # First association — high confidence
        unmatched_trks = list(range(len(self.trackers)))
        if high and self.trackers:
            ious = iou_matrix(
                [d[:4] for d in high],
                [self.trackers[i].bbox for i in unmatched_trks]
            )
            cost = 1 - ious
            rows, cols = linear_sum_assignment(cost)
            matched_det = set()
            new_unmatched_trks = []
            for r, c in zip(rows, cols):
                if ious[r, c] >= self.match_thresh:
                    self.trackers[unmatched_trks[c]].update(
                        high[r][:4], high[r][4], high[r][5])
                    matched_det.add(r)
                else:
                    new_unmatched_trks.append(unmatched_trks[c])
            for c in range(len(unmatched_trks)):
                if c not in cols or (c in cols and unmatched_trks[c] not in [unmatched_trks[cols[np.where(rows == rows[np.where(cols == c)])[0][0]]] if c in cols else -1 for _ in [0]]):
                    pass
            # Simpler: just collect unmatched
            matched_trk_indices = set()
            for r, c in zip(rows, cols):
                if ious[r, c] >= self.match_thresh:
                    matched_trk_indices.add(unmatched_trks[c])
            unmatched_trks = [i for i in unmatched_trks if i not in matched_trk_indices]
            unmatched_high = [high[i] for i in range(len(high)) if i not in matched_det]
        else:
            unmatched_high = list(high)

        # Second association — low confidence with remaining trackers
        if low and unmatched_trks:
            ious = iou_matrix(
                [d[:4] for d in low],
                [self.trackers[i].bbox for i in unmatched_trks]
            )
            cost = 1 - ious
            rows, cols = linear_sum_assignment(cost)
            matched_trk_indices2 = set()
            for r, c in zip(rows, cols):
                if ious[r, c] >= self.match_thresh:
                    self.trackers[unmatched_trks[c]].update(
                        low[r][:4], low[r][4], low[r][5])
                    matched_trk_indices2.add(unmatched_trks[c])
            unmatched_trks = [i for i in unmatched_trks if i not in matched_trk_indices2]

        # Create new tracks for unmatched high-confidence detections
        for d in unmatched_high:
            if d[4] >= self.high_thresh:
                self.trackers.append(KalmanBoxTracker(d[:4], d[4], d[5]))

        # Remove dead tracks
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        # Return active tracks
        results = []
        for t in self.trackers:
            if t.time_since_update == 0:
                results.append((t.id, *t.bbox, t.score, t.cls_id))
        return results


# ── Ground truth loader ──────────────────────────────────────────────────────

def load_visdrone_gt(annotation_path):
    """Load VisDrone MOT annotations.

    Format: frame_id, track_id, x, y, w, h, confidence, category, truncation, occlusion
    Returns: dict[frame_id] → list of (track_id, x1, y1, x2, y2, category)
    """
    gt = {}
    with open(annotation_path) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 8:
                continue
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
            category = int(parts[7])

            # Skip ignored regions (category 0) and categories we don't track
            if category not in VISDRONE_TO_COCO:
                continue

            if frame_id not in gt:
                gt[frame_id] = []
            gt[frame_id].append((track_id, x, y, x + w, y + h, category))
    return gt


# ── MOT metrics calculator ───────────────────────────────────────────────────

class MOTMetrics:
    """Tracks MOTA, MOTP, and ID switches across frames."""

    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.total_gt = 0
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_id_switches = 0
        self.total_iou = 0.0
        self.prev_matches = {}  # gt_id → track_id from previous frame

    def update(self, gt_boxes, track_boxes):
        """
        gt_boxes: list of (gt_id, x1, y1, x2, y2)
        track_boxes: list of (track_id, x1, y1, x2, y2)
        """
        self.total_gt += len(gt_boxes)

        if not gt_boxes:
            self.total_fp += len(track_boxes)
            return

        if not track_boxes:
            self.total_fn += len(gt_boxes)
            return

        # Compute IoU matrix
        gt_arr = np.array([b[1:5] for b in gt_boxes])
        tr_arr = np.array([b[1:5] for b in track_boxes])
        ious = iou_matrix_raw(gt_arr, tr_arr)

        # Hungarian matching
        cost = 1 - ious
        rows, cols = linear_sum_assignment(cost)

        matched_gt = set()
        matched_tr = set()
        current_matches = {}

        for r, c in zip(rows, cols):
            if ious[r, c] >= self.iou_threshold:
                matched_gt.add(r)
                matched_tr.add(c)
                gt_id = gt_boxes[r][0]
                tr_id = track_boxes[c][0]
                current_matches[gt_id] = tr_id
                self.total_tp += 1
                self.total_iou += ious[r, c]

                # Check for ID switch
                if gt_id in self.prev_matches and self.prev_matches[gt_id] != tr_id:
                    self.total_id_switches += 1

        self.total_fn += len(gt_boxes) - len(matched_gt)
        self.total_fp += len(track_boxes) - len(matched_tr)
        self.prev_matches = current_matches

    def mota(self):
        if self.total_gt == 0:
            return 0.0
        return 1.0 - (self.total_fn + self.total_fp + self.total_id_switches) / self.total_gt

    def motp(self):
        if self.total_tp == 0:
            return 0.0
        return self.total_iou / self.total_tp

    def summary(self):
        return {
            'MOTA': self.mota(),
            'MOTP': self.motp(),
            'TP': self.total_tp,
            'FP': self.total_fp,
            'FN': self.total_fn,
            'ID_Sw': self.total_id_switches,
            'GT': self.total_gt,
        }


def iou_matrix_raw(a, b):
    """IoU between two sets of [x1,y1,x2,y2] boxes."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


# ── Run one configuration ────────────────────────────────────────────────────

def run_config(model, image_files, gt, attack_type, intensity, conf_thresh=0.25):
    """Run detection + tracking on a full sequence with a given attack config.

    Returns MOTMetrics summary dict.
    """
    tracker = SimpleByteTrack()
    metrics = MOTMetrics()

    for i, img_path in enumerate(image_files):
        frame_id = i + 1  # VisDrone frames are 1-indexed
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        # Apply attack
        if attack_type != 'none':
            if attack_type in ('rain', 'fog', 'blur', 'low_light', 'lens_dirt', 'contrast'):
                frame = apply_degradation(frame, attack_type, intensity)
            elif attack_type in ('patch', 'stripe', 'checkerboard', 'occlusion'):
                # For adversarial attacks on objects, use GT boxes as targets
                bboxes = None
                if frame_id in gt:
                    bboxes = [(b[1], b[2], b[3], b[4]) for b in gt[frame_id]]
                frame = apply_adversarial_attack(
                    frame, attack_type, bboxes=bboxes, intensity=intensity)

        # Run YOLO
        results = model(frame, conf=conf_thresh, iou=0.45, imgsz=640,
                        classes=[0, 1, 2, 3, 5, 7], verbose=False)

        # Extract detections as (x1, y1, x2, y2, score, cls_id)
        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    score = float(box.conf[0].item())
                    cls_id = int(box.cls[0].item())
                    detections.append((x1, y1, x2, y2, score, cls_id))

        # Run tracker
        tracks = tracker.update(detections)

        # Compare to ground truth
        gt_frame = gt.get(frame_id, [])
        gt_boxes = [(b[0], b[1], b[2], b[3], b[4]) for b in gt_frame]
        tr_boxes = [(int(t[0]), t[1], t[2], t[3], t[4]) for t in tracks]
        metrics.update(gt_boxes, tr_boxes)

    return metrics.summary()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Benchmark tracking under attacks')
    parser.add_argument('--sequence', required=True, help='Path to image sequence folder')
    parser.add_argument('--annotations', required=True, help='Path to VisDrone annotation file')
    parser.add_argument('--model', default='yolov8n.pt', help='YOLOv8 model path')
    parser.add_argument('--device', default='cpu', help='Inference device')
    parser.add_argument('--max-frames', type=int, default=0, help='Limit frames (0 = all)')
    args = parser.parse_args()

    # Load model
    print(f'Loading model: {args.model}')
    model = YOLO(args.model)
    model.to(args.device)

    # Load images
    seq_path = Path(args.sequence)
    extensions = {'.jpg', '.jpeg', '.png'}
    image_files = sorted([
        str(f) for f in seq_path.iterdir()
        if f.suffix.lower() in extensions
    ])
    if args.max_frames > 0:
        image_files = image_files[:args.max_frames]
    print(f'Sequence: {seq_path.name} ({len(image_files)} frames)')

    # Load ground truth
    gt = load_visdrone_gt(args.annotations)
    total_gt_objects = sum(len(v) for v in gt.values())
    print(f'Ground truth: {total_gt_objects} annotations across {len(gt)} frames')

    # Define configurations to test
    configs = [
        ('Baseline (clean)',    'none',         0.0),
        ('Fog (light)',         'fog',          0.3),
        ('Fog (heavy)',         'fog',          0.7),
        ('Rain (light)',        'rain',         0.3),
        ('Rain (heavy)',        'rain',         0.7),
        ('Blur (light)',        'blur',         0.3),
        ('Blur (heavy)',        'blur',         0.7),
        ('Low light',           'low_light',    0.5),
        ('Low light (extreme)', 'low_light',    0.8),
        ('Contrast loss',       'contrast',     0.5),
        ('Adv. patch',          'patch',        0.5),
        ('Adv. stripe',         'stripe',       0.5),
        ('Checkerboard',        'checkerboard', 0.5),
        ('Occlusion',           'occlusion',    0.5),
    ]

    # Run all configs
    results = []
    print(f'\n{"="*80}')
    print(f'{"Config":<25} {"MOTA":>8} {"MOTP":>8} {"TP":>7} {"FP":>7} {"FN":>7} {"IDsw":>6} {"Time":>7}')
    print(f'{"="*80}')

    for name, attack_type, intensity in configs:
        t0 = time.time()
        summary = run_config(model, image_files, gt, attack_type, intensity)
        elapsed = time.time() - t0
        summary['name'] = name
        summary['time'] = elapsed
        results.append(summary)

        print(f'{name:<25} {summary["MOTA"]:>8.3f} {summary["MOTP"]:>8.3f} '
              f'{summary["TP"]:>7d} {summary["FP"]:>7d} {summary["FN"]:>7d} '
              f'{summary["ID_Sw"]:>6d} {elapsed:>6.1f}s')

    print(f'{"="*80}')

    # Summary
    baseline = results[0]
    print(f'\n--- Performance Drop vs Baseline (MOTA = {baseline["MOTA"]:.3f}) ---')
    for r in results[1:]:
        drop = baseline['MOTA'] - r['MOTA']
        pct = (drop / max(abs(baseline['MOTA']), 0.001)) * 100
        indicator = '!!' if pct > 30 else '! ' if pct > 15 else '  '
        print(f'  {indicator} {r["name"]:<25} MOTA drop: {drop:>+.3f} ({pct:>+.1f}%)')

    print(f'\nDone. {len(results)} configurations tested on {len(image_files)} frames.')


if __name__ == '__main__':
    main()
