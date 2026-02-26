"""Standalone benchmark: YOLOv8 + ByteTrack under adversarial attacks.

Runs detection and tracking across VisDrone sequences with all attack
configurations, computes MOT metrics against ground truth, and outputs
results as JSON, HTML report, and terminal table.

Supports single-sequence or multi-sequence (full split) evaluation with
correct cross-sequence metric aggregation.

Usage:
    python3 benchmark.py --sequence <path> --annotations <path> --model yolov8n.pt
    python3 benchmark.py --sequence-dir <VisDrone-split> --model yolov8n.pt
"""

import argparse
import base64
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

# Add the package source so we can import attack utilities directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tracking_ws' / 'src' / 'tracking_adversarial'))
from tracking_adversarial.utils.degradation import apply_degradation
from tracking_adversarial.utils.adversarial import apply_adversarial_attack
from tracking_adversarial.utils.gradient_attacks import apply_gradient_attack

# Gradient-based attacks need the model object (they compute gradients)
GRADIENT_ATTACKS = {'fgsm_light', 'fgsm_heavy', 'pgd_light', 'pgd_heavy'}


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


# Minimal self-contained ByteTrack implementation

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


# VisDrone ground truth loader

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


# MOT metrics

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
            'total_iou': self.total_iou,
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


# Single-sequence benchmark runner

def run_config(model, image_files, gt, attack_type, intensity, conf_thresh=0.25,
               device='cpu'):
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
            elif attack_type in GRADIENT_ATTACKS:
                frame = apply_gradient_attack(
                    frame, attack_type, model, device=device)

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


# Sample image capture

def capture_sample_image(image_path, attack_type, intensity, gt_frame_boxes=None,
                         model=None, device='cpu'):
    """Apply an attack to a single frame and return JPEG bytes for the report."""
    frame = cv2.imread(image_path)
    if frame is None:
        return None

    if attack_type != 'none':
        if attack_type in ('rain', 'fog', 'blur', 'low_light', 'lens_dirt', 'contrast'):
            frame = apply_degradation(frame, attack_type, intensity)
        elif attack_type in ('patch', 'stripe', 'checkerboard', 'occlusion'):
            bboxes = [(b[1], b[2], b[3], b[4]) for b in gt_frame_boxes] if gt_frame_boxes else None
            frame = apply_adversarial_attack(frame, attack_type, bboxes=bboxes, intensity=intensity)
        elif attack_type in GRADIENT_ATTACKS and model is not None:
            frame = apply_gradient_attack(frame, attack_type, model, device=device)

    # Resize for the report (keep it reasonable)
    h, w = frame.shape[:2]
    scale = min(480 / h, 640 / w)
    if scale < 1:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode('ascii')


# HTML report

def generate_html_report(results, samples, metadata, output_path):
    """Generate a self-contained HTML report with charts and sample images."""

    baseline = results[0]
    baseline_mota = baseline['MOTA']
    max_mota = max(max(r['MOTA'] for r in results), 0.001)
    worst = min(results, key=lambda r: r['MOTA'])
    total_idsw = sum(r['ID_Sw'] for r in results)

    # Pre-build dynamic HTML sections
    # -- Metric card classes
    mota_class = 'good' if baseline_mota > 0.5 else 'warn' if baseline_mota > 0.2 else 'bad'
    motp_class = 'good' if baseline['MOTP'] > 0.5 else 'warn' if baseline['MOTP'] > 0.3 else 'bad'

    # -- MOTA bar chart rows
    mota_bars = ''
    for r in results:
        is_baseline = r['name'] == 'Baseline (clean)'
        bar_class = 'zero' if is_baseline else ('positive' if r['MOTA'] >= baseline_mota * 0.7 else 'negative')
        width = max(2, (max(0, r['MOTA']) / max_mota) * 100)
        mota_bars += (
            '<div class="bar-row">'
            '<div class="bar-label">{name}</div>'
            '<div class="bar-track">'
            '<div class="bar-fill {bar_class}" style="width: {width:.1f}%;">{mota:.3f}</div>'
            '</div></div>\n'
        ).format(name=r['name'], bar_class=bar_class, width=width, mota=r['MOTA'])

    # -- Drop chart rows
    drop_bars = ''
    for r in results[1:]:
        drop = baseline_mota - r['MOTA']
        pct = (drop / max(abs(baseline_mota), 0.001)) * 100
        bar_class = 'negative' if pct > 15 else 'positive'
        width = min(100, max(2, abs(pct)))
        drop_bars += (
            '<div class="bar-row">'
            '<div class="bar-label">{name}</div>'
            '<div class="bar-track">'
            '<div class="bar-fill {bar_class}" style="width: {width:.1f}%;">{pct:+.1f}%</div>'
            '</div></div>\n'
        ).format(name=r['name'], bar_class=bar_class, width=width, pct=pct)

    # -- Sample images grid
    sample_html = ''
    for name, b64 in samples:
        sample_html += (
            '<div class="sample-card">'
            '<img src="data:image/jpeg;base64,{b64}" alt="{name}" />'
            '<div class="sample-label">{name}</div>'
            '</div>\n'
        ).format(b64=b64, name=name)

    # -- Results table rows
    table_rows = ''
    for r in results:
        is_baseline = r['name'] == 'Baseline (clean)'
        drop = baseline_mota - r['MOTA']
        pct = (drop / max(abs(baseline_mota), 0.001)) * 100 if not is_baseline else 0
        severity = 'baseline' if is_baseline else ('severe' if pct > 30 else 'moderate' if pct > 15 else 'mild')
        drop_text = '—' if is_baseline else '{:+.1f}%'.format(pct)
        table_rows += (
            '<tr class="{severity}">'
            '<td>{name}</td>'
            '<td>{mota:.4f}</td>'
            '<td>{motp:.4f}</td>'
            '<td>{tp}</td>'
            '<td>{fp}</td>'
            '<td>{fn}</td>'
            '<td>{idsw}</td>'
            '<td>{time:.1f}s</td>'
            '<td>{drop}</td>'
            '</tr>\n'
        ).format(
            severity=severity, name=r['name'], mota=r['MOTA'], motp=r['MOTP'],
            tp=r['TP'], fp=r['FP'], fn=r['FN'], idsw=r['ID_Sw'],
            time=r['time'], drop=drop_text,
        )

    # Build HTML by concatenation — avoids all template/brace conflicts
    css = '''<style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { background: #0d1117; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; padding: 40px 20px; max-width: 1200px; margin: 0 auto; }
    h1 { color: #58a6ff; font-size: 28px; margin-bottom: 8px; }
    h2 { color: #58a6ff; font-size: 20px; margin: 40px 0 16px; border-bottom: 1px solid #21262d; padding-bottom: 8px; }
    .subtitle { color: #8b949e; margin-bottom: 30px; }
    .meta { color: #8b949e; font-size: 13px; margin-bottom: 20px; }
    .meta span { margin-right: 20px; }
    .metrics-row { display: flex; gap: 16px; margin-bottom: 30px; flex-wrap: wrap; }
    .metric-card { background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 20px; flex: 1; min-width: 180px; }
    .metric-card .value { font-size: 32px; font-weight: bold; margin-bottom: 4px; }
    .metric-card .label { color: #8b949e; font-size: 13px; }
    .metric-card.good .value { color: #3fb950; }
    .metric-card.warn .value { color: #d29922; }
    .metric-card.bad .value { color: #f85149; }
    .chart-container { background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 24px; margin-bottom: 24px; }
    .chart-title { font-size: 14px; color: #8b949e; margin-bottom: 16px; }
    .bar-row { display: flex; align-items: center; margin-bottom: 6px; }
    .bar-label { width: 180px; font-size: 13px; color: #c9d1d9; text-align: right; padding-right: 12px; flex-shrink: 0; }
    .bar-track { flex: 1; height: 24px; background: #21262d; border-radius: 4px; position: relative; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 4px; display: flex; align-items: center; padding-left: 8px; font-size: 11px; font-weight: bold; }
    .bar-fill.positive { background: linear-gradient(90deg, #1f6feb, #58a6ff); color: #fff; }
    .bar-fill.negative { background: linear-gradient(90deg, #b62324, #f85149); color: #fff; }
    .bar-fill.zero { background: #3fb950; color: #fff; }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    th { background: #161b22; color: #8b949e; padding: 10px 12px; text-align: left; font-weight: 600; border-bottom: 2px solid #21262d; }
    td { padding: 10px 12px; border-bottom: 1px solid #21262d; }
    tr.baseline td { color: #3fb950; font-weight: 600; }
    tr.mild td:first-child { border-left: 3px solid #3fb950; }
    tr.moderate td:first-child { border-left: 3px solid #d29922; }
    tr.severe td:first-child { border-left: 3px solid #f85149; }
    tr.baseline td:first-child { border-left: 3px solid #58a6ff; }
    tr:hover { background: #161b22; }
    .samples-grid { display: flex; flex-wrap: wrap; gap: 12px; }
    .sample-card { background: #161b22; border: 1px solid #21262d; border-radius: 8px; overflow: hidden; width: calc(25% - 9px); min-width: 200px; }
    .sample-card img { width: 100%; display: block; }
    .sample-label { padding: 8px; font-size: 12px; color: #8b949e; text-align: center; }
    .footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid #21262d; color: #484f58; font-size: 12px; }
</style>'''

    parts = []
    parts.append('<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">')
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1.0">')
    parts.append('<title>Benchmark Report</title>')
    parts.append(css)
    parts.append('</head><body>')
    parts.append('<h1>Adversarial Tracking Benchmark Report</h1>')
    parts.append('<p class="subtitle">YOLOv8 + ByteTrack performance under %d attack configurations</p>' % len(results))
    parts.append('<div class="meta">')
    parts.append('<span>Model: <strong>%s</strong></span>' % metadata['model'])
    parts.append('<span>Sequence: <strong>%s</strong></span>' % metadata['sequence'])
    parts.append('<span>Frames: <strong>%s</strong></span>' % metadata['num_frames'])
    parts.append('<span>Date: <strong>%s</strong></span>' % metadata['date'])
    parts.append('</div>')

    parts.append('<div class="metrics-row">')
    parts.append('<div class="metric-card %s"><div class="value">%.3f</div><div class="label">Baseline MOTA</div></div>' % (mota_class, baseline_mota))
    parts.append('<div class="metric-card %s"><div class="value">%.3f</div><div class="label">Baseline MOTP</div></div>' % (motp_class, baseline['MOTP']))
    parts.append('<div class="metric-card warn"><div class="value">%.3f</div><div class="label">Worst MOTA (%s)</div></div>' % (worst['MOTA'], worst['name']))
    parts.append('<div class="metric-card"><div class="value">%d</div><div class="label">Total ID Switches (all configs)</div></div>' % total_idsw)
    parts.append('</div>')

    parts.append('<h2>MOTA by Attack Configuration</h2>')
    parts.append('<div class="chart-container"><div class="chart-title">Higher is better. Green = baseline, blue = mild impact, red = severe impact.</div>')
    parts.append(mota_bars)
    parts.append('</div>')

    parts.append('<h2>Performance Drop vs Baseline</h2>')
    parts.append('<div class="chart-container"><div class="chart-title">How much each attack degrades MOTA relative to clean baseline</div>')
    parts.append(drop_bars)
    parts.append('</div>')

    parts.append('<h2>Attack Samples</h2>')
    parts.append('<div class="samples-grid">' + sample_html + '</div>')

    parts.append('<h2>Full Results</h2>')
    parts.append('<table><thead><tr><th>Configuration</th><th>MOTA</th><th>MOTP</th><th>TP</th><th>FP</th><th>FN</th><th>ID Sw</th><th>Time</th><th>MOTA Drop</th></tr></thead>')
    parts.append('<tbody>' + table_rows + '</tbody></table>')

    parts.append('<div class="footer">Generated by adversarial-tracking benchmark &middot; %s &middot; %s frames &middot; %s</div>' % (metadata['date'], metadata['num_frames'], metadata['model']))
    parts.append('</body></html>')

    html = '\n'.join(parts)

    with open(output_path, 'w') as f:
        f.write(html)


# Multi-sequence support

def load_sequence(seq_path, max_frames=0):
    """Load image file paths from a sequence directory.
    Returns sorted list of image file paths.
    """
    extensions = {'.jpg', '.jpeg', '.png'}
    image_files = sorted([
        str(f) for f in Path(seq_path).iterdir()
        if f.suffix.lower() in extensions
    ])
    if max_frames > 0:
        image_files = image_files[:max_frames]
    return image_files


def discover_sequences(split_dir):
    """Discover all sequences in a VisDrone split directory.

    Expects layout:
        split_dir/sequences/<seq_name>/  (images)
        split_dir/annotations/<seq_name>.txt

    Returns list of (seq_name, image_dir, annotation_path) sorted by name.
    """
    split_dir = Path(split_dir)
    sequences_dir = split_dir / 'sequences'
    annotations_dir = split_dir / 'annotations'

    if not sequences_dir.is_dir():
        raise FileNotFoundError(f'No sequences/ directory found in {split_dir}')
    if not annotations_dir.is_dir():
        raise FileNotFoundError(f'No annotations/ directory found in {split_dir}')

    sequences = []
    for seq_dir in sorted(sequences_dir.iterdir()):
        if not seq_dir.is_dir():
            continue
        ann_file = annotations_dir / (seq_dir.name + '.txt')
        if not ann_file.exists():
            print(f'  Warning: no annotation file for {seq_dir.name}, skipping')
            continue
        sequences.append((seq_dir.name, str(seq_dir), str(ann_file)))

    if not sequences:
        raise FileNotFoundError(f'No sequences found in {sequences_dir}')

    return sequences


def aggregate_results(per_seq_results):
    """Aggregate raw MOT counts across sequences and compute MOTA/MOTP.

    per_seq_results: list of summary dicts from run_config()
    Returns a single summary dict with aggregated metrics.
    """
    agg = {'TP': 0, 'FP': 0, 'FN': 0, 'ID_Sw': 0, 'GT': 0, 'total_iou': 0.0}
    for r in per_seq_results:
        agg['TP'] += r['TP']
        agg['FP'] += r['FP']
        agg['FN'] += r['FN']
        agg['ID_Sw'] += r['ID_Sw']
        agg['GT'] += r['GT']
        agg['total_iou'] += r['total_iou']

    gt = agg['GT']
    tp = agg['TP']
    agg['MOTA'] = (1.0 - (agg['FN'] + agg['FP'] + agg['ID_Sw']) / gt) if gt > 0 else 0.0
    agg['MOTP'] = (agg['total_iou'] / tp) if tp > 0 else 0.0
    return agg


def main():
    parser = argparse.ArgumentParser(description='Benchmark tracking under attacks')
    parser.add_argument('--sequence', default='', help='Path to a single image sequence folder')
    parser.add_argument('--annotations', default='', help='Path to VisDrone annotation file (required with --sequence)')
    parser.add_argument('--sequence-dir', default='', help='Path to VisDrone split dir (e.g. VisDrone2019-MOT-val/) — runs all sequences')
    parser.add_argument('--model', default='yolov8n.pt', help='YOLOv8 model path')
    parser.add_argument('--device', default='cpu', help='Inference device')
    parser.add_argument('--max-frames', type=int, default=0, help='Limit frames per sequence (0 = all)')
    parser.add_argument('--output', default='', help='Output directory (default: auto)')
    parser.add_argument('--baseline-only', action='store_true', help='Only run clean baseline (no attacks)')
    parser.add_argument('--skip-gradient', action='store_true', help='Skip gradient attacks (FGSM/PGD) — run these on GPU via Modal instead')
    parser.add_argument('--serve', action='store_true', help='Serve the report on http://localhost:8080 after generating')
    args = parser.parse_args()

    # Validate arguments: need either --sequence + --annotations or --sequence-dir
    if args.sequence_dir and args.sequence:
        parser.error('--sequence-dir and --sequence are mutually exclusive')
    if not args.sequence_dir and not args.sequence:
        parser.error('Provide either --sequence + --annotations or --sequence-dir')
    if args.sequence and not args.annotations:
        parser.error('--annotations is required when using --sequence')

    multi_sequence = bool(args.sequence_dir)

    # Build list of (name, image_dir, annotation_path)
    if multi_sequence:
        seq_list = discover_sequences(args.sequence_dir)
        print(f'Discovered {len(seq_list)} sequences in {args.sequence_dir}')
        for name, _, _ in seq_list:
            print(f'  - {name}')
    else:
        seq_name = Path(args.sequence).name
        seq_list = [(seq_name, args.sequence, args.annotations)]

    # Output directory
    if args.output:
        output_dir = Path(args.output)
    elif multi_sequence:
        output_dir = Path(args.sequence_dir) / 'results'
    else:
        output_dir = Path(args.sequence).parent.parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f'Loading model: {args.model}')
    model = YOLO(args.model)
    model.to(args.device)

    # Pre-load all sequences (images + ground truth)
    sequences = []
    total_frames = 0
    for seq_name, img_dir, ann_path in seq_list:
        image_files = load_sequence(img_dir, args.max_frames)
        gt = load_visdrone_gt(ann_path)
        sequences.append((seq_name, image_files, gt))
        total_frames += len(image_files)
        gt_count = sum(len(v) for v in gt.values())
        print(f'  {seq_name}: {len(image_files)} frames, {gt_count} GT annotations')

    print(f'Total: {total_frames} frames across {len(sequences)} sequence(s)')

    # Define configurations to test
    if args.baseline_only:
        configs = [
            ('Baseline (clean)',    'none',         0.0),
        ]
    else:
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
            # Gradient-based attacks (use model internals — much stronger)
            ('FGSM (light)',        'fgsm_light',   0.0),
            ('FGSM (heavy)',        'fgsm_heavy',   0.0),
            ('PGD (light)',         'pgd_light',    0.0),
            ('PGD (heavy)',         'pgd_heavy',    0.0),
        ]

    if args.skip_gradient:
        configs = [(n, a, i) for n, a, i in configs if a not in GRADIENT_ATTACKS]

    # Capture sample images from the first sequence's middle frame
    first_images = sequences[0][1]
    first_gt = sequences[0][2]
    sample_idx = min(len(first_images) // 2, len(first_images) - 1)
    sample_path = first_images[sample_idx]
    sample_frame_id = sample_idx + 1
    sample_gt = first_gt.get(sample_frame_id, [])

    print(f'\nCapturing sample images from frame {sample_frame_id} of {sequences[0][0]}...')
    samples = []
    for name, attack_type, intensity in configs:
        b64 = capture_sample_image(sample_path, attack_type, intensity, sample_gt,
                                   model=model, device=args.device)
        if b64:
            samples.append((name, b64))

    # Run all configs
    results = []
    print(f'\n{"="*80}')
    print(f'{"Config":<25} {"MOTA":>8} {"MOTP":>8} {"TP":>7} {"FP":>7} {"FN":>7} {"IDsw":>6} {"Time":>7}')
    print(f'{"="*80}')

    for config_idx, (name, attack_type, intensity) in enumerate(configs):
        t0 = time.time()

        if len(sequences) == 1:
            # Single sequence — run directly
            seq_name, image_files, gt = sequences[0]
            summary = run_config(model, image_files, gt, attack_type, intensity,
                                 device=args.device)
        else:
            # Multi-sequence — run each and aggregate
            print(f'[{config_idx+1}/{len(configs)}] {name}')
            per_seq = []
            for si, (seq_name, image_files, gt) in enumerate(sequences):
                print(f'  ({si+1}/{len(sequences)}) {seq_name}...', end='', flush=True)
                seq_result = run_config(model, image_files, gt, attack_type, intensity,
                                        device=args.device)
                per_seq.append(seq_result)
                print(f' done')
            summary = aggregate_results(per_seq)

        elapsed = time.time() - t0
        summary['name'] = name
        summary['attack_type'] = attack_type
        summary['intensity'] = intensity
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

    # Save JSON
    json_path = output_dir / 'results.json'
    seq_names = [s[0] for s in sequences]
    metadata = {
        'model': args.model,
        'sequence': seq_names[0] if len(seq_names) == 1 else seq_names,
        'num_sequences': len(seq_names),
        'num_frames': total_frames,
        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'device': args.device,
    }
    with open(json_path, 'w') as f:
        json.dump({'metadata': metadata, 'results': results}, f, indent=2)
    print(f'\nResults saved to: {json_path}')

    # Generate HTML report
    # For multi-sequence, show count in the sequence field
    report_metadata = dict(metadata)
    if len(seq_names) > 1:
        report_metadata['sequence'] = f'{len(seq_names)} sequences'
    report_path = output_dir / 'report.html'
    generate_html_report(results, samples, report_metadata, report_path)
    print(f'Report saved to: {report_path}')

    print(f'\nDone. {len(results)} configurations tested on {total_frames} frames across {len(sequences)} sequence(s).')


if __name__ == '__main__':
    main()
