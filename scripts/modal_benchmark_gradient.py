"""Run gradient-attack benchmarks on Modal (A100 GPU).

Only runs the 4 gradient attack configs (FGSM light/heavy, PGD light/heavy)
which are too slow for CPU. Non-gradient benchmarks run locally via:
    python3 scripts/benchmark.py --sequence-dir ... --model ... --skip-gradient

Separate functions for YOLOv8 and YOLO26 model families so they can run
independently or in parallel.

Upload model weights (if not already on volume from training):
    modal volume put visdrone-data ~/Documents/CODE/adversarial-tracking/tracking_ws/data/yolov8n_visdrone_best.pt /weights/yolov8n_visdrone_best.pt
    modal volume put visdrone-data ~/Documents/CODE/adversarial-tracking/tracking_ws/data/yolo26n_visdrone_best.pt /weights/yolo26n_visdrone_best.pt

Run all (yolov8 + yolo26):
    modal run --detach scripts/modal_benchmark_gradient.py

Run only yolov8:
    modal run --detach scripts/modal_benchmark_gradient.py::main --family yolov8

Run only yolo26:
    modal run --detach scripts/modal_benchmark_gradient.py::main --family yolo26

Download results:
    modal volume get visdrone-data /benchmark_gradient/ ./tracking_ws/data/benchmark_gradient/
"""

import modal

app = modal.App("visdrone-gradient-benchmark")

volume = modal.Volume.from_name("visdrone-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("ultralytics", "opencv-python-headless", "scipy", "lapx")
)


# ── Shared benchmark logic (defined inside each function to stay self-contained) ──


def _run_gradient_benchmarks(model_configs: dict[str, str], family_label: str):
    """Core benchmark runner used by both yolov8 and yolo26 functions.

    Must be called inside a Modal function (has access to GPU + volume).
    """
    import json
    import time
    from datetime import datetime
    from pathlib import Path

    import cv2
    import numpy as np
    import torch
    import torch.nn as nn
    from scipy.optimize import linear_sum_assignment
    from ultralytics import YOLO

    # ── Gradient attack implementation ────────────────────────────────────

    def numpy_to_tensor(image, device='cuda'):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(device)
        _, _, h, w = tensor.shape
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h > 0 or pad_w > 0:
            tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h))
        return tensor

    def tensor_to_numpy(tensor):
        img = tensor.squeeze(0).detach().cpu()
        img = torch.clamp(img, 0.0, 1.0)
        img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    class YOLOAttackWrapper(nn.Module):
        """Differentiable wrapper for gradient attacks.

        Works with both YOLOv8 and YOLO26. YOLO26's end-to-end head
        detaches gradients internally, so we disable end2end mode to
        fall back to the one-to-many head which preserves gradients.
        """
        def __init__(self, yolo_model):
            super().__init__()
            self.inner_model = yolo_model.model

            # YOLO26 end2end mode detaches gradients — disable it
            head = self.inner_model.model[-1]
            if hasattr(head, 'end2end'):
                head.end2end = False

        def forward(self, x):
            raw_preds = self.inner_model(x)
            preds = raw_preds[0] if isinstance(raw_preds, (list, tuple)) else raw_preds
            class_confs = preds[:, 4:, :]
            max_confs = class_confs.max(dim=1).values
            return max_confs.sum()

    def fgsm_attack(image, model, epsilon=4/255, device='cuda'):
        orig_h, orig_w = image.shape[:2]
        wrapper = YOLOAttackWrapper(model)
        wrapper.eval()
        x = numpy_to_tensor(image, device)
        x.requires_grad_(True)
        confidence = wrapper(x)
        confidence.backward()
        x_adv = x.detach() - epsilon * x.grad.sign()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        x_adv = x_adv[:, :, :orig_h, :orig_w]
        return tensor_to_numpy(x_adv)

    def pgd_attack(image, model, epsilon=4/255, alpha=1/255, steps=10, device='cuda'):
        orig_h, orig_w = image.shape[:2]
        wrapper = YOLOAttackWrapper(model)
        wrapper.eval()
        x_orig = numpy_to_tensor(image, device)
        x_adv = x_orig.clone().detach()
        for _ in range(steps):
            x_adv.requires_grad_(True)
            confidence = wrapper(x_adv)
            confidence.backward()
            x_adv = x_adv.detach() - alpha * x_adv.grad.sign()
            delta = torch.clamp(x_adv - x_orig, -epsilon, epsilon)
            x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)
        x_adv = x_adv[:, :, :orig_h, :orig_w]
        return tensor_to_numpy(x_adv)

    GRADIENT_CONFIGS = {
        'fgsm_light':  {'method': 'fgsm', 'epsilon': 4 / 255},
        'fgsm_heavy':  {'method': 'fgsm', 'epsilon': 16 / 255},
        'pgd_light':   {'method': 'pgd', 'epsilon': 4 / 255, 'alpha': 1 / 255, 'steps': 10},
        'pgd_heavy':   {'method': 'pgd', 'epsilon': 16 / 255, 'alpha': 2 / 255, 'steps': 20},
    }

    def apply_gradient_attack(image, attack_type, model, device='cuda'):
        config = {**GRADIENT_CONFIGS[attack_type]}
        method = config.pop('method')
        if method == 'fgsm':
            return fgsm_attack(image, model, device=device, **config)
        else:
            return pgd_attack(image, model, device=device, **config)

    # ── VisDrone GT loader ────────────────────────────────────────────────

    VISDRONE_TO_COCO = {
        1: 0, 2: 0, 3: 1, 4: 2, 5: 7, 6: 7, 9: 5, 10: 3,
    }

    def load_visdrone_gt(annotation_path):
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
                if category not in VISDRONE_TO_COCO:
                    continue
                if frame_id not in gt:
                    gt[frame_id] = []
                gt[frame_id].append((track_id, x, y, x + w, y + h, category))
        return gt

    # ── ByteTrack ─────────────────────────────────────────────────────────

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
            for t in self.trackers:
                t.predict()
            high = [d for d in detections if d[4] >= self.high_thresh]
            low = [d for d in detections if self.low_thresh <= d[4] < self.high_thresh]

            unmatched_trks = list(range(len(self.trackers)))
            if high and self.trackers:
                ious = iou_matrix(
                    [d[:4] for d in high],
                    [self.trackers[i].bbox for i in unmatched_trks])
                cost = 1 - ious
                rows, cols = linear_sum_assignment(cost)
                matched_det = set()
                matched_trk_indices = set()
                for r, c in zip(rows, cols):
                    if ious[r, c] >= self.match_thresh:
                        self.trackers[unmatched_trks[c]].update(
                            high[r][:4], high[r][4], high[r][5])
                        matched_det.add(r)
                        matched_trk_indices.add(unmatched_trks[c])
                unmatched_trks = [i for i in unmatched_trks if i not in matched_trk_indices]
                unmatched_high = [high[i] for i in range(len(high)) if i not in matched_det]
            else:
                unmatched_high = list(high)

            if low and unmatched_trks:
                ious = iou_matrix(
                    [d[:4] for d in low],
                    [self.trackers[i].bbox for i in unmatched_trks])
                cost = 1 - ious
                rows, cols = linear_sum_assignment(cost)
                matched_trk_indices2 = set()
                for r, c in zip(rows, cols):
                    if ious[r, c] >= self.match_thresh:
                        self.trackers[unmatched_trks[c]].update(
                            low[r][:4], low[r][4], low[r][5])
                        matched_trk_indices2.add(unmatched_trks[c])
                unmatched_trks = [i for i in unmatched_trks if i not in matched_trk_indices2]

            for d in unmatched_high:
                if d[4] >= self.high_thresh:
                    self.trackers.append(KalmanBoxTracker(d[:4], d[4], d[5]))

            self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

            results = []
            for t in self.trackers:
                if t.time_since_update == 0:
                    results.append((t.id, *t.bbox, t.score, t.cls_id))
            return results

    # ── MOT Metrics ───────────────────────────────────────────────────────

    class MOTMetrics:
        def __init__(self, iou_threshold=0.5):
            self.iou_threshold = iou_threshold
            self.total_gt = 0
            self.total_tp = 0
            self.total_fp = 0
            self.total_fn = 0
            self.total_id_switches = 0
            self.total_iou = 0.0
            self.prev_matches = {}

        def update(self, gt_boxes, track_boxes):
            self.total_gt += len(gt_boxes)
            if not gt_boxes:
                self.total_fp += len(track_boxes)
                return
            if not track_boxes:
                self.total_fn += len(gt_boxes)
                return
            gt_arr = np.array([b[1:5] for b in gt_boxes])
            tr_arr = np.array([b[1:5] for b in track_boxes])
            ious = iou_matrix(gt_arr, tr_arr)
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
                    if gt_id in self.prev_matches and self.prev_matches[gt_id] != tr_id:
                        self.total_id_switches += 1
            self.total_fn += len(gt_boxes) - len(matched_gt)
            self.total_fp += len(track_boxes) - len(matched_tr)
            self.prev_matches = current_matches

        def summary(self):
            gt = self.total_gt
            tp = self.total_tp
            return {
                'MOTA': (1.0 - (self.total_fn + self.total_fp + self.total_id_switches) / gt) if gt > 0 else 0.0,
                'MOTP': (self.total_iou / tp) if tp > 0 else 0.0,
                'TP': tp, 'FP': self.total_fp, 'FN': self.total_fn,
                'ID_Sw': self.total_id_switches, 'GT': gt,
                'total_iou': self.total_iou,
            }

    # ── Benchmark runner ──────────────────────────────────────────────────

    def run_gradient_config(model, image_files, gt, attack_type, device='cuda'):
        tracker = SimpleByteTrack()
        metrics = MOTMetrics()
        for i, img_path in enumerate(image_files):
            frame_id = i + 1
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            frame = apply_gradient_attack(frame, attack_type, model, device=device)
            results = model(frame, conf=0.25, iou=0.45, imgsz=640,
                            classes=[0, 1, 2, 3, 5, 7], verbose=False)
            detections = []
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        score = float(box.conf[0].item())
                        cls_id = int(box.cls[0].item())
                        detections.append((x1, y1, x2, y2, score, cls_id))
            tracks = tracker.update(detections)
            gt_frame = gt.get(frame_id, [])
            gt_boxes = [(b[0], b[1], b[2], b[3], b[4]) for b in gt_frame]
            tr_boxes = [(int(t[0]), t[1], t[2], t[3], t[4]) for t in tracks]
            metrics.update(gt_boxes, tr_boxes)
        return metrics.summary()

    def aggregate_results(per_seq):
        agg = {'TP': 0, 'FP': 0, 'FN': 0, 'ID_Sw': 0, 'GT': 0, 'total_iou': 0.0}
        for r in per_seq:
            for k in agg:
                agg[k] += r[k]
        gt, tp = agg['GT'], agg['TP']
        agg['MOTA'] = (1.0 - (agg['FN'] + agg['FP'] + agg['ID_Sw']) / gt) if gt > 0 else 0.0
        agg['MOTP'] = (agg['total_iou'] / tp) if tp > 0 else 0.0
        return agg

    # ── Main ──────────────────────────────────────────────────────────────

    DATA_DIR = Path("/vol/data/VisDrone2019-MOT-val")
    OUTPUT_DIR = Path("/vol/benchmark_gradient")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Discover sequences
    seq_dir = DATA_DIR / "sequences"
    ann_dir = DATA_DIR / "annotations"
    sequences = []
    total_frames = 0
    for sd in sorted(seq_dir.iterdir()):
        if not sd.is_dir():
            continue
        ann = ann_dir / (sd.name + ".txt")
        if not ann.exists():
            continue
        imgs = sorted([str(f) for f in sd.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
        gt = load_visdrone_gt(str(ann))
        sequences.append((sd.name, imgs, gt))
        total_frames += len(imgs)
    print(f"Found {len(sequences)} sequences, {total_frames} total frames")

    attack_configs = [
        ('FGSM (light)', 'fgsm_light'),
        ('FGSM (heavy)', 'fgsm_heavy'),
        ('PGD (light)',  'pgd_light'),
        ('PGD (heavy)',  'pgd_heavy'),
    ]

    device = 'cuda'

    for model_name, model_path in model_configs.items():
        print(f"\n{'='*70}")
        print(f"[{family_label}] Model: {model_name} ({model_path})")
        print(f"{'='*70}")

        # Skip if results already exist on volume
        existing = OUTPUT_DIR / f"{model_name}.json"
        if existing.exists():
            print(f"  SKIPPING — results already exist at {existing}")
            continue

        is_pretrained = model_path in ('yolov8n.pt', 'yolo26n.pt')
        if not Path(model_path).exists() and not is_pretrained:
            print(f"  SKIPPING — weights not found at {model_path}")
            continue

        if not Path(model_path).exists() and is_pretrained:
            print(f"  Using pretrained {model_path} (auto-download)")

        model = YOLO(model_path)
        model.to(device)

        results = []
        for config_name, attack_type in attack_configs:
            print(f"\n  [{config_name}]")
            t0 = time.time()
            per_seq = []
            for si, (seq_name, imgs, gt) in enumerate(sequences):
                print(f"    ({si+1}/{len(sequences)}) {seq_name} ({len(imgs)} frames)...",
                      end='', flush=True)
                seq_result = run_gradient_config(model, imgs, gt, attack_type, device=device)
                per_seq.append(seq_result)
                print(f" MOTA={seq_result['MOTA']:.3f}")

            summary = aggregate_results(per_seq)
            elapsed = time.time() - t0
            summary['name'] = config_name
            summary['attack_type'] = attack_type
            summary['time'] = elapsed
            results.append(summary)

            print(f"  {config_name}: MOTA={summary['MOTA']:.4f}  MOTP={summary['MOTP']:.4f}  "
                  f"TP={summary['TP']}  FP={summary['FP']}  FN={summary['FN']}  "
                  f"IDsw={summary['ID_Sw']}  Time={elapsed:.1f}s")

        # Save results
        out_path = OUTPUT_DIR / f"{model_name}.json"
        output = {
            'metadata': {
                'model': model_path,
                'model_name': model_name,
                'family': family_label,
                'num_sequences': len(sequences),
                'num_frames': total_frames,
                'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'device': device,
                'attack_type': 'gradient_only',
            },
            'results': results,
        }
        with open(out_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n  Saved: {out_path}")
        volume.commit()

    print(f"\n\n[{family_label}] Done!")


# ── YOLOv8 gradient benchmarks ───────────────────────────────────────────


@app.function(
    image=image,
    gpu="A100",
    volumes={"/vol": volume},
    timeout=6 * 3600,
)
def benchmark_yolov8():
    from pathlib import Path

    models = {
        'yolov8n_pretrained': 'yolov8n.pt',
        'yolov8n_finetuned': '/vol/results/train/weights/best.pt',
    }
    # Check alternate weight locations
    if not Path('/vol/results/train/weights/best.pt').exists():
        alt = Path('/vol/weights/yolov8n_visdrone_best.pt')
        if alt.exists():
            models['yolov8n_finetuned'] = str(alt)

    _run_gradient_benchmarks(models, 'yolov8')


# ── YOLO26 gradient benchmarks ───────────────────────────────────────────


@app.function(
    image=image,
    gpu="A100",
    volumes={"/vol": volume},
    timeout=6 * 3600,
)
def benchmark_yolo26():
    from pathlib import Path

    models = {
        'yolo26n_pretrained': 'yolo26n.pt',
        'yolo26n_finetuned': '/vol/results_yolo26/train/weights/best.pt',
    }
    # Check alternate weight locations
    if not Path('/vol/results_yolo26/train/weights/best.pt').exists():
        alt = Path('/vol/weights/yolo26n_visdrone_best.pt')
        if alt.exists():
            models['yolo26n_finetuned'] = str(alt)

    _run_gradient_benchmarks(models, 'yolo26')


# ── Entrypoint ────────────────────────────────────────────────────────────


@app.local_entrypoint()
def main(family: str = "all"):
    """Run gradient benchmarks on Modal.

    Args:
        family: "yolov8", "yolo26", or "all" (default).
    """
    if family in ("all", "yolov8"):
        print("Launching YOLOv8 gradient benchmarks...")
        benchmark_yolov8.remote()

    if family in ("all", "yolo26"):
        print("Launching YOLO26 gradient benchmarks...")
        benchmark_yolo26.remote()

    if family not in ("all", "yolov8", "yolo26"):
        print(f"Unknown family '{family}'. Use: all, yolov8, yolo26")

    print("\nDownload results:")
    print("  modal volume get visdrone-data /benchmark_gradient/ ./tracking_ws/data/benchmark_gradient/")
