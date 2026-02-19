"""Multi-object tracking metrics computation.

Implements MOTA, MOTP, IDF1, ID switches, and track fragmentation.
Works by matching predicted tracks to ground truth using IoU, then
accumulating frame-by-frame error counts.

Terminology:
    - GT: ground truth object (annotated bounding box with ID)
    - Pred: predicted track (from the tracker)
    - TP: true positive (correct match between pred and GT)
    - FP: false positive (pred with no matching GT)
    - FN: false negative (GT with no matching pred)
    - IDSW: identity switch (GT matched to a different pred ID than last frame)
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class FrameData:
    """Detections/tracks for a single frame."""
    # Each entry: (object_id, x1, y1, x2, y2)
    objects: List[Tuple[int, float, float, float, float]] = field(default_factory=list)


def _compute_iou(box_a: Tuple, box_b: Tuple) -> float:
    """Compute IoU between two boxes in (x1, y1, x2, y2) format."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def _iou_matrix(gt_boxes: List[Tuple], pred_boxes: List[Tuple]) -> np.ndarray:
    """Build IoU matrix between ground truth and predicted boxes."""
    matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, gt in enumerate(gt_boxes):
        for j, pred in enumerate(pred_boxes):
            matrix[i, j] = _compute_iou(gt, pred)
    return matrix


class MOTAccumulator:
    """Accumulates frame-by-frame matching results for MOT metrics.

    Usage:
        acc = MOTAccumulator()
        for frame_idx in range(num_frames):
            acc.update(gt_ids, gt_boxes, pred_ids, pred_boxes)
        results = acc.compute()
    """

    def __init__(self, iou_threshold: float = 0.5) -> None:
        self.iou_threshold = iou_threshold

        # Per-frame counters
        self.num_frames = 0
        self.total_gt = 0          # Total GT objects across all frames
        self.total_tp = 0          # True positives
        self.total_fp = 0          # False positives
        self.total_fn = 0          # False negatives
        self.total_idsw = 0        # ID switches
        self.total_frag = 0        # Fragmentations
        self.total_iou = 0.0       # Sum of IoU for matched pairs (for MOTP)
        self.total_matches = 0     # Count of matched pairs (for MOTP)

        # For IDF1: track ID-level true positives
        self.gt_id_frames: Dict[int, int] = defaultdict(int)    # GT ID → frame count
        self.pred_id_frames: Dict[int, int] = defaultdict(int)  # Pred ID → frame count
        self.id_tp_frames: Dict[Tuple[int, int], int] = defaultdict(int)  # (GT, Pred) → matched frames

        # For ID switch detection: last frame's GT→Pred mapping
        self.last_match: Dict[int, int] = {}

        # For fragmentation: track whether each GT was matched last frame
        self.last_matched_gts: set = set()

    def update(
        self,
        gt_ids: List[int],
        gt_boxes: List[Tuple[float, float, float, float]],
        pred_ids: List[int],
        pred_boxes: List[Tuple[float, float, float, float]],
    ) -> Dict[str, float]:
        """Process one frame of ground truth and predictions.

        Args:
            gt_ids: Ground truth object IDs.
            gt_boxes: Ground truth boxes as (x1, y1, x2, y2).
            pred_ids: Predicted track IDs.
            pred_boxes: Predicted boxes as (x1, y1, x2, y2).

        Returns:
            Dict with per-frame metrics: tp, fp, fn, idsw.
        """
        self.num_frames += 1
        num_gt = len(gt_ids)
        num_pred = len(pred_ids)
        self.total_gt += num_gt

        # Count GT ID appearances (for IDF1)
        for gid in gt_ids:
            self.gt_id_frames[gid] += 1
        for pid in pred_ids:
            self.pred_id_frames[pid] += 1

        if num_gt == 0 and num_pred == 0:
            return {'tp': 0, 'fp': 0, 'fn': 0, 'idsw': 0}

        if num_gt == 0:
            self.total_fp += num_pred
            return {'tp': 0, 'fp': num_pred, 'fn': 0, 'idsw': 0}

        if num_pred == 0:
            fn = num_gt
            self.total_fn += fn
            # Fragmentation: GTs that were matched last frame are now lost
            for gid in gt_ids:
                if gid in self.last_matched_gts:
                    self.total_frag += 1
            self.last_matched_gts.clear()
            self.last_match.clear()
            return {'tp': 0, 'fp': 0, 'fn': fn, 'idsw': 0}

        # Compute IoU matrix and do greedy matching
        iou_mat = _iou_matrix(gt_boxes, pred_boxes)

        # Hungarian-style matching via scipy if available, else greedy
        matched_gt = set()
        matched_pred = set()
        matches = []  # (gt_idx, pred_idx)

        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(-iou_mat)
            for r, c in zip(row_ind, col_ind):
                if iou_mat[r, c] >= self.iou_threshold:
                    matches.append((r, c))
                    matched_gt.add(r)
                    matched_pred.add(c)
        except ImportError:
            # Greedy fallback
            flat_indices = np.argsort(-iou_mat.ravel())
            for idx in flat_indices:
                r, c = divmod(idx, num_pred)
                if r in matched_gt or c in matched_pred:
                    continue
                if iou_mat[r, c] < self.iou_threshold:
                    break
                matches.append((r, c))
                matched_gt.add(r)
                matched_pred.add(c)

        tp = len(matches)
        fp = num_pred - tp
        fn = num_gt - tp

        self.total_tp += tp
        self.total_fp += fp
        self.total_fn += fn

        # Accumulate IoU for MOTP
        for gi, pi in matches:
            self.total_iou += iou_mat[gi, pi]
            self.total_matches += 1

        # Check for ID switches and count IDF1 TPs
        idsw = 0
        current_match = {}
        currently_matched_gts = set()

        for gi, pi in matches:
            gid = gt_ids[gi]
            pid = pred_ids[pi]
            current_match[gid] = pid
            currently_matched_gts.add(gid)

            # IDF1 accumulation
            self.id_tp_frames[(gid, pid)] += 1

            # ID switch: same GT matched to a different pred than last frame
            if gid in self.last_match and self.last_match[gid] != pid:
                idsw += 1

        self.total_idsw += idsw

        # Fragmentation: GT was matched last frame but not this frame
        for gid in gt_ids:
            if gid in self.last_matched_gts and gid not in currently_matched_gts:
                self.total_frag += 1

        self.last_match = current_match
        self.last_matched_gts = currently_matched_gts

        return {'tp': tp, 'fp': fp, 'fn': fn, 'idsw': idsw}

    def compute(self) -> Dict[str, float]:
        """Compute final aggregated metrics.

        Returns:
            Dict with keys: mota, motp, idf1, num_switches, num_fragmentations,
            precision, recall, num_frames, total_gt, total_fp, total_fn.
        """
        # MOTA = 1 - (FN + FP + IDSW) / total_GT
        if self.total_gt > 0:
            mota = 1.0 - (self.total_fn + self.total_fp + self.total_idsw) / self.total_gt
        else:
            mota = 0.0

        # MOTP = average IoU of matched pairs
        if self.total_matches > 0:
            motp = self.total_iou / self.total_matches
        else:
            motp = 0.0

        # Precision and recall
        if self.total_tp + self.total_fp > 0:
            precision = self.total_tp / (self.total_tp + self.total_fp)
        else:
            precision = 0.0

        if self.total_tp + self.total_fn > 0:
            recall = self.total_tp / (self.total_tp + self.total_fn)
        else:
            recall = 0.0

        # IDF1: find best GT↔Pred ID mapping, then compute
        idf1 = self._compute_idf1()

        return {
            'mota': mota,
            'motp': motp,
            'idf1': idf1,
            'precision': precision,
            'recall': recall,
            'num_switches': self.total_idsw,
            'num_fragmentations': self.total_frag,
            'num_frames': self.num_frames,
            'total_gt': self.total_gt,
            'total_tp': self.total_tp,
            'total_fp': self.total_fp,
            'total_fn': self.total_fn,
        }

    def _compute_idf1(self) -> float:
        """Compute IDF1 score using the best global ID assignment."""
        if not self.id_tp_frames:
            return 0.0

        # Find best GT→Pred mapping by most co-occurring frames
        gt_ids = list(self.gt_id_frames.keys())
        pred_ids = list(self.pred_id_frames.keys())

        if not gt_ids or not pred_ids:
            return 0.0

        # Build cost matrix for assignment
        cost = np.zeros((len(gt_ids), len(pred_ids)))
        for i, gid in enumerate(gt_ids):
            for j, pid in enumerate(pred_ids):
                cost[i, j] = self.id_tp_frames.get((gid, pid), 0)

        # Optimal assignment
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(-cost)
        except ImportError:
            # Greedy fallback
            row_ind, col_ind = [], []
            used_j = set()
            for i in range(len(gt_ids)):
                best_j = -1
                best_val = 0
                for j in range(len(pred_ids)):
                    if j in used_j:
                        continue
                    if cost[i, j] > best_val:
                        best_val = cost[i, j]
                        best_j = j
                if best_j >= 0:
                    row_ind.append(i)
                    col_ind.append(best_j)
                    used_j.add(best_j)

        # Sum IDTP for matched pairs
        idtp = sum(cost[r, c] for r, c in zip(row_ind, col_ind))

        # Total GT frames and pred frames
        total_gt_frames = sum(self.gt_id_frames.values())
        total_pred_frames = sum(self.pred_id_frames.values())

        # IDFP = pred frames not matched, IDFN = gt frames not matched
        idfn = total_gt_frames - idtp
        idfp = total_pred_frames - idtp

        denom = 2 * idtp + idfp + idfn
        if denom > 0:
            return 2 * idtp / denom
        return 0.0
