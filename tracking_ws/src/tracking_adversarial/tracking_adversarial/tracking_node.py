"""ByteTrack tracking node for the adversarial tracking framework.

Subscribes to detections, applies the ByteTrack multi-object tracking
algorithm, and publishes tracked objects with persistent IDs.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)
from std_msgs.msg import Header


class KalmanBoxTracker:
    """Single-object Kalman filter tracker for bounding boxes.

    State vector: [cx, cy, w, h, vx, vy, vw, vh]
    Measurement: [cx, cy, w, h]
    """

    _count = 0

    def __init__(self, bbox: np.ndarray) -> None:
        """Initialize tracker with a bounding box [cx, cy, w, h]."""
        self.id = KalmanBoxTracker._count
        KalmanBoxTracker._count += 1

        # State: [cx, cy, w, h, vx, vy, vw, vh]
        self.state = np.zeros(8)
        self.state[:4] = bbox

        # Covariance matrix
        self.P = np.eye(8) * 10.0
        self.P[4:, 4:] *= 100.0  # High uncertainty for velocities

        # Process noise
        self.Q = np.eye(8) * 1.0
        self.Q[4:, 4:] *= 0.01

        # Measurement noise
        self.R = np.eye(4) * 1.0

        # State transition matrix
        self.F = np.eye(8)
        self.F[0, 4] = 1.0  # cx += vx
        self.F[1, 5] = 1.0  # cy += vy
        self.F[2, 6] = 1.0  # w += vw
        self.F[3, 7] = 1.0  # h += vh

        # Measurement matrix
        self.H = np.zeros((4, 8))
        self.H[:4, :4] = np.eye(4)

        self.hits = 1
        self.time_since_update = 0
        self.age = 0
        self.hit_streak = 1

    def predict(self) -> np.ndarray:
        """Predict next state."""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1

        # Clamp width/height to positive
        self.state[2] = max(self.state[2], 1.0)
        self.state[3] = max(self.state[3], 1.0)

        return self.state[:4]

    def update(self, bbox: np.ndarray) -> None:
        """Update state with a matched measurement."""
        z = bbox
        y = z - self.H @ self.state  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        self.state = self.state + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

        self.hits += 1
        self.time_since_update = 0
        self.hit_streak += 1

    def get_state(self) -> np.ndarray:
        """Return current bounding box estimate [cx, cy, w, h]."""
        return self.state[:4]


def _iou_matrix(
    boxes_a: np.ndarray, boxes_b: np.ndarray
) -> np.ndarray:
    """Compute IoU matrix between two sets of boxes in [cx, cy, w, h] format."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))

    # Convert to xyxy
    a_x1 = boxes_a[:, 0] - boxes_a[:, 2] / 2
    a_y1 = boxes_a[:, 1] - boxes_a[:, 3] / 2
    a_x2 = boxes_a[:, 0] + boxes_a[:, 2] / 2
    a_y2 = boxes_a[:, 1] + boxes_a[:, 3] / 2

    b_x1 = boxes_b[:, 0] - boxes_b[:, 2] / 2
    b_y1 = boxes_b[:, 1] - boxes_b[:, 3] / 2
    b_x2 = boxes_b[:, 0] + boxes_b[:, 2] / 2
    b_y2 = boxes_b[:, 1] + boxes_b[:, 3] / 2

    iou_mat = np.zeros((len(boxes_a), len(boxes_b)))

    for i in range(len(boxes_a)):
        xx1 = np.maximum(a_x1[i], b_x1)
        yy1 = np.maximum(a_y1[i], b_y1)
        xx2 = np.minimum(a_x2[i], b_x2)
        yy2 = np.minimum(a_y2[i], b_y2)

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_a = (a_x2[i] - a_x1[i]) * (a_y2[i] - a_y1[i])
        area_b = (b_x2 - b_x1) * (b_y2 - b_y1)
        union = area_a + area_b - inter

        iou_mat[i] = np.where(union > 0, inter / union, 0.0)

    return iou_mat


def _linear_assignment(cost_matrix: np.ndarray, threshold: float) -> Tuple[
    List[Tuple[int, int]], List[int], List[int]
]:
    """Greedy assignment based on cost matrix (IoU-based, higher is better)."""
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    matches = []
    unmatched_a = list(range(cost_matrix.shape[0]))
    unmatched_b = list(range(cost_matrix.shape[1]))

    # Use scipy's linear_sum_assignment if available, else greedy
    try:
        from scipy.optimize import linear_sum_assignment as lsa
        row_ind, col_ind = lsa(-cost_matrix)  # Minimize negative IoU

        matched_a = set()
        matched_b = set()
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] >= threshold:
                matches.append((r, c))
                matched_a.add(r)
                matched_b.add(c)

        unmatched_a = [i for i in range(cost_matrix.shape[0]) if i not in matched_a]
        unmatched_b = [j for j in range(cost_matrix.shape[1]) if j not in matched_b]
    except ImportError:
        # Greedy fallback
        used_b = set()
        for i in range(cost_matrix.shape[0]):
            best_j = -1
            best_val = threshold
            for j in range(cost_matrix.shape[1]):
                if j in used_b:
                    continue
                if cost_matrix[i, j] > best_val:
                    best_val = cost_matrix[i, j]
                    best_j = j
            if best_j >= 0:
                matches.append((i, best_j))
                used_b.add(best_j)
                unmatched_a.remove(i)

        unmatched_b = [j for j in range(cost_matrix.shape[1]) if j not in used_b]

    return matches, unmatched_a, unmatched_b


class ByteTracker:
    """ByteTrack-style multi-object tracker.

    Uses two-stage association:
    1. High-confidence detections matched to existing tracks
    2. Low-confidence detections matched to remaining tracks
    """

    def __init__(
        self,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        min_box_area: float = 10.0,
    ) -> None:
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.min_box_area = min_box_area

        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(
        self, detections: np.ndarray, scores: np.ndarray, class_ids: np.ndarray
    ) -> List[Dict]:
        """Update tracker with new detections.

        Args:
            detections: (N, 4) array of [cx, cy, w, h]
            scores: (N,) confidence scores
            class_ids: (N,) class IDs

        Returns:
            List of dicts with keys: track_id, bbox, score, class_id
        """
        self.frame_count += 1
        results = []

        # Predict all existing trackers
        predicted_boxes = []
        for trk in self.trackers:
            pred = trk.predict()
            predicted_boxes.append(pred)
        predicted_boxes = np.array(predicted_boxes) if predicted_boxes else np.empty((0, 4))

        # Split detections into high and low confidence
        high_mask = scores >= self.track_high_thresh
        low_mask = (scores >= self.track_low_thresh) & (~high_mask)

        high_dets = detections[high_mask]
        high_scores = scores[high_mask]
        high_classes = class_ids[high_mask]

        low_dets = detections[low_mask]
        low_scores = scores[low_mask]
        low_classes = class_ids[low_mask]

        # First association: high-confidence detections with all tracks
        if len(high_dets) > 0 and len(predicted_boxes) > 0:
            iou_mat = _iou_matrix(high_dets, predicted_boxes)
            matches_1, unmatched_dets_1, unmatched_trks_1 = _linear_assignment(
                iou_mat, self.match_thresh
            )
        else:
            matches_1 = []
            unmatched_dets_1 = list(range(len(high_dets)))
            unmatched_trks_1 = list(range(len(self.trackers)))

        # Update matched tracks
        for d_idx, t_idx in matches_1:
            self.trackers[t_idx].update(high_dets[d_idx])

        # Second association: low-confidence detections with remaining tracks
        remaining_trks = [self.trackers[i] for i in unmatched_trks_1]
        remaining_boxes = predicted_boxes[unmatched_trks_1] if len(unmatched_trks_1) > 0 else np.empty((0, 4))

        if len(low_dets) > 0 and len(remaining_boxes) > 0:
            iou_mat_2 = _iou_matrix(low_dets, remaining_boxes)
            matches_2, _, still_unmatched_trks = _linear_assignment(
                iou_mat_2, self.match_thresh * 0.7
            )
        else:
            matches_2 = []
            still_unmatched_trks = list(range(len(remaining_trks)))

        for d_idx, t_idx in matches_2:
            remaining_trks[t_idx].update(low_dets[d_idx])

        # Create new tracks from unmatched high-confidence detections
        for d_idx in unmatched_dets_1:
            if high_scores[d_idx] >= self.new_track_thresh:
                area = high_dets[d_idx][2] * high_dets[d_idx][3]
                if area >= self.min_box_area:
                    trk = KalmanBoxTracker(high_dets[d_idx])
                    self.trackers.append(trk)

        # Remove dead tracks
        self.trackers = [
            trk for trk in self.trackers
            if trk.time_since_update <= self.track_buffer
        ]

        # Collect active results
        for trk in self.trackers:
            if trk.time_since_update == 0 and trk.hit_streak >= 1:
                bbox = trk.get_state()
                results.append({
                    'track_id': trk.id,
                    'bbox': bbox,
                    'score': 1.0,  # Track confidence
                    'class_id': 0,  # Will be overridden below
                })

        # Map class IDs back (best-effort from last matched detection)
        # In practice, the class_id is carried forward from the detection
        for r in results:
            # Find closest detection by IoU to assign class
            r_box = r['bbox'].reshape(1, 4)
            all_dets = np.vstack([high_dets, low_dets]) if len(high_dets) > 0 or len(low_dets) > 0 else np.empty((0, 4))
            all_classes = np.concatenate([high_classes, low_classes]) if len(high_classes) > 0 or len(low_classes) > 0 else np.array([])
            all_scores = np.concatenate([high_scores, low_scores]) if len(high_scores) > 0 or len(low_scores) > 0 else np.array([])

            if len(all_dets) > 0:
                ious = _iou_matrix(r_box, all_dets)[0]
                best_idx = np.argmax(ious)
                if ious[best_idx] > 0.3:
                    r['class_id'] = int(all_classes[best_idx])
                    r['score'] = float(all_scores[best_idx])

        return results


class TrackingNode(Node):
    """ROS2 node wrapping the ByteTrack multi-object tracker."""

    def __init__(self) -> None:
        super().__init__('tracking_node')

        # Declare parameters
        self.declare_parameter('tracking.track_high_thresh', 0.5)
        self.declare_parameter('tracking.track_low_thresh', 0.1)
        self.declare_parameter('tracking.new_track_thresh', 0.6)
        self.declare_parameter('tracking.track_buffer', 30)
        self.declare_parameter('tracking.match_thresh', 0.8)
        self.declare_parameter('tracking.min_box_area', 10.0)

        # Initialize ByteTracker
        self.tracker = ByteTracker(
            track_high_thresh=self.get_parameter('tracking.track_high_thresh').value,
            track_low_thresh=self.get_parameter('tracking.track_low_thresh').value,
            new_track_thresh=self.get_parameter('tracking.new_track_thresh').value,
            track_buffer=self.get_parameter('tracking.track_buffer').value,
            match_thresh=self.get_parameter('tracking.match_thresh').value,
            min_box_area=self.get_parameter('tracking.min_box_area').value,
        )

        # Subscribe to detections
        self.det_sub = self.create_subscription(
            Detection2DArray, '/detections', self._detection_callback, 10
        )

        # Publish tracked objects
        self.track_pub = self.create_publisher(
            Detection2DArray, '/tracks', 10
        )

        self.frame_count = 0
        self.get_logger().info('Tracking node initialized (ByteTrack)')

    def _detection_callback(self, msg: Detection2DArray) -> None:
        """Process incoming detections through ByteTrack."""
        self.frame_count += 1

        # Convert detections to numpy arrays
        bboxes = []
        scores = []
        class_ids = []

        for det in msg.detections:
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            w = det.bbox.size_x
            h = det.bbox.size_y
            bboxes.append([cx, cy, w, h])

            if det.results:
                scores.append(det.results[0].hypothesis.score)
                class_ids.append(int(det.results[0].hypothesis.class_id))
            else:
                scores.append(0.5)
                class_ids.append(0)

        det_array = np.array(bboxes) if bboxes else np.empty((0, 4))
        score_array = np.array(scores)
        class_array = np.array(class_ids)

        # Run tracker
        tracks = self.tracker.update(det_array, score_array, class_array)

        # Build output message
        track_msg = Detection2DArray()
        track_msg.header = msg.header

        for trk in tracks:
            det = Detection2D()
            det.header = msg.header
            det.id = str(trk['track_id'])

            bbox = trk['bbox']
            det.bbox.center.position.x = float(bbox[0])
            det.bbox.center.position.y = float(bbox[1])
            det.bbox.size_x = float(bbox[2])
            det.bbox.size_y = float(bbox[3])

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(trk['class_id'])
            hyp.hypothesis.score = float(trk['score'])
            det.results.append(hyp)

            track_msg.detections.append(det)

        self.track_pub.publish(track_msg)

        if self.frame_count % 100 == 0:
            self.get_logger().info(
                f'Frame {self.frame_count}: {len(tracks)} active tracks'
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrackingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
