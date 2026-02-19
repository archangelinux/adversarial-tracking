"""Evaluation node for the adversarial tracking framework.

Loads ground truth annotations (VisDrone MOT format), compares them
against tracked objects frame-by-frame, computes MOTA/MOTP/IDF1,
and publishes metrics + writes results to CSV.

VisDrone MOT annotation format (per line):
    <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<score>,<category>,<truncation>,<occlusion>
"""

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import String

from tracking_adversarial.utils.metrics import MOTAccumulator


def _load_visdrone_annotations(
    ann_path: str,
) -> Dict[int, List[Tuple[int, float, float, float, float]]]:
    """Load VisDrone MOT ground truth annotations.

    Args:
        ann_path: Path to annotation file (txt) or directory of per-frame files.

    Returns:
        Dict mapping frame_index (0-based) to list of
        (object_id, x1, y1, x2, y2) tuples.
    """
    annotations: Dict[int, List[Tuple[int, float, float, float, float]]] = {}
    path = Path(ann_path)

    if path.is_file():
        # Single annotation file — VisDrone MOT format
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) < 8:
                    continue

                frame_idx = int(parts[0]) - 1  # Convert 1-based to 0-based
                obj_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                score = float(parts[6])
                category = int(parts[7])

                # Skip ignored regions (category 0) and score 0
                if category == 0 or score == 0:
                    continue

                x1, y1, x2, y2 = x, y, x + w, y + h

                if frame_idx not in annotations:
                    annotations[frame_idx] = []
                annotations[frame_idx].append((obj_id, x1, y1, x2, y2))

    elif path.is_dir():
        # Directory of per-frame annotation files
        ann_files = sorted(path.glob('*.txt'))
        for idx, ann_file in enumerate(ann_files):
            annotations[idx] = []
            with open(ann_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(',')
                    if len(parts) < 8:
                        continue

                    obj_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])

                    x1, y1, x2, y2 = x, y, x + w, y + h
                    annotations[idx].append((obj_id, x1, y1, x2, y2))

    return annotations


class EvaluationNode(Node):
    """ROS2 node that evaluates tracking performance against ground truth."""

    def __init__(self) -> None:
        super().__init__('evaluation_node')

        # Declare parameters
        self.declare_parameter('evaluation.ground_truth_file', '')
        self.declare_parameter('evaluation.output_dir', 'data/results')
        self.declare_parameter('evaluation.save_csv', True)
        self.declare_parameter('evaluation.log_interval', 100)
        self.declare_parameter('evaluation.iou_threshold', 0.5)

        gt_path = self.get_parameter('evaluation.ground_truth_file').value
        self.output_dir = self.get_parameter('evaluation.output_dir').value
        self.save_csv = self.get_parameter('evaluation.save_csv').value
        self.log_interval = self.get_parameter('evaluation.log_interval').value
        iou_threshold = self.get_parameter('evaluation.iou_threshold').value

        # Load ground truth
        self.ground_truth: Dict[int, List[Tuple]] = {}
        if gt_path:
            self.ground_truth = _load_visdrone_annotations(gt_path)
            self.get_logger().info(
                f'Loaded ground truth: {len(self.ground_truth)} frames '
                f'from {gt_path}'
            )
        else:
            self.get_logger().warn(
                'No ground truth file specified — metrics will not be computed. '
                'Set evaluation.ground_truth_file parameter.'
            )

        # Metrics accumulator
        self.accumulator = MOTAccumulator(iou_threshold=iou_threshold)

        # Subscribe to tracks (or defended tracks)
        self.track_sub = self.create_subscription(
            Detection2DArray, '/tracks', self._track_callback, 10
        )

        # Publish metrics as JSON
        self.metrics_pub = self.create_publisher(
            String, '/evaluation/metrics', 10
        )

        # CSV output
        self.csv_file = None
        self.csv_writer = None
        if self.save_csv and self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            csv_path = os.path.join(self.output_dir, 'frame_metrics.csv')
            self.csv_file = open(csv_path, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                'frame', 'num_gt', 'num_pred', 'tp', 'fp', 'fn', 'idsw',
            ])
            self.get_logger().info(f'Writing per-frame metrics to: {csv_path}')

        self.frame_index = 0
        self.get_logger().info('Evaluation node initialized')

    def _track_callback(self, msg: Detection2DArray) -> None:
        """Compare tracks against ground truth for this frame."""
        if not self.ground_truth:
            self.frame_index += 1
            return

        # Get ground truth for this frame
        gt_data = self.ground_truth.get(self.frame_index, [])
        gt_ids = [g[0] for g in gt_data]
        gt_boxes = [(g[1], g[2], g[3], g[4]) for g in gt_data]

        # Get predictions from tracker
        pred_ids = []
        pred_boxes = []
        for det in msg.detections:
            tid = int(det.id)
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            w = det.bbox.size_x
            h = det.bbox.size_y
            x1, y1 = cx - w / 2, cy - h / 2
            x2, y2 = cx + w / 2, cy + h / 2
            pred_ids.append(tid)
            pred_boxes.append((x1, y1, x2, y2))

        # Update accumulator
        frame_result = self.accumulator.update(gt_ids, gt_boxes, pred_ids, pred_boxes)

        # Write per-frame CSV
        if self.csv_writer:
            self.csv_writer.writerow([
                self.frame_index,
                len(gt_ids),
                len(pred_ids),
                frame_result['tp'],
                frame_result['fp'],
                frame_result['fn'],
                frame_result['idsw'],
            ])

        # Periodically publish aggregate metrics
        if self.frame_index > 0 and self.frame_index % self.log_interval == 0:
            metrics = self.accumulator.compute()
            self._publish_metrics(metrics)
            self.get_logger().info(
                f'Frame {self.frame_index}: '
                f'MOTA={metrics["mota"]:.3f} '
                f'MOTP={metrics["motp"]:.3f} '
                f'IDF1={metrics["idf1"]:.3f} '
                f'IDSW={metrics["num_switches"]}'
            )

        self.frame_index += 1

    def _publish_metrics(self, metrics: Dict[str, float]) -> None:
        """Publish metrics as a JSON string message."""
        msg = String()
        # Round floats for readability
        rounded = {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in metrics.items()
        }
        msg.data = json.dumps(rounded)
        self.metrics_pub.publish(msg)

    def destroy_node(self) -> None:
        """Write final metrics and clean up."""
        if self.ground_truth:
            final_metrics = self.accumulator.compute()
            self.get_logger().info('=== FINAL METRICS ===')
            self.get_logger().info(f'  MOTA:           {final_metrics["mota"]:.4f}')
            self.get_logger().info(f'  MOTP:           {final_metrics["motp"]:.4f}')
            self.get_logger().info(f'  IDF1:           {final_metrics["idf1"]:.4f}')
            self.get_logger().info(f'  Precision:      {final_metrics["precision"]:.4f}')
            self.get_logger().info(f'  Recall:         {final_metrics["recall"]:.4f}')
            self.get_logger().info(f'  ID Switches:    {final_metrics["num_switches"]}')
            self.get_logger().info(f'  Fragmentations: {final_metrics["num_fragmentations"]}')
            self.get_logger().info(f'  Frames:         {final_metrics["num_frames"]}')

            # Save final metrics to JSON
            if self.output_dir:
                json_path = os.path.join(self.output_dir, 'final_metrics.json')
                with open(json_path, 'w') as f:
                    json.dump(final_metrics, f, indent=2)
                self.get_logger().info(f'  Saved to: {json_path}')

            # Publish final metrics
            self._publish_metrics(final_metrics)

        if self.csv_file:
            self.csv_file.close()

        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EvaluationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
