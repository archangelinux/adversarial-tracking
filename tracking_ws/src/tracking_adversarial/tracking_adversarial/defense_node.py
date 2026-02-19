"""Defense node for the adversarial tracking framework.

Subscribes to tracks from the tracker, applies defense mechanisms
(temporal consistency, anomaly detection), and publishes corrected
tracks. Also attempts to recover recently lost tracks using motion
prediction.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)

from tracking_adversarial.utils.defense_utils import (
    TemporalConsistencyChecker,
    AnomalyDetector,
)


class DefenseNode(Node):
    """ROS2 node that applies defense mechanisms to tracked objects."""

    def __init__(self) -> None:
        super().__init__('defense_node')

        # Declare parameters
        self.declare_parameter('defense.enabled', True)
        self.declare_parameter('defense.temporal_consistency', True)
        self.declare_parameter('defense.anomaly_detection', True)
        self.declare_parameter('defense.max_position_jump', 100.0)
        self.declare_parameter('defense.history_length', 30)
        self.declare_parameter('defense.recovery_frames', 10)
        self.declare_parameter('defense.velocity_threshold', 3.0)
        self.declare_parameter('defense.acceleration_threshold', 3.0)

        # Load parameters
        self.enabled = self.get_parameter('defense.enabled').value
        use_temporal = self.get_parameter('defense.temporal_consistency').value
        use_anomaly = self.get_parameter('defense.anomaly_detection').value

        # Initialize defense mechanisms
        self.temporal_checker = None
        if use_temporal:
            self.temporal_checker = TemporalConsistencyChecker(
                max_position_jump=self.get_parameter('defense.max_position_jump').value,
                history_length=self.get_parameter('defense.history_length').value,
                recovery_frames=self.get_parameter('defense.recovery_frames').value,
            )
            self.get_logger().info('Temporal consistency checker: ON')

        self.anomaly_detector = None
        if use_anomaly:
            self.anomaly_detector = AnomalyDetector(
                velocity_threshold=self.get_parameter('defense.velocity_threshold').value,
                acceleration_threshold=self.get_parameter('defense.acceleration_threshold').value,
            )
            self.get_logger().info('Anomaly detector: ON')

        # Subscribe to raw tracks from the tracker
        self.track_sub = self.create_subscription(
            Detection2DArray, '/tracks', self._track_callback, 10
        )

        # Publish defended (corrected) tracks
        self.defended_pub = self.create_publisher(
            Detection2DArray, '/tracks/defended', 10
        )

        self.frame_count = 0
        self.corrections_total = 0
        self.anomalies_total = 0

        self.get_logger().info('Defense node initialized')

    def _track_callback(self, msg: Detection2DArray) -> None:
        """Apply defenses to incoming tracks and publish corrected results."""
        if not self.enabled:
            # Passthrough
            self.defended_pub.publish(msg)
            return

        self.frame_count += 1
        defended_msg = Detection2DArray()
        defended_msg.header = msg.header

        active_ids = set()
        corrections_this_frame = 0
        anomalies_this_frame = 0

        for det in msg.detections:
            track_id = int(det.id)
            active_ids.add(track_id)

            bbox = np.array([
                det.bbox.center.position.x,
                det.bbox.center.position.y,
                det.bbox.size_x,
                det.bbox.size_y,
            ])

            # --- Temporal consistency check ---
            was_corrected = False
            if self.temporal_checker is not None:
                bbox, was_corrected = self.temporal_checker.check_and_correct(
                    track_id, bbox
                )
                if was_corrected:
                    corrections_this_frame += 1

            # --- Anomaly detection ---
            is_anomalous = False
            anomaly_score = 0.0
            if self.anomaly_detector is not None:
                is_anomalous, anomaly_score = self.anomaly_detector.update(
                    track_id, bbox[:2]
                )
                if is_anomalous:
                    anomalies_this_frame += 1

            # Build corrected detection
            corrected = Detection2D()
            corrected.header = msg.header
            corrected.id = det.id
            corrected.bbox.center.position.x = float(bbox[0])
            corrected.bbox.center.position.y = float(bbox[1])
            corrected.bbox.size_x = float(bbox[2])
            corrected.bbox.size_y = float(bbox[3])

            # Carry forward class hypothesis, reduce score if anomalous
            if det.results:
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = det.results[0].hypothesis.class_id
                score = det.results[0].hypothesis.score
                if is_anomalous:
                    score *= 0.5  # Penalize anomalous tracks
                hyp.hypothesis.score = score
                corrected.results.append(hyp)

            defended_msg.detections.append(corrected)

        # --- Recover lost tracks ---
        if self.temporal_checker is not None:
            recovered = self.temporal_checker.recover_lost_tracks(active_ids)
            for track_id, recovered_bbox in recovered:
                rec_det = Detection2D()
                rec_det.header = msg.header
                rec_det.id = str(track_id)
                rec_det.bbox.center.position.x = float(recovered_bbox[0])
                rec_det.bbox.center.position.y = float(recovered_bbox[1])
                rec_det.bbox.size_x = float(recovered_bbox[2])
                rec_det.bbox.size_y = float(recovered_bbox[3])

                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = '0'
                hyp.hypothesis.score = 0.3  # Low confidence for recovered tracks
                rec_det.results.append(hyp)

                defended_msg.detections.append(rec_det)

            # Cleanup stale tracks
            self.temporal_checker.cleanup_stale(active_ids)

        if self.anomaly_detector is not None:
            self.anomaly_detector.cleanup(active_ids)

        self.defended_pub.publish(defended_msg)

        self.corrections_total += corrections_this_frame
        self.anomalies_total += anomalies_this_frame

        if self.frame_count % 100 == 0:
            self.get_logger().info(
                f'Frame {self.frame_count}: '
                f'{len(defended_msg.detections)} defended tracks '
                f'({self.corrections_total} corrections, '
                f'{self.anomalies_total} anomalies total)'
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DefenseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
