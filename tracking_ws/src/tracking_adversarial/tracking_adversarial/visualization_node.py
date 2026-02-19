"""Visualization node for the adversarial tracking framework.

Publishes RViz2 markers for bounding boxes and an annotated image overlay
showing detections, tracks, IDs, and trails.
"""

import colorsys
from typing import Dict, List, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Header
from geometry_msgs.msg import Point
from cv_bridge import CvBridge


# COCO class names for display
COCO_NAMES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
    5: 'bus', 7: 'truck',
}


def _id_to_color(track_id: int) -> Tuple[int, int, int]:
    """Generate a unique, visually distinct BGR color for a track ID."""
    hue = (track_id * 0.618033988749895) % 1.0  # golden ratio for spread
    r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


class VisualizationNode(Node):
    """ROS2 node that creates visual overlays and RViz2 markers."""

    def __init__(self) -> None:
        super().__init__('visualization_node')

        self.declare_parameter('visualization.show_detections', True)
        self.declare_parameter('visualization.show_tracks', True)
        self.declare_parameter('visualization.show_ids', True)
        self.declare_parameter('visualization.show_trails', True)
        self.declare_parameter('visualization.trail_length', 30)
        self.declare_parameter('visualization.bbox_thickness', 2)
        self.declare_parameter('visualization.font_scale', 0.6)
        self.declare_parameter('visualization.publish_rate', 30.0)

        # Load parameters
        self.show_detections = self.get_parameter('visualization.show_detections').value
        self.show_tracks = self.get_parameter('visualization.show_tracks').value
        self.show_ids = self.get_parameter('visualization.show_ids').value
        self.show_trails = self.get_parameter('visualization.show_trails').value
        self.trail_length = self.get_parameter('visualization.trail_length').value
        self.bbox_thickness = self.get_parameter('visualization.bbox_thickness').value
        self.font_scale = self.get_parameter('visualization.font_scale').value

        self.bridge = CvBridge()

        # Track history for drawing trails: {track_id: [(cx, cy), ...]}
        self.track_trails: Dict[int, List[Tuple[float, float]]] = {}

        # Latest data caches
        self.latest_image: np.ndarray | None = None
        self.latest_detections: Detection2DArray | None = None
        self.latest_tracks: Detection2DArray | None = None
        self.latest_header: Header | None = None

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self._image_callback, sensor_qos
        )
        self.detection_sub = self.create_subscription(
            Detection2DArray, '/detections', self._detection_callback, 10
        )
        self.track_sub = self.create_subscription(
            Detection2DArray, '/tracks', self._track_callback, 10
        )

        # Publishers
        self.vis_image_pub = self.create_publisher(
            Image, '/visualization/image', sensor_qos
        )
        self.marker_pub = self.create_publisher(
            MarkerArray, '/visualization/markers', 10
        )

        # Timer-driven publishing
        pub_rate = self.get_parameter('visualization.publish_rate').value
        self.create_timer(1.0 / pub_rate, self._publish_visualization)

        self.get_logger().info('Visualization node initialized')

    def _image_callback(self, msg: Image) -> None:
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.latest_header = msg.header
        except Exception as e:
            self.get_logger().error(f'CV Bridge error: {e}')

    def _detection_callback(self, msg: Detection2DArray) -> None:
        self.latest_detections = msg

    def _track_callback(self, msg: Detection2DArray) -> None:
        self.latest_tracks = msg
        self._update_trails(msg)

    def _update_trails(self, tracks: Detection2DArray) -> None:
        """Update trail history for each active track."""
        active_ids = set()
        for det in tracks.detections:
            if not det.results:
                continue
            # Track ID is stored in the detection id field
            track_id = int(det.id)
            active_ids.add(track_id)

            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y

            if track_id not in self.track_trails:
                self.track_trails[track_id] = []
            self.track_trails[track_id].append((cx, cy))

            # Trim to trail length
            if len(self.track_trails[track_id]) > self.trail_length:
                self.track_trails[track_id] = self.track_trails[track_id][
                    -self.trail_length:
                ]

        # Remove stale trails
        stale = [tid for tid in self.track_trails if tid not in active_ids]
        for tid in stale:
            # Keep for a few extra frames for visual fade-out
            trail = self.track_trails[tid]
            if len(trail) > 1:
                self.track_trails[tid] = trail[1:]
            else:
                del self.track_trails[tid]

    def _publish_visualization(self) -> None:
        """Publish annotated image and RViz2 markers."""
        if self.latest_image is None:
            return

        vis_frame = self.latest_image.copy()
        marker_array = MarkerArray()
        marker_id = 0

        # Draw detections (thin green boxes)
        if self.show_detections and self.latest_detections:
            for det in self.latest_detections.detections:
                cx = det.bbox.center.position.x
                cy = det.bbox.center.position.y
                w = det.bbox.size_x
                h = det.bbox.size_y
                x1, y1 = int(cx - w / 2), int(cy - h / 2)
                x2, y2 = int(cx + w / 2), int(cy + h / 2)

                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

                if det.results:
                    cls_id = det.results[0].hypothesis.class_id
                    score = det.results[0].hypothesis.score
                    label = COCO_NAMES.get(int(cls_id), cls_id)
                    cv2.putText(
                        vis_frame,
                        f'{label} {score:.2f}',
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.font_scale * 0.7,
                        (0, 255, 0),
                        1,
                    )

        # Draw tracks (thick colored boxes with IDs)
        if self.show_tracks and self.latest_tracks:
            for det in self.latest_tracks.detections:
                track_id = int(det.id)
                color = _id_to_color(track_id)

                cx = det.bbox.center.position.x
                cy = det.bbox.center.position.y
                w = det.bbox.size_x
                h = det.bbox.size_y
                x1, y1 = int(cx - w / 2), int(cy - h / 2)
                x2, y2 = int(cx + w / 2), int(cy + h / 2)

                cv2.rectangle(
                    vis_frame, (x1, y1), (x2, y2), color, self.bbox_thickness
                )

                # Track ID label
                if self.show_ids:
                    label = f'ID:{track_id}'
                    if det.results:
                        cls_id = det.results[0].hypothesis.class_id
                        cls_name = COCO_NAMES.get(int(cls_id), cls_id)
                        label = f'{cls_name} ID:{track_id}'

                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 2
                    )
                    cv2.rectangle(
                        vis_frame,
                        (x1, y1 - th - 8),
                        (x1 + tw + 4, y1),
                        color,
                        -1,
                    )
                    cv2.putText(
                        vis_frame,
                        label,
                        (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.font_scale,
                        (255, 255, 255),
                        2,
                    )

                # RViz2 marker for this track
                marker = Marker()
                marker.header.frame_id = 'camera_frame'
                marker.header.stamp = (
                    self.latest_header.stamp
                    if self.latest_header
                    else self.get_clock().now().to_msg()
                )
                marker.ns = 'tracks'
                marker.id = marker_id
                marker_id += 1
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.pose.position.x = cx / 100.0  # Scale to meters
                marker.pose.position.y = -cy / 100.0
                marker.pose.position.z = 0.0
                marker.scale.x = w / 100.0
                marker.scale.y = h / 100.0
                marker.scale.z = 0.1
                r, g, b = color[2] / 255.0, color[1] / 255.0, color[0] / 255.0
                marker.color = ColorRGBA(r=r, g=g, b=b, a=0.5)
                marker.lifetime.sec = 0
                marker.lifetime.nanosec = 200_000_000
                marker_array.markers.append(marker)

        # Draw trails
        if self.show_trails:
            for track_id, trail in self.track_trails.items():
                if len(trail) < 2:
                    continue
                color = _id_to_color(track_id)
                for i in range(1, len(trail)):
                    alpha = i / len(trail)  # Fade older points
                    pt1 = (int(trail[i - 1][0]), int(trail[i - 1][1]))
                    pt2 = (int(trail[i][0]), int(trail[i][1]))
                    thickness = max(1, int(alpha * 3))
                    cv2.line(vis_frame, pt1, pt2, color, thickness)

                # Trail as line strip marker
                trail_marker = Marker()
                trail_marker.header.frame_id = 'camera_frame'
                trail_marker.header.stamp = (
                    self.latest_header.stamp
                    if self.latest_header
                    else self.get_clock().now().to_msg()
                )
                trail_marker.ns = 'trails'
                trail_marker.id = marker_id
                marker_id += 1
                trail_marker.type = Marker.LINE_STRIP
                trail_marker.action = Marker.ADD
                trail_marker.scale.x = 0.02
                r, g, b = color[2] / 255.0, color[1] / 255.0, color[0] / 255.0
                trail_marker.color = ColorRGBA(r=r, g=g, b=b, a=0.7)
                trail_marker.lifetime.sec = 0
                trail_marker.lifetime.nanosec = 200_000_000
                for cx, cy in trail:
                    pt = Point(x=cx / 100.0, y=-cy / 100.0, z=0.0)
                    trail_marker.points.append(pt)
                marker_array.markers.append(trail_marker)

        # Publish annotated image
        try:
            vis_msg = self.bridge.cv2_to_imgmsg(vis_frame, encoding='bgr8')
            if self.latest_header:
                vis_msg.header = self.latest_header
            self.vis_image_pub.publish(vis_msg)
        except Exception as e:
            self.get_logger().error(f'Failed to publish visualization image: {e}')

        # Publish markers
        if marker_array.markers:
            self.marker_pub.publish(marker_array)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisualizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
