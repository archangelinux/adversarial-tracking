"""YOLOv8 detection node for the adversarial tracking framework.

Subscribes to camera images, runs YOLOv8 inference, and publishes
Detection2DArray messages with bounding boxes and class labels.

Supports three input modes:
  - Video file (MP4, AVI, etc.)
  - Image sequence folder (numbered JPGs/PNGs — e.g., VisDrone format)
  - ROS2 topic subscription (live camera or another node)
"""

from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)
from std_msgs.msg import Header
from cv_bridge import CvBridge
from ultralytics import YOLO


class DetectionNode(Node):
    """ROS2 node that runs YOLOv8 object detection on incoming images."""

    def __init__(self) -> None:
        super().__init__('detection_node')

        self.declare_parameter('video_source', '')
        self.declare_parameter('detection.model', 'yolov8n.pt')
        self.declare_parameter('detection.confidence_threshold', 0.25)
        self.declare_parameter('detection.nms_threshold', 0.45)
        self.declare_parameter('detection.device', 'cpu')
        self.declare_parameter('detection.classes', [0, 1, 2, 3, 5, 7])
        self.declare_parameter('detection.input_size', 640)

        # load params
        self.video_source = self.get_parameter('video_source').value
        model_path = self.get_parameter('detection.model').value
        self.conf_thresh = self.get_parameter('detection.confidence_threshold').value
        self.nms_thresh = self.get_parameter('detection.nms_threshold').value
        device = self.get_parameter('detection.device').value
        self.target_classes = self.get_parameter('detection.classes').value
        self.input_size = self.get_parameter('detection.input_size').value

        # initialize YOLOv8 model
        self.get_logger().info(f'Loading YOLOv8 model: {model_path}')
        self.model = YOLO(model_path)
        self.model.to(device)
        self.get_logger().info(f'Model loaded on device: {device}')

        # CV Bridge for ROS <-> OpenCV conversion
        self.bridge = CvBridge()

        # Subscribe to both raw and degraded image topics.
        # In baseline mode, only raw is used. In adversarial mode,
        # the attack node publishes degraded frames and detection
        # runs on those instead (degraded callback skips the timer-based detection).
        self.use_degraded = False
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self._image_callback, 10,
        )
        self.degraded_sub = self.create_subscription(
            Image, '/camera/degraded', self._degraded_image_callback, 10,
        )

        # publisher for detections
        self.detection_pub = self.create_publisher(Detection2DArray, '/detections', 10)

        # publisher for the raw image (passthrough so downstream nodes can use it)
        self.image_pub = self.create_publisher(Image, '/detection/image', 10)

        # playback state
        self.video_cap = None       # used for video files
        self.image_files = []       # used for image sequences
        self.image_index = 0        # current position in image sequence
        self.video_timer = None
        self.playback_fps = 30.0

        if self.video_source:
            self._start_playback()

        self.frame_count = 0
        self.get_logger().info('Detection node initialized')

    def _start_playback(self) -> None:
        """Start playback from either a video file or image sequence folder."""
        source = Path(self.video_source)

        # raw image publisher for playback
        self.raw_image_pub = self.create_publisher(
            Image, '/camera/image_raw_source', 10
        )

        if source.is_dir():
            # Image sequence mode — find all images and sort them
            extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
            self.image_files = sorted([
                str(f) for f in source.iterdir()
                if f.suffix.lower() in extensions
            ])

            if not self.image_files:
                self.get_logger().error(f'No images found in: {source}')
                return

            self.image_index = 0
            self.get_logger().info(
                f'Image sequence: {source} ({len(self.image_files)} frames)'
            )
            self.video_timer = self.create_timer(
                1.0 / self.playback_fps, self._image_seq_callback
            )

        elif source.is_file():
            # Video file mode
            self.video_cap = cv2.VideoCapture(str(source))
            if not self.video_cap.isOpened():
                self.get_logger().error(f'Cannot open video: {source}')
                return

            self.playback_fps = self.video_cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.get_logger().info(
                f'Video file: {source} at {self.playback_fps:.1f} FPS'
            )
            self.video_timer = self.create_timer(
                1.0 / self.playback_fps, self._video_frame_callback
            )
        else:
            self.get_logger().error(
                f'Source not found: {source} — provide a video file or image folder'
            )

    def _make_header(self) -> Header:
        """Create a stamped header for the current frame."""
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'camera_frame'
        return header

    def _publish_and_detect(self, frame: np.ndarray) -> None:
        """Publish a frame as a ROS Image and run detection on it.

        Always publishes the raw frame to /camera/image_raw_source.
        Only runs detection here if no degraded frames are being received
        (i.e., no attack node is active). If an attack node is upstream,
        detection runs via _degraded_image_callback instead.
        """
        header = self._make_header()
        img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        img_msg.header = header
        self.raw_image_pub.publish(img_msg)

        # Only detect on clean frames if no attack node is providing degraded frames
        if not self.use_degraded:
            self._run_detection(frame, header)

    def _image_seq_callback(self) -> None:
        """Read next image from the sequence folder."""
        if self.image_index >= len(self.image_files):
            # Loop back to start
            self.image_index = 0

        path = self.image_files[self.image_index]
        frame = cv2.imread(path)
        self.image_index += 1

        if frame is None:
            self.get_logger().warn(f'Failed to read: {path}')
            return

        self._publish_and_detect(frame)

    def _video_frame_callback(self) -> None:
        """Read next frame from a video file."""
        if self.video_cap is None or not self.video_cap.isOpened():
            return

        ret, frame = self.video_cap.read()
        if not ret:
            # loop the video
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.video_cap.read()
            if not ret:
                self.get_logger().warn('Video playback ended')
                self.video_timer.cancel()
                return

        self._publish_and_detect(frame)

    def _image_callback(self, msg: Image) -> None:
        """Process an incoming raw image message through YOLOv8."""
        if self.use_degraded:
            return  # Skip raw frames when degraded frames are available

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CV Bridge error: {e}')
            return

        self._run_detection(frame, msg.header)
        self.image_pub.publish(msg)

    def _degraded_image_callback(self, msg: Image) -> None:
        """Process a degraded (attacked) image through YOLOv8.

        When this fires, it means an attack node is upstream. We switch
        to using degraded frames for detection instead of clean ones.
        """
        self.use_degraded = True

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CV Bridge error: {e}')
            return

        self._run_detection(frame, msg.header)
        self.image_pub.publish(msg)

    def _run_detection(self, frame: np.ndarray, header: Header) -> None:
        """Run YOLOv8 inference and publish detections."""
        self.frame_count += 1

        # run inference
        results = self.model(
            frame,
            conf=self.conf_thresh,
            iou=self.nms_thresh,
            imgsz=self.input_size,
            classes=self.target_classes,
            verbose=False,
        )

        # build Detection2DArray message
        det_array = Detection2DArray()
        det_array.header = header

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    det = Detection2D()
                    det.header = header

                    # Bounding box: YOLO gives xyxy, we need center + size
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    det.bbox.center.position.x = float((x1 + x2) / 2.0)
                    det.bbox.center.position.y = float((y1 + y2) / 2.0)
                    det.bbox.size_x = float(x2 - x1)
                    det.bbox.size_y = float(y2 - y1)

                    # Class hypothesis
                    hyp = ObjectHypothesisWithPose()
                    hyp.hypothesis.class_id = str(int(box.cls[0].item()))
                    hyp.hypothesis.score = float(box.conf[0].item())
                    det.results.append(hyp)

                    det_array.detections.append(det)

        self.detection_pub.publish(det_array)

        if self.frame_count % 100 == 0:
            self.get_logger().info(
                f'Frame {self.frame_count}: {len(det_array.detections)} detections'
            )

    def destroy_node(self) -> None:
        """Clean up resources."""
        if self.video_cap is not None:
            self.video_cap.release()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
