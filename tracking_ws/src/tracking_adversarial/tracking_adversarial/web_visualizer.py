"""Web-based visualization dashboard for the tracking framework.

Serves an MJPEG stream of annotated frames over HTTP so you can
view live tracking results in any browser.

Open http://localhost:8080 to see the dashboard.
"""

import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import colorsys


# COCO class names for display
COCO_NAMES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
    5: 'bus', 7: 'truck',
}

# Global reference to the latest frame — shared between ROS2 thread and HTTP thread
_latest_frame: Optional[bytes] = None
_frame_lock = threading.Lock()

# Metrics state
_metrics: dict = {
    'frame_count': 0,
    'num_detections': 0,
    'num_tracks': 0,
}


def _id_to_color(track_id: int):
    """Generate a unique BGR color for a track ID."""
    hue = (track_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


class MJPEGHandler(BaseHTTPRequestHandler):
    """HTTP handler that serves an MJPEG stream and a simple HTML dashboard."""

    def do_GET(self):
        if self.path == '/':
            self._serve_dashboard()
        elif self.path == '/stream':
            self._serve_mjpeg()
        else:
            self.send_error(404)

    def _serve_dashboard(self):
        """Serve a simple HTML page with the video stream and metrics."""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>Tracking Dashboard</title>
    <style>
        body {
            background: #1a1a2e; color: #eee; font-family: 'Courier New', monospace;
            margin: 0; padding: 20px; display: flex; flex-direction: column; align-items: center;
        }
        h1 { color: #e94560; margin-bottom: 10px; }
        .subtitle { color: #888; margin-bottom: 20px; }
        img {
            border: 2px solid #e94560; border-radius: 4px;
            max-width: 95vw; max-height: 80vh;
        }
        .info { margin-top: 15px; color: #0f3460; background: #e94560;
                padding: 8px 16px; border-radius: 4px; font-weight: bold; }
    </style>
</head>
<body>
    <h1>Adversarial Tracking Dashboard</h1>
    <p class="subtitle">Live detection + tracking visualization</p>
    <img src="/stream" alt="Live Stream" />
    <p class="info">Detection (YOLOv8) + Tracking (ByteTrack) | Green = detections | Colored = tracks</p>
</body>
</html>"""
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())

    def _serve_mjpeg(self):
        """Serve a continuous MJPEG stream."""
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()

        while True:
            with _frame_lock:
                frame_bytes = _latest_frame

            if frame_bytes is None:
                # No frame yet — send a blank placeholder
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, 'Waiting for frames...', (120, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
                _, buf = cv2.imencode('.jpg', blank)
                frame_bytes = buf.tobytes()

            try:
                self.wfile.write(b'--frame\r\n')
                self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                self.wfile.write(frame_bytes)
                self.wfile.write(b'\r\n')
            except BrokenPipeError:
                break

            # ~30 FPS cap
            threading.Event().wait(0.033)

    def log_message(self, format, *args):
        """Suppress default HTTP logging — too noisy."""
        pass


class WebVisualizer(Node):
    """ROS2 node that renders annotated frames and serves them over HTTP."""

    def __init__(self) -> None:
        super().__init__('web_visualizer')

        self.declare_parameter('web.port', 8080)
        self.declare_parameter('visualization.show_detections', True)
        self.declare_parameter('visualization.show_tracks', True)
        self.declare_parameter('visualization.show_ids', True)
        self.declare_parameter('visualization.show_trails', True)
        self.declare_parameter('visualization.trail_length', 30)
        self.declare_parameter('visualization.bbox_thickness', 2)
        self.declare_parameter('visualization.font_scale', 0.6)

        self.port = self.get_parameter('web.port').value
        self.show_detections = self.get_parameter('visualization.show_detections').value
        self.show_tracks = self.get_parameter('visualization.show_tracks').value
        self.show_ids = self.get_parameter('visualization.show_ids').value
        self.show_trails = self.get_parameter('visualization.show_trails').value
        self.trail_length = self.get_parameter('visualization.trail_length').value
        self.bbox_thickness = self.get_parameter('visualization.bbox_thickness').value
        self.font_scale = self.get_parameter('visualization.font_scale').value

        self.bridge = CvBridge()

        # State
        self.latest_image: Optional[np.ndarray] = None
        self.latest_detections: Optional[Detection2DArray] = None
        self.latest_tracks: Optional[Detection2DArray] = None
        self.track_trails: dict = {}  # {track_id: [(cx, cy), ...]}
        self.frame_count = 0
        self.use_degraded = False  # Switch to degraded-only once attack node is active

        # Subscribers — listen to both raw and degraded image topics.
        # In baseline mode, only raw publishes. In adversarial mode,
        # degraded publishes and takes priority (raw is ignored).
        self.create_subscription(
            Image, '/camera/image_raw_source', self._raw_image_cb, 10
        )
        self.create_subscription(
            Image, '/camera/degraded', self._degraded_image_cb, 10
        )
        self.create_subscription(
            Detection2DArray, '/detections', self._det_cb, 10
        )
        self.create_subscription(
            Detection2DArray, '/tracks', self._track_cb, 10
        )
        self.create_subscription(
            Detection2DArray, '/tracks/defended', self._track_defended_cb, 10
        )

        # Render timer — 30 FPS
        self.create_timer(1.0 / 30.0, self._render)

        # Start HTTP server in a background thread
        self._start_http_server()

        self.get_logger().info(
            f'Web visualizer running — open http://localhost:{self.port}'
        )

    def _start_http_server(self):
        server = HTTPServer(('0.0.0.0', self.port), MJPEGHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

    def _raw_image_cb(self, msg: Image):
        """Use raw frames only when no attack node is active."""
        if self.use_degraded:
            return
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'CV Bridge error: {e}')

    def _degraded_image_cb(self, msg: Image):
        """Once degraded frames arrive, use them exclusively."""
        self.use_degraded = True
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'CV Bridge error: {e}')

    def _track_defended_cb(self, msg: Detection2DArray):
        """If defended tracks are available, use them instead of raw tracks."""
        self._track_cb(msg)

    def _det_cb(self, msg: Detection2DArray):
        self.latest_detections = msg

    def _track_cb(self, msg: Detection2DArray):
        self.latest_tracks = msg
        # Update trails
        active_ids = set()
        for det in msg.detections:
            tid = int(det.id)
            active_ids.add(tid)
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            if tid not in self.track_trails:
                self.track_trails[tid] = []
            self.track_trails[tid].append((cx, cy))
            if len(self.track_trails[tid]) > self.trail_length:
                self.track_trails[tid] = self.track_trails[tid][-self.trail_length:]

        # Clean stale trails
        for tid in list(self.track_trails.keys()):
            if tid not in active_ids:
                trail = self.track_trails[tid]
                if len(trail) > 1:
                    self.track_trails[tid] = trail[1:]
                else:
                    del self.track_trails[tid]

    def _render(self):
        """Compose the annotated frame and update the global JPEG buffer."""
        global _latest_frame

        if self.latest_image is None:
            return

        self.frame_count += 1
        frame = self.latest_image.copy()
        h, w = frame.shape[:2]

        num_dets = 0
        num_tracks = 0

        # Draw detections — thin green boxes
        if self.show_detections and self.latest_detections:
            num_dets = len(self.latest_detections.detections)
            for det in self.latest_detections.detections:
                cx = det.bbox.center.position.x
                cy = det.bbox.center.position.y
                bw = det.bbox.size_x
                bh = det.bbox.size_y
                x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
                x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

                if det.results:
                    cls_id = det.results[0].hypothesis.class_id
                    score = det.results[0].hypothesis.score
                    label = COCO_NAMES.get(int(cls_id), cls_id)
                    cv2.putText(frame, f'{label} {score:.2f}', (x1, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, self.font_scale * 0.7,
                                (0, 255, 0), 1)

        # Draw tracks — thick colored boxes with IDs
        if self.show_tracks and self.latest_tracks:
            num_tracks = len(self.latest_tracks.detections)
            for det in self.latest_tracks.detections:
                tid = int(det.id)
                color = _id_to_color(tid)
                cx = det.bbox.center.position.x
                cy = det.bbox.center.position.y
                bw = det.bbox.size_x
                bh = det.bbox.size_y
                x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
                x2, y2 = int(cx + bw / 2), int(cy + bh / 2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.bbox_thickness)

                if self.show_ids:
                    label = f'ID:{tid}'
                    if det.results:
                        cls_id = det.results[0].hypothesis.class_id
                        cls_name = COCO_NAMES.get(int(cls_id), cls_id)
                        label = f'{cls_name} #{tid}'
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 2)
                    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                                (255, 255, 255), 2)

        # Draw trails
        if self.show_trails:
            for tid, trail in self.track_trails.items():
                if len(trail) < 2:
                    continue
                color = _id_to_color(tid)
                for i in range(1, len(trail)):
                    alpha = i / len(trail)
                    pt1 = (int(trail[i - 1][0]), int(trail[i - 1][1]))
                    pt2 = (int(trail[i][0]), int(trail[i][1]))
                    thickness = max(1, int(alpha * 3))
                    cv2.line(frame, pt1, pt2, color, thickness)

        # HUD overlay — stats bar at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 32), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        stats = (
            f'Frame: {self.frame_count}  |  '
            f'Detections: {num_dets}  |  '
            f'Tracks: {num_tracks}'
        )
        cv2.putText(frame, stats, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)

        # Encode to JPEG and update global buffer
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with _frame_lock:
            _latest_frame = buf.tobytes()


def main(args=None):
    rclpy.init(args=args)
    node = WebVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
