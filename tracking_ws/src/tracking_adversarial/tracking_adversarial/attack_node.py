"""Attack node for the adversarial tracking framework.

Subscribes to camera images, applies environmental degradation or
adversarial attacks, and publishes degraded images for downstream
detection/tracking to process.
"""

from typing import List, Tuple

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge

from tracking_adversarial.utils.degradation import apply_degradation
from tracking_adversarial.utils.adversarial import apply_adversarial_attack


# Environmental degradation types (applied to the whole image)
ENVIRONMENTAL_ATTACKS = {'rain', 'fog', 'blur', 'low_light', 'lens_dirt', 'contrast'}

# Adversarial attacks (applied to specific bounding boxes)
ADVERSARIAL_ATTACKS = {'patch', 'stripe', 'checkerboard', 'occlusion'}

# Gradient-based attacks (FGSM, PGD) — benchmark-only.
# These require direct access to the model's PyTorch internals to compute
# gradients, which the attack node doesn't have (it only sees images).
# Use scripts/benchmark.py to run these attacks.
GRADIENT_ATTACKS = {'fgsm_light', 'fgsm_heavy', 'pgd_light', 'pgd_heavy'}


class AttackNode(Node):
    """ROS2 node that applies attacks to camera images."""

    def __init__(self) -> None:
        super().__init__('attack_node')

        # Declare parameters
        self.declare_parameter('attack.enabled', False)
        self.declare_parameter('attack.type', 'none')
        self.declare_parameter('attack.intensity', 0.5)
        self.declare_parameter('attack.input_topic', '/camera/image_raw_source')
        self.declare_parameter('attack.patch_size', 50)
        self.declare_parameter('attack.stripe_width', 10)
        self.declare_parameter('attack.checkerboard_size', 15)
        self.declare_parameter('attack.occlusion_ratio', 0.3)

        # Load parameters
        self.attack_enabled = self.get_parameter('attack.enabled').value
        self.attack_type = self.get_parameter('attack.type').value
        self.patch_size = self.get_parameter('attack.patch_size').value
        self.stripe_width = self.get_parameter('attack.stripe_width').value
        self.checkerboard_size = self.get_parameter('attack.checkerboard_size').value
        self.occlusion_ratio = self.get_parameter('attack.occlusion_ratio').value
        input_topic = self.get_parameter('attack.input_topic').value

        # Ensure intensity is float (LaunchConfiguration may pass strings)
        raw_intensity = self.get_parameter('attack.intensity').value
        self.intensity = float(raw_intensity)

        self.bridge = CvBridge()

        # Cache of recent detections for adversarial attacks that target objects
        self.latest_bboxes: List[Tuple[int, int, int, int]] = []

        # Subscribe to raw camera image (configurable topic)
        # Use default RELIABLE QoS with depth 10 — matches the publisher
        self.image_sub = self.create_subscription(
            Image, input_topic, self._image_callback, 10
        )

        # Subscribe to detections so we can target adversarial attacks at objects
        self.det_sub = self.create_subscription(
            Detection2DArray, '/detections', self._detection_callback, 10
        )

        # Publish degraded image
        self.degraded_pub = self.create_publisher(
            Image, '/camera/degraded', 10
        )

        # Parameter callback for dynamic reconfiguration
        self.add_on_set_parameters_callback(self._param_callback)

        self.frame_count = 0
        status = 'ACTIVE' if self.attack_enabled else 'STANDBY'
        self.get_logger().info(
            f'Attack node initialized [{status}] — type: {self.attack_type}, '
            f'intensity: {self.intensity}, listening on: {input_topic}'
        )

    def _param_callback(self, params):
        """Handle dynamic parameter updates."""
        from rcl_interfaces.msg import SetParametersResult
        for param in params:
            if param.name == 'attack.enabled':
                self.attack_enabled = param.value
            elif param.name == 'attack.type':
                self.attack_type = param.value
            elif param.name == 'attack.intensity':
                self.intensity = float(param.value)
        return SetParametersResult(successful=True)

    def _detection_callback(self, msg: Detection2DArray) -> None:
        """Cache detection bounding boxes for adversarial attacks."""
        self.latest_bboxes = []
        for det in msg.detections:
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            w = det.bbox.size_x
            h = det.bbox.size_y
            x1, y1 = int(cx - w / 2), int(cy - h / 2)
            x2, y2 = int(cx + w / 2), int(cy + h / 2)
            self.latest_bboxes.append((x1, y1, x2, y2))

    def _image_callback(self, msg: Image) -> None:
        """Apply attack to incoming image and publish."""
        if not self.attack_enabled or self.attack_type == 'none':
            self.degraded_pub.publish(msg)
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'CV Bridge error: {e}')
            return

        self.frame_count += 1

        # Apply the configured attack
        try:
            if self.attack_type in ENVIRONMENTAL_ATTACKS:
                attacked = apply_degradation(
                    frame, self.attack_type, self.intensity
                )
            elif self.attack_type in ADVERSARIAL_ATTACKS:
                attacked = apply_adversarial_attack(
                    frame,
                    self.attack_type,
                    bboxes=self.latest_bboxes if self.latest_bboxes else None,
                    intensity=self.intensity,
                    patch_size=self.patch_size,
                    stripe_width=self.stripe_width,
                    checkerboard_size=self.checkerboard_size,
                    occlusion_ratio=self.occlusion_ratio,
                )
            elif self.attack_type in GRADIENT_ATTACKS:
                self.get_logger().warn(
                    f'Gradient attack "{self.attack_type}" requires model access '
                    f'and cannot run in the attack node. Use scripts/benchmark.py '
                    f'instead. Passing through unchanged.'
                )
                attacked = frame
            else:
                self.get_logger().warn(
                    f'Unknown attack type: {self.attack_type}, passing through'
                )
                attacked = frame
        except Exception as e:
            self.get_logger().error(f'Attack application failed: {e}')
            attacked = frame

        # Publish degraded image
        try:
            out_msg = self.bridge.cv2_to_imgmsg(attacked, encoding='bgr8')
            out_msg.header = msg.header
            self.degraded_pub.publish(out_msg)
        except Exception as e:
            self.get_logger().error(f'Failed to publish degraded image: {e}')

        if self.frame_count == 1:
            self.get_logger().info(
                f'First frame received — applying {self.attack_type} '
                f'(intensity={self.intensity:.2f})'
            )
        elif self.frame_count % 100 == 0:
            self.get_logger().info(
                f'Frame {self.frame_count}: Applied {self.attack_type} '
                f'(intensity={self.intensity:.2f})'
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AttackNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
