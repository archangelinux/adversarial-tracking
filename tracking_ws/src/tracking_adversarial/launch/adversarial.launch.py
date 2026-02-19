"""Adversarial launch file — baseline + attacks enabled.

The attack node reads raw frames from /camera/image_raw_source,
applies the attack, and publishes to /camera/degraded. The detection
node auto-detects degraded frames and switches to using them.

Usage:
    ros2 launch tracking_adversarial adversarial.launch.py \
        video_source:=/path/to/video attack_type:=fog attack_intensity:=0.7
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    video_source_arg = DeclareLaunchArgument(
        'video_source', default_value='',
        description='Path to video file or image sequence folder',
    )
    model_arg = DeclareLaunchArgument(
        'model', default_value='yolov8n.pt',
        description='YOLOv8 model variant',
    )
    device_arg = DeclareLaunchArgument(
        'device', default_value='cpu',
        description='Inference device: cpu, cuda:0, or mps',
    )
    attack_type_arg = DeclareLaunchArgument(
        'attack_type', default_value='fog',
        description='Attack type: rain, fog, blur, low_light, lens_dirt, contrast, patch, stripe, checkerboard, occlusion',
    )
    attack_intensity_arg = DeclareLaunchArgument(
        'attack_intensity', default_value='0.5',
        description='Attack intensity 0.0-1.0',
    )

    video_source = LaunchConfiguration('video_source')
    model = LaunchConfiguration('model')
    device = LaunchConfiguration('device')
    attack_type = LaunchConfiguration('attack_type')
    attack_intensity = LaunchConfiguration('attack_intensity')

    # Detection node — reads video, publishes raw to /camera/image_raw_source.
    # Also subscribes to /camera/degraded and auto-switches when attack node is active.
    detection_node = Node(
        package='tracking_adversarial',
        executable='detection_node',
        name='detection_node',
        output='screen',
        parameters=[{
            'video_source': video_source,
            'detection.model': model,
            'detection.device': device,
            'detection.confidence_threshold': 0.25,
            'detection.nms_threshold': 0.45,
            'detection.classes': [0, 1, 2, 3, 5, 7],
            'detection.input_size': 640,
        }],
    )

    # Attack node — subscribes to /camera/image_raw_source, publishes /camera/degraded
    attack_node = Node(
        package='tracking_adversarial',
        executable='attack_node',
        name='attack_node',
        output='screen',
        parameters=[{
            'attack.enabled': True,
            'attack.type': attack_type,
            'attack.intensity': attack_intensity,
            'attack.input_topic': '/camera/image_raw_source',
        }],
    )

    # Tracking node
    tracking_node = Node(
        package='tracking_adversarial',
        executable='tracking_node',
        name='tracking_node',
        output='screen',
        parameters=[{
            'tracking.track_high_thresh': 0.5,
            'tracking.track_low_thresh': 0.1,
            'tracking.new_track_thresh': 0.6,
            'tracking.track_buffer': 30,
            'tracking.match_thresh': 0.8,
            'tracking.min_box_area': 10.0,
        }],
    )

    # Web visualizer — subscribes to both raw and degraded, shows whichever is available
    web_visualizer = Node(
        package='tracking_adversarial',
        executable='web_visualizer',
        name='web_visualizer',
        output='screen',
        parameters=[{
            'web.port': 8080,
            'visualization.show_detections': True,
            'visualization.show_tracks': True,
            'visualization.show_ids': True,
            'visualization.show_trails': True,
            'visualization.trail_length': 30,
            'visualization.bbox_thickness': 2,
            'visualization.font_scale': 0.6,
        }],
    )

    return LaunchDescription([
        video_source_arg,
        model_arg,
        device_arg,
        attack_type_arg,
        attack_intensity_arg,
        detection_node,
        attack_node,
        tracking_node,
        web_visualizer,
    ])
