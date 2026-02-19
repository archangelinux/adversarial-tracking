"""Baseline launch file — detection + tracking + visualization.

No attacks, no defenses. Establishes baseline performance.

Usage:
    ros2 launch tracking_adversarial baseline.launch.py
    ros2 launch tracking_adversarial baseline.launch.py video_source:=/path/to/video.mp4
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare arguments that can be passed from the command line
    video_source_arg = DeclareLaunchArgument(
        'video_source',
        default_value='',
        description='Path to video file. Empty string uses live camera.',
    )

    model_arg = DeclareLaunchArgument(
        'model',
        default_value='yolov8n.pt',
        description='YOLOv8 model variant (yolov8n/s/m/l/x.pt)',
    )

    device_arg = DeclareLaunchArgument(
        'device',
        default_value='cpu',
        description='Inference device: cpu, cuda:0, or mps',
    )

    # Shared parameter values from launch arguments
    video_source = LaunchConfiguration('video_source')
    model = LaunchConfiguration('model')
    device = LaunchConfiguration('device')

    # --- Detection Node ---
    # Reads video frames, runs YOLOv8, publishes /detections
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

    # --- Tracking Node ---
    # Subscribes to /detections, runs ByteTrack, publishes /tracks
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

    # --- Web Visualizer ---
    # Subscribes to /detections + /tracks + /camera/image_raw_source
    # Serves annotated video at http://localhost:8080
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
        detection_node,
        tracking_node,
        web_visualizer,
    ])
