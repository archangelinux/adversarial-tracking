"""Evaluation launch file — run tracking with metrics computation.

Can be used in any mode (baseline, adversarial, defended) by setting
the attack and defense parameters. Adds the evaluation node that
compares tracker output against ground truth annotations.

Usage:
    # Baseline evaluation
    ros2 launch tracking_adversarial evaluation.launch.py \
        video_source:=/path/to/images \
        ground_truth:=/path/to/annotations.txt

    # Adversarial evaluation
    ros2 launch tracking_adversarial evaluation.launch.py \
        video_source:=/path/to/images \
        ground_truth:=/path/to/annotations.txt \
        attack_enabled:=true attack_type:=fog attack_intensity:=0.5

    # Defended evaluation
    ros2 launch tracking_adversarial evaluation.launch.py \
        video_source:=/path/to/images \
        ground_truth:=/path/to/annotations.txt \
        attack_enabled:=true attack_type:=fog attack_intensity:=0.5 \
        defense_enabled:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # --- Arguments ---
    args = [
        DeclareLaunchArgument('video_source', default_value=''),
        DeclareLaunchArgument('model', default_value='yolov8n.pt'),
        DeclareLaunchArgument('device', default_value='cpu'),
        DeclareLaunchArgument('ground_truth', default_value='',
                              description='Path to VisDrone annotation file'),
        DeclareLaunchArgument('output_dir', default_value='/tracking_ws/data/results'),
        DeclareLaunchArgument('attack_enabled', default_value='false'),
        DeclareLaunchArgument('attack_type', default_value='fog'),
        DeclareLaunchArgument('attack_intensity', default_value='0.5'),
        DeclareLaunchArgument('defense_enabled', default_value='false'),
    ]

    video_source = LaunchConfiguration('video_source')
    model = LaunchConfiguration('model')
    device = LaunchConfiguration('device')
    ground_truth = LaunchConfiguration('ground_truth')
    output_dir = LaunchConfiguration('output_dir')
    attack_enabled = LaunchConfiguration('attack_enabled')
    attack_type = LaunchConfiguration('attack_type')
    attack_intensity = LaunchConfiguration('attack_intensity')
    defense_enabled = LaunchConfiguration('defense_enabled')

    # --- Detection Node ---
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

    # --- Attack Node (conditional) ---
    attack_node = Node(
        package='tracking_adversarial',
        executable='attack_node',
        name='attack_node',
        output='screen',
        condition=IfCondition(attack_enabled),
        parameters=[{
            'attack.enabled': True,
            'attack.type': attack_type,
            'attack.intensity': attack_intensity,
        }],
        remappings=[
            ('/camera/image_raw', '/camera/image_raw_source'),
        ],
    )

    # --- Tracking Node ---
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

    # --- Defense Node (conditional) ---
    defense_node = Node(
        package='tracking_adversarial',
        executable='defense_node',
        name='defense_node',
        output='screen',
        condition=IfCondition(defense_enabled),
        parameters=[{
            'defense.enabled': True,
            'defense.temporal_consistency': True,
            'defense.anomaly_detection': True,
            'defense.max_position_jump': 100.0,
            'defense.history_length': 30,
            'defense.recovery_frames': 10,
            'defense.velocity_threshold': 3.0,
            'defense.acceleration_threshold': 3.0,
        }],
    )

    # --- Evaluation Node ---
    evaluation_node = Node(
        package='tracking_adversarial',
        executable='evaluation_node',
        name='evaluation_node',
        output='screen',
        parameters=[{
            'evaluation.ground_truth_file': ground_truth,
            'evaluation.output_dir': output_dir,
            'evaluation.save_csv': True,
            'evaluation.log_interval': 100,
            'evaluation.iou_threshold': 0.5,
        }],
    )

    # --- Web Visualizer ---
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
        *args,
        detection_node,
        attack_node,
        tracking_node,
        defense_node,
        evaluation_node,
        web_visualizer,
    ])
