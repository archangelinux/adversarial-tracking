from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'tracking_adversarial'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*')),
    ],
    install_requires=[
        'setuptools',
        'ultralytics',
        'boxmot',
        'opencv-python',
        'numpy',
        'motmetrics',
        'matplotlib',
        'scipy',
    ],
    zip_safe=True,
    maintainer='angie',
    maintainer_email='angie@example.com',
    description='Adversarial perception testing framework for object tracking',
    license='MIT',
    entry_points={
        'console_scripts': [
            'detection_node = tracking_adversarial.detection_node:main',
            'tracking_node = tracking_adversarial.tracking_node:main',
            'attack_node = tracking_adversarial.attack_node:main',
            'visualization_node = tracking_adversarial.visualization_node:main',
            'web_visualizer = tracking_adversarial.web_visualizer:main',
            'defense_node = tracking_adversarial.defense_node:main',
            'evaluation_node = tracking_adversarial.evaluation_node:main',
        ],
    },
)
