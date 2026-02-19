FROM ros:humble

# Install Python deps and OpenCV system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-opencv \
    ros-humble-cv-bridge \
    ros-humble-vision-msgs \
    ros-humble-image-transport \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages — pin numpy<2 because cv_bridge was compiled against 1.x
RUN pip3 install --no-cache-dir \
    "numpy<2" \
    ultralytics \
    opencv-python-headless \
    scipy \
    motmetrics \
    matplotlib

# Set up workspace
WORKDIR /tracking_ws
COPY tracking_ws/src src/

# Build the ROS2 package
RUN . /opt/ros/humble/setup.sh && \
    colcon build --packages-select tracking_adversarial

# Source both ROS2 and our package on every shell
# Also add a rebuild alias for convenience when code changes
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /tracking_ws/install/setup.bash" >> /root/.bashrc && \
    echo "alias rebuild='cd /tracking_ws && colcon build --packages-select tracking_adversarial && source install/setup.bash'" >> /root/.bashrc

CMD ["bash"]
