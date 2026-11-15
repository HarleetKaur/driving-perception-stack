#!/usr/bin/env bash
# Run the 3-node ROS2 demo inside the Docker container.
# Usage: bash scripts/run_ros2_demo.sh [MAX_FRAMES] [STRIDE] [RATE_HZ]
#   MAX_FRAMES: 0 = whole video   STRIDE: use every Nth video frame   RATE_HZ: frames per second to publish
set -e
source /opt/ros/humble/setup.bash
cd /ws

MAX_FRAMES=${1:-0}
STRIDE=${2:-3}
RATE=$(printf "%.1f" "${3:-2.0}")
FPS=$(printf "%.1f" $((30 / STRIDE)))   # output plays at real-time speed for a 30 fps clip
mkdir -p results/ros2

python3 src/ros2_nodes/perception_node.py > results/ros2/perception.log 2>&1 &
PERC_PID=$!
python3 src/ros2_nodes/video_publisher.py --ros-args \
    -p max_frames:=$MAX_FRAMES -p stride:=$STRIDE -p rate_hz:=$RATE > results/ros2/publisher.log 2>&1 &
PUB_PID=$!
trap 'kill $PERC_PID $PUB_PID 2>/dev/null' EXIT

python3 src/ros2_nodes/overlay_recorder.py --ros-args -p fps:=$FPS