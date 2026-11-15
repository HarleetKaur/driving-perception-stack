"""ROS2 node: records /perception/overlay frames to an mp4 file.

Stops by itself when no frame has arrived for idle_timeout_s seconds.
Parameters: output, fps, idle_timeout_s.
"""
import os
import time

import cv2
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import Image

from ros_utils import image_to_array


class OverlayRecorder(Node):
    def __init__(self):
        super().__init__("overlay_recorder")
        self.declare_parameter("output", "/ws/results/ros2/ros2_demo.mp4")
        self.declare_parameter("fps", 10.0)
        self.declare_parameter("idle_timeout_s", 20.0)
        self.output = self.get_parameter("output").value
        self.fps = float(self.get_parameter("fps").value)
        self.idle_timeout = float(self.get_parameter("idle_timeout_s").value)

        self.writer = None
        self.frames = 0
        self.last_rx = None
        self.done = False
        self.create_subscription(Image, "/perception/overlay", self.on_image, 10)
        self.create_timer(1.0, self.check_idle)
        self.get_logger().info(f"Waiting for frames; will write {self.output} at {self.fps} fps")

    def on_image(self, msg):
        try:
            frame = image_to_array(msg)
        except ValueError as e:
            self.get_logger().warn(f"Skipping bad image: {e}")
            return
        if self.writer is None:
            os.makedirs(os.path.dirname(self.output), exist_ok=True)
            h, w = frame.shape[:2]
            self.writer = cv2.VideoWriter(
                self.output, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        self.writer.write(frame)
        self.frames += 1
        self.last_rx = time.time()
        if self.frames % 20 == 0:
            self.get_logger().info(f"Recorded {self.frames} frames")

    def check_idle(self):
        if self.last_rx is not None and time.time() - self.last_rx > self.idle_timeout:
            self.finish()

    def finish(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        self.get_logger().info(f"Saved {self.frames} frames to {self.output}")
        self.done = True


def main():
    rclpy.init()
    node = OverlayRecorder()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.5)
    except (KeyboardInterrupt, ExternalShutdownException):
        if not node.done:
            node.finish()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()