"""ROS2 node: plays a video file as if it were a live camera.

Publishes sensor_msgs/Image (bgr8) on /camera/image_raw.
Parameters: video, rate_hz, stride, width, max_frames.
"""
import time

import cv2
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import Image

from ros_utils import array_to_image


class VideoPublisher(Node):
    def __init__(self):
        super().__init__("video_publisher")
        self.declare_parameter("video", "/ws/data/raw/videos/clip.mp4")
        self.declare_parameter("rate_hz", 2.0)
        self.declare_parameter("stride", 3)       # publish every Nth video frame
        self.declare_parameter("width", 960)      # resize so messages stay small
        self.declare_parameter("max_frames", 0)   # 0 = whole video

        path = self.get_parameter("video").value
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        self.stride = max(1, int(self.get_parameter("stride").value))
        self.width = int(self.get_parameter("width").value)
        self.max_frames = int(self.get_parameter("max_frames").value)
        rate = float(self.get_parameter("rate_hz").value)
        if rate <= 0:
            raise ValueError("rate_hz must be greater than 0")

        self.pub = self.create_publisher(Image, "/camera/image_raw", 10)
        self.sent = 0
        self.start_at = None
        self.timer = self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(f"Video: {path}. Waiting for a subscriber on /camera/image_raw ...")

    def tick(self):
        if self.start_at is None:
            if self.pub.get_subscription_count() == 0:
                return
            self.start_at = time.time() + 3.0  # short pause so every node is ready
            self.get_logger().info("Subscriber found, starting in 3 s")
        if time.time() < self.start_at:
            return

        ok = True
        for _ in range(self.stride - 1):  # skip frames between published ones
            ok = self.cap.grab()
            if not ok:
                break
        frame = None
        if ok:
            ok, frame = self.cap.read()
        if not ok or frame is None or (self.max_frames and self.sent >= self.max_frames):
            self.get_logger().info(f"Done: published {self.sent} frames")
            self.timer.cancel()
            return

        h, w = frame.shape[:2]
        if w != self.width:
            frame = cv2.resize(frame, (self.width, int(h * self.width / w)))
        self.pub.publish(array_to_image(frame, self.get_clock().now().to_msg()))
        self.sent += 1


def main():
    rclpy.init()
    node = VideoPublisher()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()