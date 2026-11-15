"""Helpers to convert between numpy images and ROS2 sensor_msgs/Image (bgr8)."""
import numpy as np
from sensor_msgs.msg import Image


def array_to_image(frame, stamp, frame_id="camera"):
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(frame.shape[0])
    msg.width = int(frame.shape[1])
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = np.ascontiguousarray(frame).tobytes()
    return msg


def image_to_array(msg):
    """Return a BGR numpy image, or raise ValueError if the message is not a valid bgr8 image."""
    if msg.encoding != "bgr8":
        raise ValueError(f"unsupported encoding '{msg.encoding}' (expected bgr8)")
    if msg.height == 0 or msg.width == 0:
        raise ValueError("empty image")
    expected = msg.height * msg.width * 3
    if len(msg.data) != expected:
        raise ValueError(f"bad data size {len(msg.data)}, expected {expected}")
    arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    return arr.reshape(msg.height, msg.width, 3).copy()