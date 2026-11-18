"""Tests for the ROS2 image helpers. Run INSIDE the Docker container."""
import sys
import unittest
from pathlib import Path

import numpy as np
from builtin_interfaces.msg import Time
from sensor_msgs.msg import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "ros2_nodes"))
from ros_utils import array_to_image, image_to_array  # noqa: E402


class TestRosUtils(unittest.TestCase):
    def test_round_trip(self):
        frame = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
        msg = array_to_image(frame, Time())
        self.assertEqual((msg.height, msg.width, msg.encoding), (48, 64, "bgr8"))
        self.assertTrue(np.array_equal(image_to_array(msg), frame))

    def test_wrong_encoding_is_rejected(self):
        msg = array_to_image(np.zeros((4, 4, 3), dtype=np.uint8), Time())
        msg.encoding = "rgb8"
        with self.assertRaises(ValueError):
            image_to_array(msg)

    def test_wrong_data_size_is_rejected(self):
        msg = array_to_image(np.zeros((4, 4, 3), dtype=np.uint8), Time())
        msg.data = msg.data[:-5]
        with self.assertRaises(ValueError):
            image_to_array(msg)

    def test_empty_image_is_rejected(self):
        with self.assertRaises(ValueError):
            image_to_array(Image())


if __name__ == "__main__":
    unittest.main()