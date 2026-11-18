"""Tests for the classical lane detector (runs on the Mac, no ROS2 needed)."""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "lanes"))
from classical import LaneDetector  # noqa: E402


class TestLaneDetector(unittest.TestCase):
    def test_black_frame_finds_no_lanes(self):
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        overlay, found = LaneDetector().detect(frame)
        self.assertEqual(overlay.shape, frame.shape)
        self.assertFalse(found["left"])
        self.assertFalse(found["right"])

    def test_tiny_frame_does_not_crash(self):
        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        overlay, _ = LaneDetector().detect(frame)
        self.assertEqual(overlay.shape, frame.shape)

    def test_synthetic_lane_lines_are_found(self):
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        cv2.line(frame, (270, 415), (415, 320), (255, 255, 255), 6)  # left marking
        cv2.line(frame, (700, 415), (545, 320), (255, 255, 255), 6)  # right marking
        _, found = LaneDetector(smooth=0.0).detect(frame)
        self.assertTrue(found["left"])
        self.assertTrue(found["right"])


if __name__ == "__main__":
    unittest.main()