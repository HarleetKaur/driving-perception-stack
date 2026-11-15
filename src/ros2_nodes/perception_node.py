"""ROS2 node: detection + ByteTrack tracking + classical lane detection.

Subscribes:  /camera/image_raw       (sensor_msgs/Image, bgr8)
Publishes:   /perception/overlay     (sensor_msgs/Image, annotated frame)
             /perception/detections  (std_msgs/String, JSON: frame, objects, lanes, timings)
Parameters:  weights, imgsz, conf, roi_bottom_y
"""
import json
import sys
import time
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLO

from ros_utils import array_to_image, image_to_array

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "lanes"))
from classical import LaneDetector  # noqa: E402

COLOURS = {"vehicle": (0, 200, 0), "pedestrian": (0, 0, 255), "cyclist": (255, 128, 0)}


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")
        self.declare_parameter("weights", str(ROOT / "models" / "improved_best.pt"))
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("roi_bottom_y", 0.70)

        weights = Path(self.get_parameter("weights").value)
        if not weights.exists():
            raise FileNotFoundError(f"Weights not found: {weights}")
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.conf = float(self.get_parameter("conf").value)

        self.get_logger().info(f"Loading model {weights} ...")
        self.model = YOLO(str(weights))
        self.lanes = LaneDetector(roi={"bottom_y": float(self.get_parameter("roi_bottom_y").value)})
        self.count = 0

        self.overlay_pub = self.create_publisher(Image, "/perception/overlay", 10)
        self.det_pub = self.create_publisher(String, "/perception/detections", 10)
        self.create_subscription(Image, "/camera/image_raw", self.on_image, 10)
        self.get_logger().info("Ready: waiting for frames on /camera/image_raw")

    def on_image(self, msg):
        try:
            frame = image_to_array(msg)
        except ValueError as e:
            self.get_logger().warn(f"Skipping bad image: {e}")
            return

        t0 = time.time()
        overlay, found = self.lanes.detect(frame)
        t1 = time.time()
        res = self.model.track(
            frame, persist=True, tracker="bytetrack.yaml",
            conf=self.conf, imgsz=self.imgsz, device="cpu", verbose=False,
        )[0]
        t2 = time.time()

        objects = []
        if res.boxes is not None and len(res.boxes) > 0:
            n = len(res.boxes)
            ids = res.boxes.id.int().tolist() if res.boxes.id is not None else [-1] * n
            classes = res.boxes.cls.int().tolist()
            confs = res.boxes.conf.tolist()
            for tid, cls, conf, box in zip(ids, classes, confs, res.boxes.xyxy.tolist()):
                name = self.model.names[cls]
                x1, y1, x2, y2 = [int(v) for v in box]
                colour = COLOURS.get(name, (255, 255, 255))
                cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, 2)
                label = f"{name} #{tid}" if tid >= 0 else name
                cv2.putText(overlay, label, (x1, max(y1 - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
                objects.append({"id": tid, "class": name, "conf": round(conf, 3), "box": [x1, y1, x2, y2]})

        lane_text = f"L:{'Y' if found['left'] else '-'} R:{'Y' if found['right'] else '-'}"
        hud = f"frame {self.count} | objects {len(objects)} | lanes {lane_text}"
        cv2.putText(overlay, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        self.overlay_pub.publish(array_to_image(overlay, msg.header.stamp, msg.header.frame_id))
        payload = {
            "frame": self.count,
            "objects": objects,
            "lanes": found,
            "ms": {"lanes": round((t1 - t0) * 1000), "yolo_track": round((t2 - t1) * 1000)},
        }
        self.det_pub.publish(String(data=json.dumps(payload)))

        if self.count % 10 == 0:
            self.get_logger().info(
                f"frame {self.count}: lanes {payload['ms']['lanes']} ms, "
                f"yolo+track {payload['ms']['yolo_track']} ms, objects {len(objects)}")
        self.count += 1


def main():
    rclpy.init()
    node = PerceptionNode()
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