# Autonomous Driving Perception Stack

A vision pipeline that detects lanes, vehicles, and pedestrians, tracks them,
estimates distance, and runs on a camera feed through ROS2.

**Status:** In progress (started 30 Oct 2026)

## Planned stack
YOLO, OpenCV, ByteTrack, MiDaS, ROS2, ONNX, TensorRT, Weights & Biases

## Dataset
KITTI (Karlsruhe Institute of Technology / Toyota Technological Institute), loaded via Ultralytics.

BDD100K's official portal was unavailable on 2 Nov 2026 and its label files were not obtained in time, so KITTI (listed in the assignment) was used for training.

### Class imbalance (EDA finding)

KITTI is heavily imbalanced: car is about 71% of all boxes, while person_sitting is about 0.5%.

Per-class metrics are therefore reported, not only an overall score.

## Setup
[To be completed]