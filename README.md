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

## Object Tracking (YOLO + ByteTrack)

### Tracking Results

The tracking pipeline was tested on a driving video containing multiple vehicles. It processed 1,537 frames and generated 6,857 tracking records with 73 unique track IDs.

During visual inspection, vehicles generally retained their assigned IDs while visible. No clear ID switches were observed during the inspection.

The tracker assigned non-consecutive IDs, such as 11, 25, 31 and 250. These numbers represent tracking identifiers, not the total number of vehicles.

A total of 18 tracks lasted fewer than five frames.

**Output files:**
- `tracked.mp4` – Annotated tracking video.
- `tracks.csv` – Tracking information for detected objects.
- `summary.json` – Tracking statistics.

## Lane detection: classical vs segmentation
Classical: OpenCV (white/yellow paint mask, Canny edges, Hough lines, smoothing). Both lanes
detected in 97.7% of 1,537 video frames (detection rate, not accuracy).
Segmentation: pretrained SegFormer-B0 (Cityscapes, nvidia/segformer-b0-finetuned-cityscapes-1024-1024),
used as-is. It predicts the road surface, not individual lane markings, so the methods answer related
but different questions. Compared on the same 20 frames: 99.8% of classical line pixels fall on
predicted road. This is a sanity check, not accuracy: any line on a wide road scores high, and no
ground-truth lane labels were used. Observed failure cases: [from observations.md].