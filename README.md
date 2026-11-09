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


## Detection results (YOLO, KITTI, 3 classes)
Classes: vehicle (car, van, truck, tram), pedestrian (pedestrian, person_sitting), cyclist.
`misc` dropped. Splits: train 5,485 / val 500 (random slice of KITTI train) / test 1,496
(official KITTI val, never used for tuning).

| Class | Baseline YOLO11n (10 ep) mAP50 / recall | Improved YOLO11s (30 ep) mAP50 / recall |
|---|---|---|
| vehicle | 0.887 / 0.788 | 0.959 / 0.904 |
| pedestrian | 0.584 / 0.514 | 0.753 / 0.621 |
| cyclist | 0.493 / 0.440 | 0.780 / 0.702 |
| **all (mAP50)** | **0.655** | **0.831** |
| **all (mAP50-95)** | **0.374** | **0.556** |

The improved model changes two things at once (larger model and longer training), so the gain
cannot be attributed to either alone. Single run per model. Pedestrian recall is the weakest
point (0.621): about 4 in 10 pedestrians are missed. Training runs are logged in Weights & Biases.


## Object Tracking (YOLO + ByteTrack)

### Tracking Results

The tracking pipeline was tested on a driving video containing multiple vehicles. It processed 1,537 frames and generated 6,857 tracking records with 73 unique track IDs.

During visual inspection, vehicles generally retained their assigned IDs while visible. No clear ID switches were observed during the inspection.

The tracker assigned non-consecutive IDs, such as 11, 25, 31 and 250. These numbers represent tracking identifiers, not the total number of vehicles.

A total of 18 tracks lasted fewer than five frames. 

Tracking run: Improved YOLO11s weights, imgsz=640, conf=0.25, using ByteTrack for object tracking.

 Observed issues: In crowded traffic, overlapping bounding boxes and labels make vehicle tracking less reliable. Vehicles moving close together or becoming partially hidden can be lost behind other objects. The bad tracking example also shows a low-confidence detection (ID 38, confidence 0.28) on a vehicle in the adjacent roadway.

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
ground-truth lane labels were used.
Observed failure cases: In some frames, the segmentation mask includes parts of the vehicle hood and nearby roadside areas, incorrectly classifying them as road. The classical lane detection method also struggles with curved turn lanes because straight-line detection cannot follow curves accurately.

## Depth and distance
Monocular depth: Intel/dpt-hybrid-midas (MiDaS-family, zero-shot) combined with YOLO boxes.
Output is RELATIVE distance (NEAR/MID/FAR, arbitrary cut-offs), not metres: the camera is
uncalibrated, so metric distance was not measured.
Detector settings: the first run at imgsz=640, conf=0.25 drew no boxes on the three sample frames
I inspected; rerunning at imgsz=1280 gave 129 detections over the 20 sample frames. The tracking
run at 640 produced detections over the full video, so the 640 vs 1280 difference was not
quantified. This clip is wide-angle and differs from KITTI (domain gap).
Cross-check on 20 frames (129 objects, all frames with 3+ objects): mean Spearman correlation
between depth ordering and box-bottom position = 0.92 (median 0.96). This shows the depth
ordering is consistent with a flat-road geometric cue; it is not a measurement of distance
accuracy, and monocular depth models learn the same cue, so the agreement is partly expected.
Small sample, single clip. Failure case: dense distant traffic produces overlapping tiny boxes
with unreliable labels.