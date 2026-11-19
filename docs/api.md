# Data and API documentation

## ROS2 nodes (src/ros2_nodes, run inside Docker, ROS2 Humble)
| Node | Subscribes | Publishes | Parameters (default) |
|---|---|---|---|
| video_publisher | none | /camera/image_raw | video (/ws/data/raw/videos/clip.mp4), rate_hz (2.0), stride (3), width (960), max_frames (0 = all) |
| perception_node | /camera/image_raw | /perception/overlay, /perception/detections | weights (models/improved_best.pt), imgsz (640), conf (0.25), roi_bottom_y (0.70) |
| overlay_recorder | /perception/overlay | none (writes a file) | output (/ws/results/ros2/ros2_demo.mp4), fps (10.0), idle_timeout_s (20.0) |

## Topics and message formats
- `/camera/image_raw` and `/perception/overlay`: `sensor_msgs/Image`, encoding `bgr8`, 960 px wide.
- `/perception/detections`: `std_msgs/String` containing JSON:
```json
{
  "frame": 12,
  "objects": [
    {"id": 3, "class": "vehicle", "conf": 0.87, "box": [412, 288, 470, 330]}
  ],
  "lanes": {"left": true, "right": false},
  "ms": {"lanes": 25, "yolo_track": 180}
}
```
`id` is the ByteTrack track ID (-1 if the object is not yet tracked). `box` is x1, y1, x2, y2 in pixels
of the 960 px wide frame. `lanes` says whether each lane line was detected in this frame. `ms` is the
processing time per stage. (The values above only show the format; they are not measured results.)

## Error handling
Images with the wrong encoding, an empty size, or a wrong data length are skipped with a warning
(`ros_utils.image_to_array`). Missing weights or video files stop the node at startup with a clear error.

## Files and formats
| File | Format |
|---|---|
| data/processed/kitti3/images and labels/{train,val,test} | YOLO labels: `class x_center y_center width height` (0-1 scale); classes 0 vehicle, 1 pedestrian, 2 cyclist |
| results/detection/test_results.json | per-model and per-class precision, recall, mAP50, mAP50-95 on the test split |
| results/tracking/tracks.csv | frame, track_id, class, conf, x1, y1, x2, y2 |
| results/tracking/summary.json | frames, unique tracks, per-class track counts and lengths |
| results/lanes/classical_summary.json | frames, both-lanes and at-least-one-lane detection rates, region of interest used |
| results/lanes/comparison/segmentation_summary.json | per-frame share of classical line pixels on predicted road |
| results/depth/depth_summary.json | detection settings, objects, Spearman correlation of depth ordering |
| results/benchmark/benchmark_results.csv | format, GPU, latency (ms), FPS, end-to-end ms, test mAP |
| results/ros2/ros2_demo.mp4 | annotated demo video (not stored in git; in the Drive submission folder) |