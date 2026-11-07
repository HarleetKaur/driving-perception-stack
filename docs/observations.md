# Observations and Limitations

## Detection (YOLO)
- Test-set results: See `results/detection/test_results.json`.
- Weakest class: Refer to the test results JSON for the class with the lowest performance and its precision, recall, and mAP values.

## Tracking (ByteTrack), clip: `data/raw/videos/clip.mp4`
- ID switches: Vehicle IDs are assigned and maintained across consecutive frames in good cases. In crowded traffic, overlapping bounding boxes and nearby vehicles make tracking less reliable.
- Objects lost behind other objects: Vehicles moving close together or becoming partially hidden can make tracking difficult. Further frame-by-frame inspection is needed to confirm specific cases.
- False detections: The bad tracking example shows a low-confidence detection (ID 38, confidence 0.28) on a vehicle in the adjacent roadway. Some bounding boxes and labels also overlap in crowded traffic.
- Track statistics: See `results/tracking/summary.json`.

## Classical lane detection
- Detection rate (not accuracy): Both lanes detected in 97.7% of 1,537 frames.
- Failure cases: In the bad example, one lane marking is missing or incorrectly detected. The detector also struggles around intersections, turning lanes, and road markings.
- Known limitation: A straight-line model cannot accurately follow curved roads or turning lanes.