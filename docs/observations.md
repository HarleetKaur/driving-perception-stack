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

## Lane comparison

- **Blue region covers the road:** The segmentation model successfully identifies most of the road surface ahead of the vehicle, especially in relatively straight road sections.

- **Paints the hood:** In some frames, the blue segmentation region extends onto the front hood of the vehicle instead of covering only the road.

- **Where both methods agree:** On relatively straight roads, the classical green lane lines generally fall within the blue road region predicted by the segmentation model.

- **Where they disagree:** Around curves and intersections, classical lane detection follows visible lane markings, while segmentation predicts a broader road surface. The two methods therefore do not always match.

- **Problem frame:** In some frames, the segmentation mask includes parts of the vehicle hood and nearby roadside areas, showing that the model does not perfectly separate the drivable road from surrounding regions.

## Depth and Distance Observations
- Depth estimation identifies nearby vehicles as NEAR and distant vehicles as FAR, helping distinguish objects based on their relative distance.
- The detector successfully identifies the nearby black SUV as NEAR and farther vehicles as MID or FAR.
- The detector required imgsz=1280 to detect objects properly in this wide-angle cl