"""Detect and track objects in a video with YOLO + ByteTrack.

Usage:
    python src/tracking/track_video.py \
        --weights models/improved_best.pt \
        --video data/raw/videos/clip.mp4 \
        --out results/tracking
"""
import argparse
import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


def pick_device():
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--out", default="results/tracking", type=Path)
    parser.add_argument("--conf", default=0.25, type=float)
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--max-frames", default=0, type=int, help="0 = whole video")
    args = parser.parse_args()

    for path, label in ((args.weights, "weights"), (args.video, "video")):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    args.out.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    print(f"Using device: {device}")
    model = YOLO(str(args.weights))

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(args.out / "tracked.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    rows, frames_per_track, track_class = [], defaultdict(int), {}
    frame_idx, start = 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and frame_idx >= args.max_frames):
            break
        result = model.track(
            frame, persist=True, tracker="bytetrack.yaml",
            conf=args.conf, imgsz=args.imgsz, device=device, verbose=False,
        )[0]
        writer.write(result.plot())
        if result.boxes is not None and result.boxes.id is not None:
            ids = result.boxes.id.int().tolist()
            classes = result.boxes.cls.int().tolist()
            confs = result.boxes.conf.tolist()
            boxes = result.boxes.xyxy.tolist()
            for tid, cls, conf, box in zip(ids, classes, confs, boxes):
                name = model.names[cls]
                rows.append([frame_idx, tid, name, round(conf, 3)] + [round(v, 1) for v in box])
                frames_per_track[tid] += 1
                track_class[tid] = name
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  processed {frame_idx} frames")
    cap.release()
    writer.release()
    elapsed = time.time() - start

    with open(args.out / "tracks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "track_id", "class", "conf", "x1", "y1", "x2", "y2"])
        w.writerows(rows)

    summary = {
        "video": str(args.video), "frames": frame_idx, "device": str(device),
        "processing_fps_including_drawing_and_writing": round(frame_idx / elapsed, 1),
        "unique_tracks": len(frames_per_track), "per_class": {},
    }
    for cls in sorted(set(track_class.values())):
        lengths = [frames_per_track[t] for t, c in track_class.items() if c == cls]
        summary["per_class"][cls] = {
            "tracks": len(lengths),
            "median_track_length_frames": statistics.median(lengths),
            "short_tracks_under_5_frames": sum(1 for n in lengths if n < 5),
        }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nSaved video, tracks.csv and summary.json to {args.out}/")


if __name__ == "__main__":
    main()