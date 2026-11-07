"""Classical lane detection with OpenCV.

Pipeline: white/yellow paint mask + Canny edges + region mask + Hough lines,
then length-weighted averaging and geometry checks.

Usage:
    # 1) look at the region of interest first
    python src/lanes/classical.py --video data/raw/videos/clip.mp4 --out results/lanes --debug-roi
    # 2) run the whole video
    python src/lanes/classical.py --video data/raw/videos/clip.mp4 --out results/lanes
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# Region of interest as fractions of image width/height. Tune with --debug-roi.
# bottom_y must stay ABOVE the car hood.
DEFAULT_ROI = {
    "top_y": 0.58, "bottom_y": 0.78,
    "top_left": 0.44, "top_right": 0.56,
    "bottom_left": 0.25, "bottom_right": 0.82,
}


class LaneDetector:
    """Finds the left and right lane lines in one frame."""

    MIN_SLOPE, MAX_SLOPE = 0.5, 3.0  # ignore near-flat and near-vertical segments

    def __init__(self, roi=None, smooth: float = 0.8):
        self.roi = dict(DEFAULT_ROI)
        if roi:
            self.roi.update(roi)
        self.smooth = smooth  # 0 = no smoothing, closer to 1 = smoother
        self.prev = {"left": None, "right": None}
        self.misses = {"left": 0, "right": 0}

    # ---------- image processing ----------
    def _roi_polygon(self, h, w):
        r = self.roi
        return np.array([
            (int(r["bottom_left"] * w), int(r["bottom_y"] * h)),
            (int(r["top_left"] * w), int(r["top_y"] * h)),
            (int(r["top_right"] * w), int(r["top_y"] * h)),
            (int(r["bottom_right"] * w), int(r["bottom_y"] * h)),
        ], dtype=np.int32)

    @staticmethod
    def _paint_mask(frame):
        """Pixels that look like white or yellow road paint."""
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)  # order: hue, lightness, saturation
        white = cv2.inRange(hls, (0, 140, 0), (255, 255, 130))
        yellow = cv2.inRange(hls, (12, 80, 90), (35, 255, 255))
        return cv2.bitwise_or(white, yellow)

    def masked_edges(self, frame):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        paint = cv2.dilate(self._paint_mask(frame), np.ones((7, 7), np.uint8))
        edges = cv2.bitwise_and(edges, paint)
        roi_mask = np.zeros_like(edges)
        cv2.fillPoly(roi_mask, [self._roi_polygon(h, w)], 255)
        return cv2.bitwise_and(edges, roi_mask)

    # ---------- geometry ----------
    def _split(self, lines, w):
        """Group segments into left/right by slope (image y points down)."""
        left, right = [], []
        for x1, y1, x2, y2 in lines:
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if not (self.MIN_SLOPE <= abs(slope) <= self.MAX_SLOPE):
                continue
            intercept = y1 - slope * x1
            length = float(np.hypot(x2 - x1, y2 - y1))
            mid_x = (x1 + x2) / 2
            if slope < 0 and mid_x < 0.55 * w:
                left.append((slope, intercept, length))
            elif slope > 0 and mid_x > 0.45 * w:
                right.append((slope, intercept, length))
        return left, right

    @staticmethod
    def _fit(group):
        """Length-weighted average line. Returns (params, n_segments) or None."""
        if not group:
            return None
        arr = np.array(group)
        weights = arr[:, 2]
        params = np.array([np.average(arr[:, 0], weights=weights),
                           np.average(arr[:, 1], weights=weights)])
        return params, len(group)

    @staticmethod
    def _x_at(params, y):
        slope, intercept = params
        return (y - intercept) / slope

    # ---------- main entry ----------
    def detect(self, frame):
        """Return (overlay_image, {'left': bool, 'right': bool})."""
        h, w = frame.shape[:2]
        segs = cv2.HoughLinesP(self.masked_edges(frame), 1, np.pi / 180, threshold=20,
                               minLineLength=25, maxLineGap=50)
        lines = segs.reshape(-1, 4).tolist() if segs is not None else []
        left, right = self._split(lines, w)
        fits = {"left": self._fit(left), "right": self._fit(right)}

        y_top, y_bot = int(self.roi["top_y"] * h), int(self.roi["bottom_y"] * h)

        # Geometry check: lines must not cross and must not be on top of each other.
        if fits["left"] and fits["right"]:
            lp, rp = fits["left"][0], fits["right"][0]
            crossed = (self._x_at(lp, y_top) >= self._x_at(rp, y_top)
                       or self._x_at(lp, y_bot) >= self._x_at(rp, y_bot))
            too_close = (self._x_at(rp, y_bot) - self._x_at(lp, y_bot)) < 0.10 * w
            if crossed or too_close:  # keep the side with more supporting segments
                if fits["left"][1] >= fits["right"][1]:
                    fits["right"] = None
                else:
                    fits["left"] = None

        
        overlay = frame.copy()
        found = {"left": False, "right": False}
        HOLD_FRAMES = 12  # keep showing the last line for this many missed frames
        for side in ("left", "right"):
            fit = fits[side]
            colour = (0, 255, 0)  # green = detected in this frame
            if fit is None:
                self.misses[side] += 1
                if self.prev[side] is None or self.misses[side] > HOLD_FRAMES:
                    self.prev[side] = None
                    continue
                params = self.prev[side]  # held from earlier frames
                colour = (0, 200, 255)    # orange = held, NOT detected in this frame
            else:
                self.misses[side] = 0
                params = fit[0]
                if self.prev[side] is not None:
                    params = self.smooth * self.prev[side] + (1 - self.smooth) * params
                self.prev[side] = params
                found[side] = True
            p1 = (int(self._x_at(params, y_bot)), y_bot)
            p2 = (int(self._x_at(params, y_top)), y_top)
            cv2.line(overlay, p1, p2, colour, 6)
        return cv2.addWeighted(frame, 0.4, overlay, 0.6, 0), found

    def debug_image(self, frame):
        """Frame with the region outlined (left) next to what the detector sees (right)."""
        h, w = frame.shape[:2]
        shown = frame.copy()
        cv2.polylines(shown, [self._roi_polygon(h, w)], True, (0, 255, 255), 4)
        edges = cv2.cvtColor(self.masked_edges(frame), cv2.COLOR_GRAY2BGR)
        return np.hstack([shown, edges])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--out", default="results/lanes", type=Path)
    parser.add_argument("--n-frames", default=20, type=int, help="frames to save for comparison")
    parser.add_argument("--debug-roi", action="store_true",
                        help="save images showing the region of interest, then stop")
    for key, default in DEFAULT_ROI.items():
        parser.add_argument(f"--roi-{key.replace('_', '-')}", type=float, default=default)
    args = parser.parse_args()

    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "frames").mkdir(parents=True, exist_ok=True)
    roi = {key: getattr(args, f"roi_{key}") for key in DEFAULT_ROI}

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detector = LaneDetector(roi=roi)

    if args.debug_roi:
        for frac in (0.0, 0.25, 0.5, 0.75):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frac * max(total - 1, 0)))
            ok, frame = cap.read()
            if ok:
                path = args.out / f"roi_debug_{int(frac * 100)}.jpg"
                cv2.imwrite(str(path), detector.debug_image(frame))
                print(f"Saved {path}")
        cap.release()
        return

    writer = cv2.VideoWriter(str(args.out / "classical_lanes.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    keep = set(np.linspace(0, max(total - 1, 0), args.n_frames, dtype=int).tolist())
    idx, both, any_lane = 0, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        overlay, found = detector.detect(frame)
        writer.write(overlay)
        both += found["left"] and found["right"]
        any_lane += found["left"] or found["right"]
        if idx in keep:
            cv2.imwrite(str(args.out / "frames" / f"frame_{idx:05d}_original.jpg"), frame)
            cv2.imwrite(str(args.out / "frames" / f"frame_{idx:05d}_classical.jpg"), overlay)
        idx += 1
    cap.release()
    writer.release()

    summary = {
        "frames": idx,
        "both_lanes_found_pct": round(100 * both / max(idx, 1), 1),
        "at_least_one_lane_found_pct": round(100 * any_lane / max(idx, 1), 1),
        "roi_used": roi,
        "note": "Detection rate only. This is NOT accuracy: no ground-truth lane labels used.",
    }
    (args.out / "classical_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nSaved to {args.out}/")


if __name__ == "__main__":
    main()