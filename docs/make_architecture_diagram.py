"""Draws the architecture diagram to docs/architecture.png.

Run from the project root:  python docs/make_architecture_diagram.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent / "architecture.png"
BLUE, GREEN, ORANGE, GREY = "#d6e6f5", "#d9efd9", "#fde4c8", "#eeeeee"

fig, ax = plt.subplots(figsize=(16, 8))
ax.set_xlim(0, 160)
ax.set_ylim(0, 80)
ax.axis("off")


def box(x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5",
                                fc=color, ec="#333333", lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)


def arrow(x1, y1, x2, y2, label=None, label_xy=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#333333"))
    if label:
        lx, ly = label_xy if label_xy else ((x1 + x2) / 2, max(y1, y2) + 2.5)
        ax.text(lx, ly, label, ha="center", va="bottom", fontsize=8, style="italic")


ax.text(80, 76, "Autonomous Driving Perception Stack: architecture",
        ha="center", fontsize=14, weight="bold")

box(2, 52, 18, 16, "clip.mp4\n(recorded\ndriving video)", GREY)
box(34, 52, 26, 16, "video_publisher\nevery 3rd frame\nresized to 960 px", BLUE)
box(82, 50, 30, 20, "perception_node\nYOLO11s + ByteTrack\n+ classical lane\ndetector (CPU)", GREEN)
box(134, 52, 24, 16, "overlay_recorder\nwrites\nros2_demo.mp4", BLUE)
arrow(20.5, 60, 33.5, 60)
arrow(60.5, 60, 81.5, 60, "/camera/image_raw")
arrow(112.5, 60, 133.5, 60, "/perception/overlay")

box(66, 30, 30, 12, "/perception/detections\nJSON: ids, classes,\nboxes, lane flags", ORANGE)
arrow(92, 49.5, 92, 42.5)

box(2, 30, 56, 14, "Offline analysis in Colab (T4 GPU):\nSegFormer lane comparison,\n"
    "DPT-Hybrid relative depth,\nONNX + TensorRT benchmark", GREY)

box(2, 6, 22, 16, "KITTI\n(Ultralytics\nauto-download)", GREY)
box(34, 6, 26, 16, "prepare_data.py\n3 classes\ntrain / val / test", BLUE)
box(70, 6, 34, 16, "Colab: fine-tune\nYOLO11n (baseline)\nvs YOLO11s (improved)\nlogged in W&B", GREEN)
box(114, 6, 30, 16, "models/\nimproved_best.pt", ORANGE)
arrow(24.5, 14, 33.5, 14)
arrow(60.5, 14, 69.5, 14)
arrow(104.5, 14, 113.5, 14)
arrow(124, 22.5, 108, 49.5, "weights", label_xy=(126, 35))

ax.text(80, 1.5, "Top: ROS2 demo in Docker (CPU; a recorded video replaces CARLA). "
        "Bottom: data preparation and training in Colab.", ha="center", fontsize=9)

fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("Saved", OUT)