"""EDA for the Ultralytics KITTI detection dataset.

Downloads KITTI automatically on first run (about 390 MB), then counts
objects per class and checks object sizes.

Usage:
    python src/eda_kitti.py --out results/eda
"""
import argparse
from collections import Counter
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # save charts to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics.data.utils import check_det_dataset, img2label_paths

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def list_images(folder) -> list:
    """Return all image paths in a folder (the config may give a list)."""
    if isinstance(folder, (list, tuple)):
        folder = folder[0]
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Image folder not found: {folder}")
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def read_boxes(label_path: Path) -> list:
    """Read one YOLO label file: each line is 'class x y w h' (0-1 scale)."""
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue  # skip malformed lines
        boxes.append((int(parts[0]), *map(float, parts[1:])))
    return boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/eda", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    info = check_det_dataset("kitti.yaml")  # downloads if needed
    names = info["names"]  # {0: 'car', 1: 'van', ...}

    rows, summary = [], []
    for split in ("train", "val"):
        images = list_images(info[split])
        if not images:
            raise ValueError(f"No images found for split '{split}'.")
        labels = img2label_paths([str(p) for p in images])
        sample = cv2.imread(str(images[0]))
        img_h, img_w = sample.shape[:2]

        counts, empty_images = Counter(), 0
        for lab in labels:
            boxes = read_boxes(Path(lab))
            if not boxes:
                empty_images += 1
            for cls, _x, _y, w, h in boxes:
                counts[names[cls]] += 1
                rows.append(
                    {"split": split, "class": names[cls], "area_px": w * img_w * h * img_h}
                )
        summary.append(
            f"{split}: {len(images)} images, {sum(counts.values())} boxes, "
            f"{empty_images} images with no objects, image size {img_w}x{img_h}"
        )

    df = pd.DataFrame(rows)
    table = df.groupby(["class", "split"]).size().unstack(fill_value=0)
    table["total"] = table.sum(axis=1)
    table = table.sort_values("total", ascending=False)
    table.to_csv(args.out / "class_counts.csv")

    ax = table[["train", "val"]].plot(kind="bar", figsize=(9, 5))
    ax.set_title("KITTI: objects per class")
    ax.set_ylabel("count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(args.out / "class_counts.png", dpi=150)
    plt.close()

    small = (df["area_px"] < 32 * 32).mean() * 100
    top = table.index[0]
    lines = summary + [
        f"Small objects (<32x32 px): {small:.1f}%",
        f"Most common class: {top} ({table.loc[top, 'total'] / table['total'].sum() * 100:.1f}%)",
        f"Rarest class: {table.index[-1]} ({table['total'].iloc[-1]} boxes)",
        "",
        "Class counts:",
        table.to_string(),
    ]
    text = "\n".join(lines)
    (args.out / "summary.txt").write_text(text)
    print(text)
    print(f"\nSaved to {args.out}/")


if __name__ == "__main__":
    main()