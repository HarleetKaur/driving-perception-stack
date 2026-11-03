"""Build a 3-class KITTI dataset (vehicle, pedestrian, cyclist) with splits.

Splits:
  train = official KITTI train, minus a held-out validation slice
  val   = random slice of official train (used for tuning)
  test  = official KITTI val (never used for tuning)

Usage:
    python src/prepare_data.py --out data/processed/kitti3
"""
import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

from ultralytics.data.utils import check_det_dataset, img2label_paths

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
NEW_NAMES = ["vehicle", "pedestrian", "cyclist"]
# 'misc' is intentionally absent: it is dropped.
MERGE = {
    "car": "vehicle",
    "van": "vehicle",
    "truck": "vehicle",
    "tram": "vehicle",
    "pedestrian": "pedestrian",
    "person_sitting": "pedestrian",
    "cyclist": "cyclist",
}


def list_images(folder) -> list:
    if isinstance(folder, (list, tuple)):
        folder = folder[0]
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Image folder not found: {folder}")
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def convert_label_file(label_path: Path, old_names: dict) -> list:
    """Return the label lines rewritten with the new 3-class ids."""
    if not label_path.exists():
        return []
    out = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue  # skip malformed lines
        new_name = MERGE.get(old_names[int(parts[0])])
        if new_name is None:
            continue  # dropped class (misc)
        out.append(f"{NEW_NAMES.index(new_name)} " + " ".join(parts[1:]))
    return out


def write_split(images: list, split: str, out: Path, old_names: dict) -> Counter:
    img_dir = out / "images" / split
    lab_dir = out / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    for img in images:
        shutil.copy2(img, img_dir / img.name)
        src_label = Path(img2label_paths([str(img)])[0])
        lines = convert_label_file(src_label, old_names)
        (lab_dir / f"{img.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        for line in lines:
            counts[NEW_NAMES[int(line.split()[0])]] += 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/processed/kitti3", type=Path)
    parser.add_argument("--val-size", default=500, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"{args.out} is not empty. Delete it first to rebuild.")

    info = check_det_dataset("kitti.yaml")
    old_names = info["names"]
    official_train = list_images(info["train"])
    test_imgs = list_images(info["val"])

    rng = random.Random(args.seed)
    shuffled = official_train[:]
    rng.shuffle(shuffled)
    val_imgs = shuffled[: args.val_size]
    train_imgs = shuffled[args.val_size :]

    # Safety check: no image may appear in two splits.
    sets = [{p.stem for p in s} for s in (train_imgs, val_imgs, test_imgs)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise RuntimeError("Splits overlap. Data leakage risk, stopping.")

    results = {}
    for split, imgs in (("train", train_imgs), ("val", val_imgs), ("test", test_imgs)):
        results[split] = (len(imgs), write_split(imgs, split, args.out, old_names))

    yaml_text = "train: images/train\nval: images/val\ntest: images/test\nnames:\n"
    yaml_text += "".join(f"  {i}: {n}\n" for i, n in enumerate(NEW_NAMES))
    (args.out / "kitti3.yaml").write_text(yaml_text)

    print(f"{'split':<8}{'images':>8}" + "".join(f"{n:>13}" for n in NEW_NAMES))
    for split, (n_img, counts) in results.items():
        print(f"{split:<8}{n_img:>8}" + "".join(f"{counts[n]:>13}" for n in NEW_NAMES))
    print(f"\nDataset written to {args.out}/ (config: kitti3.yaml)")


if __name__ == "__main__":
    main()