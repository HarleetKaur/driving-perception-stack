"""Checks the prepared 3-class dataset (skipped if data/processed/kitti3 is missing)."""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "processed" / "kitti3"
EXTS = {".png", ".jpg", ".jpeg"}


@unittest.skipUnless(ROOT.exists(), "prepared dataset not found")
class TestDataset(unittest.TestCase):
    def _stems(self, split):
        return {p.stem for p in (ROOT / "images" / split).iterdir() if p.suffix.lower() in EXTS}

    def test_splits_do_not_overlap(self):
        train, val, test = (self._stems(s) for s in ("train", "val", "test"))
        self.assertFalse(train & val)
        self.assertFalse(train & test)
        self.assertFalse(val & test)

    def test_every_image_has_a_label_file(self):
        for split in ("train", "val", "test"):
            labels = {p.stem for p in (ROOT / "labels" / split).glob("*.txt")}
            self.assertEqual(self._stems(split) - labels, set(), f"missing labels in {split}")

    def test_label_class_ids_are_valid(self):
        for split in ("train", "val", "test"):
            for p in (ROOT / "labels" / split).glob("*.txt"):
                for line in p.read_text().splitlines():
                    self.assertIn(int(line.split()[0]), (0, 1, 2), f"bad class id in {p}")


if __name__ == "__main__":
    unittest.main()