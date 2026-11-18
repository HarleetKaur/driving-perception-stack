"""Tests for the 3-class label conversion in src/prepare_data.py."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prepare_data import NEW_NAMES, convert_label_file  # noqa: E402

OLD_NAMES = {0: "car", 1: "van", 2: "truck", 3: "pedestrian",
             4: "person_sitting", 5: "cyclist", 6: "tram", 7: "misc"}


class TestConvertLabels(unittest.TestCase):
    def _convert(self, text):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.txt"
            p.write_text(text)
            return convert_label_file(p, OLD_NAMES)

    def test_classes_are_merged(self):
        text = "".join(f"{c} 0.5 0.5 0.1 0.1\n" for c in (0, 1, 3, 4, 5, 6))
        ids = [int(line.split()[0]) for line in self._convert(text)]
        self.assertEqual(ids, [0, 0, 1, 1, 2, 0])  # car, van, ped, sitting, cyclist, tram

    def test_misc_is_dropped(self):
        self.assertEqual(self._convert("7 0.5 0.5 0.1 0.1\n"), [])

    def test_malformed_lines_are_skipped(self):
        out = self._convert("0 0.5 0.5\nnot a label\n0 0.5 0.5 0.1 0.1\n")
        self.assertEqual(len(out), 1)

    def test_missing_file_returns_empty(self):
        self.assertEqual(convert_label_file(Path("/nonexistent/x.txt"), OLD_NAMES), [])

    def test_new_class_names(self):
        self.assertEqual(NEW_NAMES, ["vehicle", "pedestrian", "cyclist"])


if __name__ == "__main__":
    unittest.main()