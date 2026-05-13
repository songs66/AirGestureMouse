from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from air_gesture_mouse.config import ControlAreaConfig
from air_gesture_mouse.geometry import build_control_bounds, clamp


class GeometryTests(unittest.TestCase):
    def test_clamp_limits_value_to_range(self):
        self.assertEqual(clamp(-1, 0, 10), 0)
        self.assertEqual(clamp(11, 0, 10), 10)
        self.assertEqual(clamp(5, 0, 10), 5)

    def test_build_control_bounds_applies_margin_and_offset(self):
        bounds = build_control_bounds(
            frame_width=1280,
            frame_height=720,
            config=ControlAreaConfig(margin_x=220, margin_y=120, offset_y=-80),
        )

        self.assertEqual(bounds.left, 220)
        self.assertEqual(bounds.right, 1060)
        self.assertEqual(bounds.top, 40)
        self.assertEqual(bounds.bottom, 520)


if __name__ == "__main__":
    unittest.main()
