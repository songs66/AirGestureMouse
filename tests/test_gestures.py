from dataclasses import dataclass
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from air_gesture_mouse.gestures import calculate_pinch_ratios, is_fist, is_three_finger_scroll, read_finger_state


@dataclass
class Point:
    x: float
    y: float


def make_landmarks():
    landmarks = [Point(0.0, 0.5) for _ in range(21)]
    landmarks[4] = Point(0.2, 0.2)
    landmarks[5] = Point(0.1, 0.5)
    landmarks[17] = Point(0.5, 0.5)
    return landmarks


class GestureTests(unittest.TestCase):
    def test_detects_three_finger_scroll_pose(self):
        landmarks = make_landmarks()
        landmarks[6] = Point(0.1, 0.5)
        landmarks[8] = Point(0.1, 0.2)
        landmarks[10] = Point(0.2, 0.5)
        landmarks[12] = Point(0.2, 0.2)
        landmarks[14] = Point(0.3, 0.5)
        landmarks[16] = Point(0.3, 0.2)
        landmarks[18] = Point(0.4, 0.3)
        landmarks[20] = Point(0.4, 0.6)

        fingers = read_finger_state(landmarks)

        self.assertTrue(is_three_finger_scroll(fingers))
        self.assertFalse(is_fist(fingers))

    def test_calculates_pinch_ratios_relative_to_palm_width(self):
        landmarks = make_landmarks()
        landmarks[12] = Point(0.2, 0.2)
        landmarks[16] = Point(0.4, 0.2)

        ratios = calculate_pinch_ratios(landmarks)

        self.assertAlmostEqual(ratios.left, 0.0)
        self.assertAlmostEqual(ratios.right, 0.5)


if __name__ == "__main__":
    unittest.main()
