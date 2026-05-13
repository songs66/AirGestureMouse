from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class ControlBounds:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def contains(self, x: int, y: int) -> bool:
        return self.left <= x <= self.right and self.top <= y <= self.bottom


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def normalized_distance(point_a, point_b) -> float:
    dx = point_a.x - point_b.x
    dy = point_a.y - point_b.y
    return sqrt(dx * dx + dy * dy)


def build_control_bounds(frame_width: int, frame_height: int, config) -> ControlBounds:
    left = int(clamp(config.margin_x, 0, frame_width - 1))
    right = int(clamp(frame_width - config.margin_x, 0, frame_width - 1))
    top = int(clamp(config.margin_y + config.offset_y, 0, frame_height - 1))
    bottom = int(clamp(frame_height - config.margin_y + config.offset_y, 0, frame_height - 1))

    if right <= left or bottom <= top:
        raise ValueError("控制区域参数错误，请调小 margin 或调整 offset。")

    return ControlBounds(left=left, top=top, right=right, bottom=bottom)
