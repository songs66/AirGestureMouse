from dataclasses import dataclass

import pyautogui

from .geometry import clamp


@dataclass(frozen=True)
class ScreenSize:
    width: int
    height: int


class MouseController:
    def __init__(self, smoothing: float):
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0

        width, height = pyautogui.size()
        self.screen = ScreenSize(width=width, height=height)
        self.smoothing = smoothing
        self.previous_x = width / 2
        self.previous_y = height / 2

    def move_from_control_point(self, x: float, y: float, bounds) -> tuple[int, int]:
        target_x = (x - bounds.left) / bounds.width * self.screen.width
        target_y = (y - bounds.top) / bounds.height * self.screen.height

        target_x = clamp(target_x, 0, self.screen.width - 1)
        target_y = clamp(target_y, 0, self.screen.height - 1)

        current_x = self.previous_x + (target_x - self.previous_x) * self.smoothing
        current_y = self.previous_y + (target_y - self.previous_y) * self.smoothing

        pyautogui.moveTo(int(current_x), int(current_y))

        self.previous_x = current_x
        self.previous_y = current_y

        return int(current_x), int(current_y)

    def click(self, button: str) -> None:
        pyautogui.click(button=button)

    def scroll(self, amount: int) -> None:
        pyautogui.scroll(amount)
