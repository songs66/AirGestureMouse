from dataclasses import dataclass, field


@dataclass(frozen=True)
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720


@dataclass(frozen=True)
class ControlAreaConfig:
    margin_x: int = 220
    margin_y: int = 120
    offset_y: int = -80


@dataclass(frozen=True)
class WindowConfig:
    name: str = "AirGestureMouse"
    width: int = 640
    height: int = 360
    taskbar_offset: int = 80


@dataclass(frozen=True)
class GestureConfig:
    smoothing: float = 0.18
    left_pinch_trigger_ratio: float = 0.28
    left_pinch_release_ratio: float = 0.43
    right_pinch_trigger_ratio: float = 0.28
    right_pinch_release_ratio: float = 0.43
    pinch_confirm_frames: int = 4
    click_cooldown: float = 0.45
    click_freeze_time: float = 0.25
    scroll_trigger_delta: int = 10
    scroll_speed: int = 3
    scroll_cooldown: float = 0.04
    fist_confirm_frames: int = 7
    pause_toggle_cooldown: float = 0.8


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    control_area: ControlAreaConfig = field(default_factory=ControlAreaConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    mediapipe_detection_confidence: float = 0.7
    mediapipe_tracking_confidence: float = 0.7
    mediapipe_model_complexity: int = 1
