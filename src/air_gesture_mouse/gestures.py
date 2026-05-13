from dataclasses import dataclass

from .geometry import normalized_distance


@dataclass(frozen=True)
class FingerState:
    index_extended: bool
    middle_extended: bool
    ring_extended: bool
    pinky_extended: bool
    index_folded: bool
    middle_folded: bool
    ring_folded: bool
    pinky_folded: bool


@dataclass(frozen=True)
class PinchRatios:
    left: float
    right: float


def is_finger_extended(landmarks, tip_id: int, pip_id: int, margin: float = 0.02) -> bool:
    return landmarks[tip_id].y < landmarks[pip_id].y - margin


def is_finger_folded(landmarks, tip_id: int, pip_id: int, margin: float = 0.02) -> bool:
    return landmarks[tip_id].y > landmarks[pip_id].y + margin


def read_finger_state(landmarks) -> FingerState:
    return FingerState(
        index_extended=is_finger_extended(landmarks, 8, 6),
        middle_extended=is_finger_extended(landmarks, 12, 10),
        ring_extended=is_finger_extended(landmarks, 16, 14),
        pinky_extended=is_finger_extended(landmarks, 20, 18),
        index_folded=is_finger_folded(landmarks, 8, 6),
        middle_folded=is_finger_folded(landmarks, 12, 10),
        ring_folded=is_finger_folded(landmarks, 16, 14),
        pinky_folded=is_finger_folded(landmarks, 20, 18),
    )


def is_fist(fingers: FingerState) -> bool:
    return (
        fingers.index_folded
        and fingers.middle_folded
        and fingers.ring_folded
        and fingers.pinky_folded
    )


def is_three_finger_scroll(fingers: FingerState) -> bool:
    return (
        fingers.index_extended
        and fingers.middle_extended
        and fingers.ring_extended
        and fingers.pinky_folded
    )


def calculate_pinch_ratios(landmarks) -> PinchRatios:
    palm_width = normalized_distance(landmarks[5], landmarks[17])

    if palm_width < 0.0001:
        return PinchRatios(left=999, right=999)

    return PinchRatios(
        left=normalized_distance(landmarks[4], landmarks[12]) / palm_width,
        right=normalized_distance(landmarks[4], landmarks[16]) / palm_width,
    )
