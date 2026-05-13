from dataclasses import dataclass
import time

import cv2
import mediapipe as mp

from .config import AppConfig
from .geometry import build_control_bounds, clamp
from .gestures import (
    calculate_pinch_ratios,
    is_fist,
    is_three_finger_scroll,
    read_finger_state,
)
from .mouse import MouseController


@dataclass
class RuntimeState:
    left_pinch_frames: int = 0
    right_pinch_frames: int = 0
    left_pinch_active: bool = False
    right_pinch_active: bool = False
    last_click_time: float = 0
    mouse_frozen_until: float = 0
    last_scroll_y: float | None = None
    last_scroll_time: float = 0
    paused: bool = False
    fist_frames: int = 0
    fist_active: bool = False
    last_pause_toggle_time: float = 0

    def reset_click_candidates(self) -> None:
        self.left_pinch_frames = 0
        self.right_pinch_frames = 0

    def reset_hand_tracking(self) -> None:
        self.left_pinch_frames = 0
        self.right_pinch_frames = 0
        self.left_pinch_active = False
        self.right_pinch_active = False
        self.last_scroll_y = None
        self.fist_frames = 0
        self.fist_active = False


def draw_text(frame, text: str, position: tuple[int, int], color=(255, 255, 255), scale=0.72, thickness=2) -> None:
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def keep_window_topmost(window_name: str) -> None:
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass


class GestureMouseApp:
    def __init__(self, config: AppConfig | None = None):
        self.config = config or AppConfig()
        self.state = RuntimeState()
        self.mouse = MouseController(self.config.gesture.smoothing)
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

    def run(self) -> None:
        camera = self.config.camera
        window = self.config.window

        cap = cv2.VideoCapture(camera.index)
        if not cap.isOpened():
            print("无法打开摄像头，请检查摄像头权限或 camera.index 配置。")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera.height)

        hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=self.config.mediapipe_model_complexity,
            min_detection_confidence=self.config.mediapipe_detection_confidence,
            min_tracking_confidence=self.config.mediapipe_tracking_confidence,
        )

        self._create_window()
        self._print_banner()

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("无法读取摄像头画面。")
                    break

                frame = cv2.flip(frame, 1)
                self._process_frame(frame, hands)

                cv2.imshow(window.name, frame)
                keep_window_topmost(window.name)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            hands.close()
            cap.release()
            cv2.destroyAllWindows()

    def _create_window(self) -> None:
        window = self.config.window
        screen = self.mouse.screen
        window_y = screen.height - window.height - window.taskbar_offset

        cv2.namedWindow(window.name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window.name, window.width, window.height)
        cv2.moveWindow(window.name, 0, window_y)
        keep_window_topmost(window.name)

    def _print_banner(self) -> None:
        print(f"屏幕尺寸: {self.mouse.screen.width} x {self.mouse.screen.height}")
        print("AirGestureMouse 启动成功")
        print("食指移动: 鼠标移动")
        print("拇指 + 中指捏合: 左键点击")
        print("拇指 + 无名指捏合: 右键点击")
        print("食指 + 中指 + 无名指上下移动: 滚轮")
        print("握拳: 暂停 / 恢复控制")
        print("按 q 退出程序")

    def _process_frame(self, frame, hands) -> None:
        frame_height, frame_width, _ = frame.shape
        bounds = build_control_bounds(frame_width, frame_height, self.config.control_area)

        cv2.rectangle(frame, (bounds.left, bounds.top), (bounds.right, bounds.bottom), (255, 0, 255), 2)
        draw_text(frame, "Control Area", (bounds.left, max(bounds.top - 10, 30)), (255, 0, 255))

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        if not results.multi_hand_landmarks:
            self.state.reset_hand_tracking()
            self._draw_no_hand_overlay(frame)
            return

        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = hand_landmarks.landmark
        self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        self._handle_landmarks(frame, landmarks, bounds, frame_width, frame_height)

    def _draw_no_hand_overlay(self, frame) -> None:
        pause_text = "Paused: ON" if self.state.paused else "Paused: OFF"
        draw_text(frame, "No Hand Detected", (20, 40), (0, 0, 255))
        draw_text(frame, pause_text, (20, 80), (0, 255, 255) if self.state.paused else (255, 255, 255))

    def _handle_landmarks(self, frame, landmarks, bounds, frame_width: int, frame_height: int) -> None:
        state = self.state
        gesture = self.config.gesture
        now = time.time()

        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]

        index_x = int(index_tip.x * frame_width)
        index_y = int(index_tip.y * frame_height)
        middle_x = int(middle_tip.x * frame_width)
        middle_y = int(middle_tip.y * frame_height)
        ring_x = int(ring_tip.x * frame_width)
        ring_y = int(ring_tip.y * frame_height)
        thumb_x = int(thumb_tip.x * frame_width)
        thumb_y = int(thumb_tip.y * frame_height)

        index_in_area = bounds.contains(index_x, index_y)
        mapped_x = clamp(index_x, bounds.left, bounds.right)
        mapped_y = clamp(index_y, bounds.top, bounds.bottom)
        mapped_draw_x = int(mapped_x)
        mapped_draw_y = int(mapped_y)

        self._draw_landmark_helpers(
            frame,
            index=(index_x, index_y),
            middle=(middle_x, middle_y),
            ring=(ring_x, ring_y),
            thumb=(thumb_x, thumb_y),
            mapped=(mapped_draw_x, mapped_draw_y),
            index_in_area=index_in_area,
        )

        fingers = read_finger_state(landmarks)
        fist_candidate = is_fist(fingers)
        scroll_candidate = is_three_finger_scroll(fingers)
        pinch_ratios = calculate_pinch_ratios(landmarks)

        left_pinch_candidate = (
            not scroll_candidate and pinch_ratios.left < gesture.left_pinch_trigger_ratio
        )
        right_pinch_candidate = (
            not scroll_candidate and pinch_ratios.right < gesture.right_pinch_trigger_ratio
        )
        click_candidate = left_pinch_candidate or right_pinch_candidate

        self._update_pause_state(fist_candidate, now)

        status_text = "Gesture Control Paused"
        status_color = (0, 255, 255)
        mode_text = "Mode: Paused"

        if not state.paused:
            status_text, status_color, mode_text = self._handle_active_control(
                mapped_x=mapped_x,
                mapped_y=mapped_y,
                index_in_area=index_in_area,
                scroll_candidate=scroll_candidate,
                click_candidate=click_candidate,
                fist_candidate=fist_candidate,
                left_pinch_candidate=left_pinch_candidate,
                right_pinch_candidate=right_pinch_candidate,
                pinch_ratios=pinch_ratios,
                scroll_y=(index_y + middle_y + ring_y) / 3,
                now=now,
                bounds=bounds,
            )

        self._draw_status_overlay(
            frame,
            index=(index_x, index_y),
            mapped=(mapped_draw_x, mapped_draw_y),
            status_text=status_text,
            status_color=status_color,
            mode_text=mode_text,
            pinch_ratios=pinch_ratios,
            fingers=fingers,
        )

    def _update_pause_state(self, fist_candidate: bool, now: float) -> None:
        state = self.state
        gesture = self.config.gesture

        if fist_candidate:
            state.fist_frames += 1
            can_toggle = (
                state.fist_frames >= gesture.fist_confirm_frames
                and not state.fist_active
                and now - state.last_pause_toggle_time > gesture.pause_toggle_cooldown
            )
            if can_toggle:
                state.paused = not state.paused
                state.fist_active = True
                state.last_pause_toggle_time = now
                state.left_pinch_frames = 0
                state.right_pinch_frames = 0
                state.left_pinch_active = False
                state.right_pinch_active = False
                state.last_scroll_y = None
                print("已暂停手势控制" if state.paused else "已恢复手势控制")
        else:
            state.fist_frames = 0
            state.fist_active = False

    def _handle_active_control(
        self,
        *,
        mapped_x: float,
        mapped_y: float,
        index_in_area: bool,
        scroll_candidate: bool,
        click_candidate: bool,
        fist_candidate: bool,
        left_pinch_candidate: bool,
        right_pinch_candidate: bool,
        pinch_ratios,
        scroll_y: float,
        now: float,
        bounds,
    ) -> tuple[str, tuple[int, int, int], str]:
        state = self.state
        gesture = self.config.gesture

        scroll_mode = scroll_candidate and not click_candidate and not fist_candidate
        if scroll_mode:
            self._handle_scroll(scroll_y, now)
            state.left_pinch_frames = 0
            state.right_pinch_frames = 0
            state.left_pinch_active = False
            state.right_pinch_active = False
            return "Three-Finger Scroll", (255, 255, 0), "Mode: Three-Finger Scroll"

        state.last_scroll_y = None
        mouse_frozen = now < state.mouse_frozen_until or click_candidate

        if not mouse_frozen:
            self.mouse.move_from_control_point(mapped_x, mapped_y, bounds)
            status = "Mouse Moving" if index_in_area else "Mouse Edge Moving"
            mode = "Mode: Move" if index_in_area else "Mode: Edge Clamp"
            status_color = (0, 255, 0)
        else:
            status = "Mouse Frozen For Click"
            mode = "Mode: Click Candidate"
            status_color = (0, 255, 255)

        ambiguous_pinch = left_pinch_candidate and right_pinch_candidate
        self._handle_clicks(
            left_pinch_candidate=left_pinch_candidate,
            right_pinch_candidate=right_pinch_candidate,
            ambiguous_pinch=ambiguous_pinch,
            pinch_ratios=pinch_ratios,
            now=now,
        )

        if ambiguous_pinch:
            return "Ambiguous Pinch Ignored", (0, 0, 255), "Mode: Ambiguous Pinch"

        return status, status_color, mode

    def _handle_scroll(self, scroll_y: float, now: float) -> None:
        state = self.state
        gesture = self.config.gesture

        if state.last_scroll_y is None:
            state.last_scroll_y = scroll_y
            return

        dy = scroll_y - state.last_scroll_y
        if abs(dy) < gesture.scroll_trigger_delta:
            return

        if now - state.last_scroll_time <= gesture.scroll_cooldown:
            return

        scroll_amount = int(-dy / gesture.scroll_trigger_delta * gesture.scroll_speed)
        if scroll_amount != 0:
            self.mouse.scroll(scroll_amount)
            state.last_scroll_time = now

        state.last_scroll_y = scroll_y

    def _handle_clicks(
        self,
        *,
        left_pinch_candidate: bool,
        right_pinch_candidate: bool,
        ambiguous_pinch: bool,
        pinch_ratios,
        now: float,
    ) -> None:
        state = self.state
        gesture = self.config.gesture

        if ambiguous_pinch:
            state.reset_click_candidates()
            self._release_inactive_pinches(pinch_ratios)
            return

        if left_pinch_candidate:
            state.left_pinch_frames += 1
            if self._can_trigger_click(state.left_pinch_frames, state.left_pinch_active, now):
                self.mouse.click("left")
                state.left_pinch_active = True
                state.last_click_time = now
                state.mouse_frozen_until = now + gesture.click_freeze_time
                print("触发左键点击：拇指 + 中指捏合")
        elif pinch_ratios.left > gesture.left_pinch_release_ratio:
            state.left_pinch_frames = 0
            state.left_pinch_active = False

        if right_pinch_candidate:
            state.right_pinch_frames += 1
            if self._can_trigger_click(state.right_pinch_frames, state.right_pinch_active, now):
                self.mouse.click("right")
                state.right_pinch_active = True
                state.last_click_time = now
                state.mouse_frozen_until = now + gesture.click_freeze_time
                print("触发右键点击：拇指 + 无名指捏合")
        elif pinch_ratios.right > gesture.right_pinch_release_ratio:
            state.right_pinch_frames = 0
            state.right_pinch_active = False

    def _can_trigger_click(self, frame_count: int, already_active: bool, now: float) -> bool:
        gesture = self.config.gesture
        return (
            frame_count >= gesture.pinch_confirm_frames
            and not already_active
            and now - self.state.last_click_time > gesture.click_cooldown
        )

    def _release_inactive_pinches(self, pinch_ratios) -> None:
        gesture = self.config.gesture
        if pinch_ratios.left > gesture.left_pinch_release_ratio:
            self.state.left_pinch_active = False
        if pinch_ratios.right > gesture.right_pinch_release_ratio:
            self.state.right_pinch_active = False

    def _draw_landmark_helpers(
        self,
        frame,
        *,
        index: tuple[int, int],
        middle: tuple[int, int],
        ring: tuple[int, int],
        thumb: tuple[int, int],
        mapped: tuple[int, int],
        index_in_area: bool,
    ) -> None:
        cv2.circle(frame, index, 12, (0, 255, 0), cv2.FILLED)
        cv2.circle(frame, mapped, 8, (255, 255, 255), cv2.FILLED)

        if not index_in_area:
            cv2.line(frame, index, mapped, (255, 255, 255), 2)

        cv2.circle(frame, middle, 10, (0, 255, 255), cv2.FILLED)
        cv2.circle(frame, ring, 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(frame, thumb, 10, (255, 255, 0), cv2.FILLED)
        cv2.line(frame, thumb, middle, (0, 255, 255), 2)
        cv2.line(frame, thumb, ring, (255, 0, 255), 2)

    def _draw_status_overlay(
        self,
        frame,
        *,
        index: tuple[int, int],
        mapped: tuple[int, int],
        status_text: str,
        status_color: tuple[int, int, int],
        mode_text: str,
        pinch_ratios,
        fingers,
    ) -> None:
        pause_text = "Paused: ON" if self.state.paused else "Paused: OFF"

        draw_text(frame, f"Index: ({index[0]}, {index[1]})", (20, 40), (0, 255, 0))
        draw_text(frame, f"Mapped: ({mapped[0]}, {mapped[1]})", (20, 70), (255, 255, 255), scale=0.65)
        draw_text(frame, status_text, (20, 105), status_color)
        draw_text(frame, mode_text, (20, 140), (255, 255, 255))
        draw_text(frame, f"Left Pinch Ratio: {pinch_ratios.left:.2f}", (20, 175), (0, 255, 255), scale=0.65)
        draw_text(frame, f"Right Pinch Ratio: {pinch_ratios.right:.2f}", (20, 205), (255, 0, 255), scale=0.65)
        draw_text(frame, pause_text, (20, 235), (0, 255, 255) if self.state.paused else (255, 255, 255), scale=0.65)
        finger_state_text = (
            f"I:{int(fingers.index_extended)} "
            f"M:{int(fingers.middle_extended)} "
            f"R:{int(fingers.ring_extended)} "
            f"P:{int(fingers.pinky_extended)}"
        )
        draw_text(frame, finger_state_text, (20, 265), (200, 200, 200), scale=0.65)


def main() -> None:
    GestureMouseApp().run()
