import cv2
import mediapipe as mp
import pyautogui
import math
import time


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def calc_distance(p1, p2):
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    return math.sqrt(dx * dx + dy * dy)


def is_finger_extended(landmarks, tip_id, pip_id, margin=0.02):
    """
    判断手指是否伸出。
    MediaPipe 图像坐标中，y 越小越靠上。
    """
    return landmarks[tip_id].y < landmarks[pip_id].y - margin


def is_finger_folded(landmarks, tip_id, pip_id, margin=0.02):
    """
    判断手指是否弯曲。
    """
    return landmarks[tip_id].y > landmarks[pip_id].y + margin


def draw_text(frame, text, position, color=(255, 255, 255), scale=0.72, thickness=2):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness
    )


def keep_window_topmost(window_name):
    """
    尽量保持 OpenCV 窗口置顶。
    部分 OpenCV / 系统环境可能不支持 WND_PROP_TOPMOST，
    所以这里用 try 防止程序崩溃。
    """
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass


def main():
    # ==============================
    # 1. 基础参数
    # ==============================

    CAMERA_INDEX = 0

    # 紫色控制框参数
    CONTROL_MARGIN_X = 220
    CONTROL_MARGIN_Y = 120
    CONTROL_OFFSET_Y = -80

    # 鼠标平滑系数
    SMOOTHING = 0.18

    # ==============================
    # 2. 点击参数：触发距离较小，减少误触
    # ==============================

    # 拇指 + 中指：左键
    LEFT_PINCH_TRIGGER_RATIO = 0.28
    LEFT_PINCH_RELEASE_RATIO = 0.43

    # 拇指 + 无名指：右键
    RIGHT_PINCH_TRIGGER_RATIO = 0.28
    RIGHT_PINCH_RELEASE_RATIO = 0.43

    # 连续多少帧检测到捏合才触发点击
    PINCH_CONFIRM_FRAMES = 4

    # 点击冷却时间
    CLICK_COOLDOWN = 0.45

    # 点击时冻结鼠标，避免点击瞬间光标漂移
    CLICK_FREEZE_TIME = 0.25

    # ==============================
    # 3. 滚轮参数
    # ==============================

    # 三指上下移动触发滚轮
    SCROLL_TRIGGER_DELTA = 10
    SCROLL_SPEED = 3
    SCROLL_COOLDOWN = 0.04

    # ==============================
    # 4. 暂停 / 恢复参数
    # ==============================

    FIST_CONFIRM_FRAMES = 7
    PAUSE_TOGGLE_COOLDOWN = 0.8

    # ==============================
    # 5. PyAutoGUI 初始化
    # ==============================

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0

    screen_width, screen_height = pyautogui.size()
    print(f"屏幕尺寸: {screen_width} x {screen_height}")

    # ==============================
    # 6. 视频窗口参数
    # ==============================

    WINDOW_NAME = "Gesture Mouse HCI System - Final TopMost"

    # 视频窗口大小
    WINDOW_WIDTH = 640
    WINDOW_HEIGHT = 360

    # 显示在屏幕左下角
    # 减去 80 是为了避免被 Windows 任务栏遮挡
    WINDOW_X = 0
    WINDOW_Y = screen_height - WINDOW_HEIGHT - 80

    prev_mouse_x = screen_width / 2
    prev_mouse_y = screen_height / 2

    # ==============================
    # 7. 状态变量
    # ==============================

    left_pinch_frame_count = 0
    right_pinch_frame_count = 0

    left_pinch_active = False
    right_pinch_active = False

    last_click_time = 0
    mouse_frozen_until = 0

    last_scroll_y = None
    last_scroll_time = 0

    paused = False
    fist_frame_count = 0
    fist_active = False
    last_pause_toggle_time = 0

    # ==============================
    # 8. 初始化摄像头
    # ==============================

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # ==============================
    # 9. 初始化 MediaPipe
    # ==============================

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    print("最终置顶版空气鼠标启动成功")
    print("食指移动：鼠标移动")
    print("食指移出紫色框：鼠标被限制在屏幕边缘继续移动")
    print("拇指 + 中指捏合：左键点击")
    print("拇指 + 无名指捏合：右键点击")
    print("食指 + 中指 + 无名指伸出并上下移动：滚轮")
    print("握拳：暂停 / 恢复控制")
    print("视频窗口：左下角、小窗口、始终置顶")
    print("按 q 退出程序")

    # 创建可调整大小的 OpenCV 窗口
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # 设置窗口大小
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)

    # 移动窗口到屏幕左下角
    cv2.moveWindow(WINDOW_NAME, WINDOW_X, WINDOW_Y)

    # 设置窗口置顶
    keep_window_topmost(WINDOW_NAME)

    # ==============================
    # 10. 主循环
    # ==============================

    while True:
        ret, frame = cap.read()

        if not ret:
            print("无法读取摄像头画面")
            break

        frame = cv2.flip(frame, 1)

        frame_height, frame_width, _ = frame.shape

        # ==============================
        # 11. 控制区域
        # ==============================

        x1 = CONTROL_MARGIN_X
        x2 = frame_width - CONTROL_MARGIN_X

        y1 = CONTROL_MARGIN_Y + CONTROL_OFFSET_Y
        y2 = frame_height - CONTROL_MARGIN_Y + CONTROL_OFFSET_Y

        x1 = int(clamp(x1, 0, frame_width - 1))
        x2 = int(clamp(x2, 0, frame_width - 1))
        y1 = int(clamp(y1, 0, frame_height - 1))
        y2 = int(clamp(y2, 0, frame_height - 1))

        if x2 <= x1 or y2 <= y1:
            print("控制区域参数错误")
            break

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        draw_text(frame, "Control Area", (x1, max(y1 - 10, 30)), (255, 0, 255))

        # ==============================
        # 12. 手部检测
        # ==============================

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        rgb_frame.flags.writeable = False
        results = hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        current_time = time.time()

        status_text = "No Hand Detected"
        status_color = (0, 0, 255)
        mode_text = "Mode: --"
        pause_text = "Paused: ON" if paused else "Paused: OFF"

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = hand_landmarks.landmark

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # ==============================
            # 13. 关键点
            # ==============================

            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            ring_tip = landmarks[16]

            index_mcp = landmarks[5]
            pinky_mcp = landmarks[17]

            index_pip = 6
            middle_pip = 10
            ring_pip = 14
            pinky_pip = 18

            index_x = int(index_tip.x * frame_width)
            index_y = int(index_tip.y * frame_height)

            middle_x = int(middle_tip.x * frame_width)
            middle_y = int(middle_tip.y * frame_height)

            ring_x = int(ring_tip.x * frame_width)
            ring_y = int(ring_tip.y * frame_height)

            thumb_x = int(thumb_tip.x * frame_width)
            thumb_y = int(thumb_tip.y * frame_height)

            # ==============================
            # 14. 边界钳制核心逻辑
            # ==============================

            index_in_control_area = x1 <= index_x <= x2 and y1 <= index_y <= y2

            # 只要检测到了手，就认为食指存在
            index_detected = True

            # 将食指坐标钳制到紫色框边界
            mapped_index_x = clamp(index_x, x1, x2)
            mapped_index_y = clamp(index_y, y1, y2)

            mapped_draw_x = int(mapped_index_x)
            mapped_draw_y = int(mapped_index_y)

            # 真实食指位置：绿色
            cv2.circle(frame, (index_x, index_y), 12, (0, 255, 0), cv2.FILLED)

            # 被钳制后的映射点：白色
            cv2.circle(frame, (mapped_draw_x, mapped_draw_y), 8, (255, 255, 255), cv2.FILLED)

            # 如果食指在框外，画一条白线连接真实点和映射点
            if not index_in_control_area:
                cv2.line(
                    frame,
                    (index_x, index_y),
                    (mapped_draw_x, mapped_draw_y),
                    (255, 255, 255),
                    2
                )

            # 中指：黄色
            cv2.circle(frame, (middle_x, middle_y), 10, (0, 255, 255), cv2.FILLED)

            # 无名指：紫色
            cv2.circle(frame, (ring_x, ring_y), 10, (255, 0, 255), cv2.FILLED)

            # 拇指：青色
            cv2.circle(frame, (thumb_x, thumb_y), 10, (255, 255, 0), cv2.FILLED)

            # 捏合辅助线
            cv2.line(frame, (thumb_x, thumb_y), (middle_x, middle_y), (0, 255, 255), 2)
            cv2.line(frame, (thumb_x, thumb_y), (ring_x, ring_y), (255, 0, 255), 2)

            # ==============================
            # 15. 手指伸缩状态
            # ==============================

            index_extended = is_finger_extended(landmarks, 8, index_pip)
            middle_extended = is_finger_extended(landmarks, 12, middle_pip)
            ring_extended = is_finger_extended(landmarks, 16, ring_pip)
            pinky_extended = is_finger_extended(landmarks, 20, pinky_pip)

            index_folded = is_finger_folded(landmarks, 8, index_pip)
            middle_folded = is_finger_folded(landmarks, 12, middle_pip)
            ring_folded = is_finger_folded(landmarks, 16, ring_pip)
            pinky_folded = is_finger_folded(landmarks, 20, pinky_pip)

            # 握拳：四根非拇指手指都弯曲
            fist_candidate = (
                index_folded
                and middle_folded
                and ring_folded
                and pinky_folded
            )

            # 三指滚轮：
            # 食指 + 中指 + 无名指伸出，小指弯曲
            scroll_mode_candidate = (
                index_extended
                and middle_extended
                and ring_extended
                and pinky_folded
            )

            # ==============================
            # 16. 捏合比例
            # ==============================

            palm_width = calc_distance(index_mcp, pinky_mcp)

            if palm_width < 0.0001:
                left_pinch_ratio = 999
                right_pinch_ratio = 999
            else:
                left_pinch_ratio = calc_distance(thumb_tip, middle_tip) / palm_width
                right_pinch_ratio = calc_distance(thumb_tip, ring_tip) / palm_width

            # 三指滚轮时不触发点击
            left_pinch_candidate = (
                index_detected
                and not scroll_mode_candidate
                and left_pinch_ratio < LEFT_PINCH_TRIGGER_RATIO
            )

            right_pinch_candidate = (
                index_detected
                and not scroll_mode_candidate
                and right_pinch_ratio < RIGHT_PINCH_TRIGGER_RATIO
            )

            click_candidate = left_pinch_candidate or right_pinch_candidate

            # ==============================
            # 17. 握拳：暂停 / 恢复
            # ==============================

            if fist_candidate:
                fist_frame_count += 1

                if (
                    fist_frame_count >= FIST_CONFIRM_FRAMES
                    and not fist_active
                    and current_time - last_pause_toggle_time > PAUSE_TOGGLE_COOLDOWN
                ):
                    paused = not paused
                    fist_active = True
                    last_pause_toggle_time = current_time

                    left_pinch_frame_count = 0
                    right_pinch_frame_count = 0
                    left_pinch_active = False
                    right_pinch_active = False
                    last_scroll_y = None

                    if paused:
                        print("已暂停手势控制")
                    else:
                        print("已恢复手势控制")
            else:
                fist_frame_count = 0
                fist_active = False

            if paused:
                status_text = "Gesture Control Paused"
                status_color = (0, 255, 255)
                mode_text = "Mode: Paused"

            else:
                # ==============================
                # 18. 三指滚轮模式
                # ==============================

                scroll_mode = (
                    index_detected
                    and scroll_mode_candidate
                    and not click_candidate
                    and not fist_candidate
                )

                if scroll_mode:
                    mode_text = "Mode: Three-Finger Scroll"
                    status_text = "Three-Finger Scroll"
                    status_color = (255, 255, 0)

                    # 用三根手指指尖 y 坐标平均值作为滚轮参考点
                    scroll_y = (index_y + middle_y + ring_y) / 3

                    if last_scroll_y is None:
                        last_scroll_y = scroll_y
                    else:
                        dy = scroll_y - last_scroll_y

                        if (
                            abs(dy) >= SCROLL_TRIGGER_DELTA
                            and current_time - last_scroll_time > SCROLL_COOLDOWN
                        ):
                            # dy > 0：手向下移动，页面向下滚
                            # pyautogui.scroll 正数向上，负数向下
                            scroll_amount = int(-dy / SCROLL_TRIGGER_DELTA * SCROLL_SPEED)

                            if scroll_amount != 0:
                                pyautogui.scroll(scroll_amount)
                                last_scroll_time = current_time

                            last_scroll_y = scroll_y

                    left_pinch_frame_count = 0
                    right_pinch_frame_count = 0
                    left_pinch_active = False
                    right_pinch_active = False

                else:
                    last_scroll_y = None

                    # ==============================
                    # 19. 鼠标移动：边界钳制版
                    # ==============================

                    mouse_frozen = (
                        current_time < mouse_frozen_until
                        or click_candidate
                    )

                    if index_detected and not mouse_frozen:
                        # 用 mapped_index_x / mapped_index_y 映射
                        # 食指出框时，鼠标会停在屏幕边缘继续沿边缘移动
                        target_mouse_x = (mapped_index_x - x1) / (x2 - x1) * screen_width
                        target_mouse_y = (mapped_index_y - y1) / (y2 - y1) * screen_height

                        target_mouse_x = clamp(target_mouse_x, 0, screen_width - 1)
                        target_mouse_y = clamp(target_mouse_y, 0, screen_height - 1)

                        current_mouse_x = prev_mouse_x + (target_mouse_x - prev_mouse_x) * SMOOTHING
                        current_mouse_y = prev_mouse_y + (target_mouse_y - prev_mouse_y) * SMOOTHING

                        pyautogui.moveTo(int(current_mouse_x), int(current_mouse_y))

                        prev_mouse_x = current_mouse_x
                        prev_mouse_y = current_mouse_y

                        if index_in_control_area:
                            status_text = "Mouse Moving"
                            mode_text = "Mode: Move"
                        else:
                            status_text = "Mouse Edge Moving"
                            mode_text = "Mode: Edge Clamp"

                        status_color = (0, 255, 0)

                    elif index_detected and mouse_frozen:
                        status_text = "Mouse Frozen For Click"
                        status_color = (0, 255, 255)
                        mode_text = "Mode: Click Candidate"

                    else:
                        status_text = "No Index Detected"
                        status_color = (0, 0, 255)
                        mode_text = "Mode: Waiting"

                    # ==============================
                    # 20. 点击逻辑
                    # ==============================

                    ambiguous_pinch = left_pinch_candidate and right_pinch_candidate

                    if index_detected and not ambiguous_pinch:
                        # 左键：拇指 + 中指
                        if left_pinch_candidate:
                            left_pinch_frame_count += 1

                            if (
                                left_pinch_frame_count >= PINCH_CONFIRM_FRAMES
                                and not left_pinch_active
                                and current_time - last_click_time > CLICK_COOLDOWN
                            ):
                                pyautogui.click(button="left")

                                left_pinch_active = True
                                last_click_time = current_time
                                mouse_frozen_until = current_time + CLICK_FREEZE_TIME

                                print("触发左键点击：拇指 + 中指捏合")

                        elif left_pinch_ratio > LEFT_PINCH_RELEASE_RATIO:
                            left_pinch_frame_count = 0
                            left_pinch_active = False

                        # 右键：拇指 + 无名指
                        if right_pinch_candidate:
                            right_pinch_frame_count += 1

                            if (
                                right_pinch_frame_count >= PINCH_CONFIRM_FRAMES
                                and not right_pinch_active
                                and current_time - last_click_time > CLICK_COOLDOWN
                            ):
                                pyautogui.click(button="right")

                                right_pinch_active = True
                                last_click_time = current_time
                                mouse_frozen_until = current_time + CLICK_FREEZE_TIME

                                print("触发右键点击：拇指 + 无名指捏合")

                        elif right_pinch_ratio > RIGHT_PINCH_RELEASE_RATIO:
                            right_pinch_frame_count = 0
                            right_pinch_active = False

                    else:
                        left_pinch_frame_count = 0
                        right_pinch_frame_count = 0

                        if left_pinch_ratio > LEFT_PINCH_RELEASE_RATIO:
                            left_pinch_active = False

                        if right_pinch_ratio > RIGHT_PINCH_RELEASE_RATIO:
                            right_pinch_active = False

                    if ambiguous_pinch:
                        mode_text = "Mode: Ambiguous Pinch"
                        status_text = "Ambiguous Pinch Ignored"
                        status_color = (0, 0, 255)

            # ==============================
            # 21. 显示状态信息
            # ==============================

            draw_text(frame, f"Index: ({index_x}, {index_y})", (20, 40), (0, 255, 0))
            draw_text(frame, f"Mapped: ({mapped_draw_x}, {mapped_draw_y})", (20, 70), (255, 255, 255), scale=0.65)
            draw_text(frame, status_text, (20, 105), status_color)
            draw_text(frame, mode_text, (20, 140), (255, 255, 255))

            draw_text(
                frame,
                f"Left Pinch Ratio: {left_pinch_ratio:.2f}",
                (20, 175),
                (0, 255, 255),
                scale=0.65
            )

            draw_text(
                frame,
                f"Right Pinch Ratio: {right_pinch_ratio:.2f}",
                (20, 205),
                (255, 0, 255),
                scale=0.65
            )

            draw_text(
                frame,
                pause_text,
                (20, 235),
                (0, 255, 255) if paused else (255, 255, 255),
                scale=0.65
            )

            finger_state_text = (
                f"I:{int(index_extended)} "
                f"M:{int(middle_extended)} "
                f"R:{int(ring_extended)} "
                f"P:{int(pinky_extended)}"
            )

            draw_text(frame, finger_state_text, (20, 265), (200, 200, 200), scale=0.65)

        else:
            # 真正检测不到手时，鼠标才停止移动
            left_pinch_frame_count = 0
            right_pinch_frame_count = 0
            left_pinch_active = False
            right_pinch_active = False
            last_scroll_y = None
            fist_frame_count = 0
            fist_active = False

            draw_text(frame, status_text, (20, 40), status_color)
            draw_text(
                frame,
                pause_text,
                (20, 80),
                (0, 255, 255) if paused else (255, 255, 255)
            )

        cv2.imshow(WINDOW_NAME, frame)

        # 持续保持窗口置顶
        keep_window_topmost(WINDOW_NAME)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    hands.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()