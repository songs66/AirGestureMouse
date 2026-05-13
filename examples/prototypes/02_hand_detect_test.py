import cv2
import mediapipe as mp
import pyautogui
import math
import time


def clamp(value, min_value, max_value):
    """
    将 value 限制在 [min_value, max_value] 范围内。
    """
    return max(min_value, min(value, max_value))


def calc_distance(p1, p2):
    """
    计算两个 MediaPipe 关键点之间的二维距离。
    使用归一化坐标，不使用像素坐标。
    """
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    return math.sqrt(dx * dx + dy * dy)


def main():
    # ==============================
    # 1. 基础参数配置
    # ==============================

    CAMERA_INDEX = 0

    # 控制区域参数
    CONTROL_MARGIN_X = 220
    CONTROL_MARGIN_Y = 120
    CONTROL_OFFSET_Y = -80

    # 鼠标平滑系数
    SMOOTHING = 0.18

    # ==============================
    # 2. 左键点击参数
    # ==============================

    # 拇指 + 中指 捏合触发阈值
    # 数值越大，越容易触发
    # 数值越小，需要捏得更紧
    MIDDLE_PINCH_TRIGGER_RATIO = 0.35

    # 松开阈值，必须大于触发阈值
    MIDDLE_PINCH_RELEASE_RATIO = 0.50

    # 连续多少帧检测到捏合才触发点击
    PINCH_CONFIRM_FRAMES = 3

    # 点击冷却时间，防止连点
    CLICK_COOLDOWN = 0.35

    # 点击时冻结鼠标移动的时间，防止点击瞬间漂移
    CLICK_FREEZE_TIME = 0.20

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0

    screen_width, screen_height = pyautogui.size()
    print(f"屏幕尺寸: {screen_width} x {screen_height}")

    prev_mouse_x = screen_width / 2
    prev_mouse_y = screen_height / 2

    # 点击状态机变量
    pinch_frame_count = 0
    left_pinch_active = False
    last_click_time = 0

    # 鼠标冻结时间点
    mouse_frozen_until = 0

    # ==============================
    # 3. 初始化摄像头
    # ==============================

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # ==============================
    # 4. 初始化 MediaPipe Hands
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

    print("空气鼠标启动成功")
    print("食指：控制鼠标移动")
    print("拇指 + 中指捏合：左键点击")
    print("按 q 退出程序")

    # ==============================
    # 5. 主循环
    # ==============================

    while True:
        ret, frame = cap.read()

        if not ret:
            print("无法读取摄像头画面")
            break

        frame = cv2.flip(frame, 1)

        frame_height, frame_width, _ = frame.shape

        # ==============================
        # 6. 计算有效控制区域
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
            print("控制区域参数错误，请检查 CONTROL_MARGIN_X / CONTROL_MARGIN_Y / CONTROL_OFFSET_Y")
            break

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

        cv2.putText(
            frame,
            "Control Area",
            (x1, max(y1 - 10, 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 255),
            2
        )

        # ==============================
        # 7. 手部检测
        # ==============================

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        rgb_frame.flags.writeable = False
        results = hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        status_text = "No Hand Detected"
        status_color = (0, 0, 255)
        pinch_text = "Middle Pinch: --"
        click_text = ""

        current_time = time.time()

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = hand_landmarks.landmark

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # ==============================
            # 8. 获取关键点
            # ==============================

            # 4  ：拇指指尖
            # 8  ：食指指尖，负责鼠标移动
            # 12 ：中指指尖，负责点击判断
            # 5  ：食指根部
            # 17 ：小指根部
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            index_mcp = landmarks[5]
            pinky_mcp = landmarks[17]

            index_x = int(index_tip.x * frame_width)
            index_y = int(index_tip.y * frame_height)

            thumb_x = int(thumb_tip.x * frame_width)
            thumb_y = int(thumb_tip.y * frame_height)

            middle_x = int(middle_tip.x * frame_width)
            middle_y = int(middle_tip.y * frame_height)

            # 食指：绿色圆点，负责移动
            cv2.circle(frame, (index_x, index_y), 12, (0, 255, 0), cv2.FILLED)

            # 拇指：青色圆点
            cv2.circle(frame, (thumb_x, thumb_y), 10, (255, 255, 0), cv2.FILLED)

            # 中指：黄色圆点
            cv2.circle(frame, (middle_x, middle_y), 10, (0, 255, 255), cv2.FILLED)

            # 画出拇指和中指之间的连线
            cv2.line(
                frame,
                (thumb_x, thumb_y),
                (middle_x, middle_y),
                (0, 255, 255),
                2
            )

            # ==============================
            # 9. 计算拇指 + 中指捏合比例
            # ==============================

            pinch_distance = calc_distance(thumb_tip, middle_tip)

            # 用手掌宽度作为参考距离，减少手远近变化带来的影响
            palm_width = calc_distance(index_mcp, pinky_mcp)

            if palm_width < 0.0001:
                middle_pinch_ratio = 999
            else:
                middle_pinch_ratio = pinch_distance / palm_width

            pinch_text = f"Middle Pinch Ratio: {middle_pinch_ratio:.2f}"

            finger_in_control_area = x1 <= index_x <= x2 and y1 <= index_y <= y2

            # 是否正在进行点击手势
            middle_pinch_candidate = (
                finger_in_control_area
                and middle_pinch_ratio < MIDDLE_PINCH_TRIGGER_RATIO
            )

            # ==============================
            # 10. 食指控制鼠标移动
            # ==============================

            # 如果正在点击，或者刚刚点击完，短暂冻结鼠标
            mouse_frozen = current_time < mouse_frozen_until or middle_pinch_candidate

            if finger_in_control_area and not mouse_frozen:
                target_mouse_x = (index_x - x1) / (x2 - x1) * screen_width
                target_mouse_y = (index_y - y1) / (y2 - y1) * screen_height

                target_mouse_x = clamp(target_mouse_x, 0, screen_width - 1)
                target_mouse_y = clamp(target_mouse_y, 0, screen_height - 1)

                current_mouse_x = prev_mouse_x + (target_mouse_x - prev_mouse_x) * SMOOTHING
                current_mouse_y = prev_mouse_y + (target_mouse_y - prev_mouse_y) * SMOOTHING

                pyautogui.moveTo(current_mouse_x, current_mouse_y)

                prev_mouse_x = current_mouse_x
                prev_mouse_y = current_mouse_y

                status_text = "Mouse Moving"
                status_color = (0, 255, 0)

            elif finger_in_control_area and mouse_frozen:
                status_text = "Mouse Frozen For Click"
                status_color = (0, 255, 255)

            else:
                status_text = "Finger Outside Control Area"
                status_color = (0, 0, 255)

            # ==============================
            # 11. 左键点击状态机
            # ==============================

            if finger_in_control_area:
                if middle_pinch_ratio < MIDDLE_PINCH_TRIGGER_RATIO:
                    pinch_frame_count += 1

                    if (
                        pinch_frame_count >= PINCH_CONFIRM_FRAMES
                        and not left_pinch_active
                        and current_time - last_click_time > CLICK_COOLDOWN
                    ):
                        pyautogui.click(button="left")

                        left_pinch_active = True
                        last_click_time = current_time
                        mouse_frozen_until = current_time + CLICK_FREEZE_TIME

                        click_text = "Left Click!"
                        print("触发左键点击：拇指 + 中指捏合")

                elif middle_pinch_ratio > MIDDLE_PINCH_RELEASE_RATIO:
                    # 松开后才允许下一次点击
                    pinch_frame_count = 0
                    left_pinch_active = False
            else:
                pinch_frame_count = 0
                left_pinch_active = False

            # ==============================
            # 12. 显示状态信息
            # ==============================

            cv2.putText(
                frame,
                f"Index: ({index_x}, {index_y})",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                status_text,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2
            )

            cv2.putText(
                frame,
                pinch_text,
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            if click_text:
                cv2.putText(
                    frame,
                    click_text,
                    (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    3
                )

        else:
            pinch_frame_count = 0
            left_pinch_active = False

            cv2.putText(
                frame,
                status_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2
            )

        # ==============================
        # 13. 显示画面
        # ==============================

        cv2.imshow("Gesture Mouse - Middle Pinch Left Click", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ==============================
    # 14. 释放资源
    # ==============================

    hands.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()