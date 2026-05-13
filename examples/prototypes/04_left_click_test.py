import cv2
import mediapipe as mp
import pyautogui
import math
import time


def clamp(value, min_value, max_value):
    """
    将 value 限制在 [min_value, max_value] 范围内。
    防止坐标超出摄像头画面或屏幕范围。
    """
    return max(min_value, min(value, max_value))


def calc_distance(p1, p2):
    """
    计算两个 MediaPipe 关键点之间的二维距离。
    注意：这里用的是归一化坐标，不是像素坐标。
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
    # 2. 点击识别参数
    # ==============================

    # 捏合触发阈值
    # 数值越大，越容易触发点击
    # 数值越小，需要捏得更紧才触发
    PINCH_TRIGGER_RATIO = 0.35

    # 松开阈值
    # 必须大于触发阈值，否则容易在临界点反复抖动
    PINCH_RELEASE_RATIO = 0.50

    # 连续多少帧检测到捏合，才认为是真点击
    PINCH_CONFIRM_FRAMES = 3

    # 点击冷却时间，单位秒
    CLICK_COOLDOWN = 0.35

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0

    screen_width, screen_height = pyautogui.size()
    print(f"屏幕尺寸: {screen_width} x {screen_height}")

    prev_mouse_x = screen_width / 2
    prev_mouse_y = screen_height / 2

    # 左键点击状态机变量
    pinch_frame_count = 0
    left_pinch_active = False
    last_click_time = 0

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
    print("伸出食指移动鼠标")
    print("拇指 + 食指捏合：左键点击")
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
        pinch_text = "Pinch: --"
        click_text = ""

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
            # 8  ：食指指尖
            # 5  ：食指根部
            # 17 ：小指根部
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            index_mcp = landmarks[5]
            pinky_mcp = landmarks[17]

            finger_x = int(index_tip.x * frame_width)
            finger_y = int(index_tip.y * frame_height)

            thumb_x = int(thumb_tip.x * frame_width)
            thumb_y = int(thumb_tip.y * frame_height)

            cv2.circle(frame, (finger_x, finger_y), 12, (0, 255, 0), cv2.FILLED)
            cv2.circle(frame, (thumb_x, thumb_y), 10, (255, 255, 0), cv2.FILLED)

            # 画出拇指和食指之间的连线
            cv2.line(
                frame,
                (thumb_x, thumb_y),
                (finger_x, finger_y),
                (255, 255, 0),
                2
            )

            # ==============================
            # 9. 食指控制鼠标移动
            # ==============================

            finger_in_control_area = x1 <= finger_x <= x2 and y1 <= finger_y <= y2

            if finger_in_control_area:
                target_mouse_x = (finger_x - x1) / (x2 - x1) * screen_width
                target_mouse_y = (finger_y - y1) / (y2 - y1) * screen_height

                target_mouse_x = clamp(target_mouse_x, 0, screen_width - 1)
                target_mouse_y = clamp(target_mouse_y, 0, screen_height - 1)

                current_mouse_x = prev_mouse_x + (target_mouse_x - prev_mouse_x) * SMOOTHING
                current_mouse_y = prev_mouse_y + (target_mouse_y - prev_mouse_y) * SMOOTHING

                pyautogui.moveTo(current_mouse_x, current_mouse_y)

                prev_mouse_x = current_mouse_x
                prev_mouse_y = current_mouse_y

                status_text = "Mouse Moving"
                status_color = (0, 255, 0)
            else:
                status_text = "Finger Outside Control Area"
                status_color = (0, 0, 255)

            # ==============================
            # 10. 左键点击识别
            # ==============================

            # 拇指和食指指尖距离
            pinch_distance = calc_distance(thumb_tip, index_tip)

            # 用手掌宽度作为参考距离，避免手离摄像头远近影响判断
            palm_width = calc_distance(index_mcp, pinky_mcp)

            if palm_width < 0.0001:
                pinch_ratio = 999
            else:
                pinch_ratio = pinch_distance / palm_width

            pinch_text = f"Pinch Ratio: {pinch_ratio:.2f}"

            current_time = time.time()

            # 只有食指在控制区域内时，才允许点击
            if finger_in_control_area:
                if pinch_ratio < PINCH_TRIGGER_RATIO:
                    pinch_frame_count += 1

                    # 连续多帧捏合 + 当前不是已触发状态 + 冷却时间结束
                    if (
                        pinch_frame_count >= PINCH_CONFIRM_FRAMES
                        and not left_pinch_active
                        and current_time - last_click_time > CLICK_COOLDOWN
                    ):
                        pyautogui.click(button="left")
                        left_pinch_active = True
                        last_click_time = current_time
                        click_text = "Left Click!"
                        print("触发左键点击")

                elif pinch_ratio > PINCH_RELEASE_RATIO:
                    # 手指松开后，允许下一次点击
                    pinch_frame_count = 0
                    left_pinch_active = False
            else:
                # 手指不在控制区时，不进行点击判断
                pinch_frame_count = 0
                left_pinch_active = False

            # ==============================
            # 11. 显示状态信息
            # ==============================

            cv2.putText(
                frame,
                f"Finger: ({finger_x}, {finger_y})",
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
                (255, 255, 0),
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
        # 12. 显示画面
        # ==============================

        cv2.imshow("Gesture Mouse Left Click Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ==============================
    # 13. 释放资源
    # ==============================

    hands.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()