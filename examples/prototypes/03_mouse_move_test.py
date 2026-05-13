import cv2
import mediapipe as mp
import pyautogui


def clamp(value, min_value, max_value):
    """
    将 value 限制在 [min_value, max_value] 范围内。
    防止坐标超出摄像头画面或屏幕范围。
    """
    return max(min_value, min(value, max_value))


def main():
    # ==============================
    # 1. 基础参数配置
    # ==============================

    # 摄像头编号，内置摄像头通常是 0
    CAMERA_INDEX = 0

    # 控制区域参数
    # 左右缩小的距离，数值越大，紫色框越窄
    CONTROL_MARGIN_X = 220

    # 上下缩小的距离，数值越大，紫色框越矮
    CONTROL_MARGIN_Y = 120

    # 控制区域整体上下偏移
    # 负数表示向上移动，正数表示向下移动
    CONTROL_OFFSET_Y = -80

    # 鼠标平滑系数
    # 数值越大，鼠标越灵敏；数值越小，鼠标越平滑
    # 控制区域缩小后，建议比之前稍微低一点
    SMOOTHING = 0.18

    # PyAutoGUI 安全保护：
    # 鼠标移动到屏幕左上角会触发保护异常，防止鼠标失控
    pyautogui.FAILSAFE = True

    # 去掉 PyAutoGUI 每次操作后的默认暂停，提高响应速度
    pyautogui.PAUSE = 0

    # 获取屏幕尺寸
    screen_width, screen_height = pyautogui.size()
    print(f"屏幕尺寸: {screen_width} x {screen_height}")

    # 上一次鼠标位置，用于平滑滤波
    prev_mouse_x = screen_width / 2
    prev_mouse_y = screen_height / 2

    # ==============================
    # 2. 初始化摄像头
    # ==============================

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # ==============================
    # 3. 初始化 MediaPipe Hands
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
    print("按 q 退出程序")

    # ==============================
    # 4. 主循环
    # ==============================

    while True:
        ret, frame = cap.read()

        if not ret:
            print("无法读取摄像头画面")
            break

        # 镜像翻转：
        # 这样你向右移动手，画面中的手和鼠标也向右移动
        frame = cv2.flip(frame, 1)

        frame_height, frame_width, _ = frame.shape

        # ==============================
        # 5. 计算有效控制区域
        # ==============================

        # x 方向：左右缩小
        x1 = CONTROL_MARGIN_X
        x2 = frame_width - CONTROL_MARGIN_X

        # y 方向：上下缩小，并整体向上移动
        y1 = CONTROL_MARGIN_Y + CONTROL_OFFSET_Y
        y2 = frame_height - CONTROL_MARGIN_Y + CONTROL_OFFSET_Y

        # 防止控制区域超出摄像头画面
        x1 = int(clamp(x1, 0, frame_width - 1))
        x2 = int(clamp(x2, 0, frame_width - 1))
        y1 = int(clamp(y1, 0, frame_height - 1))
        y2 = int(clamp(y2, 0, frame_height - 1))

        # 如果参数设置异常，避免除以 0
        if x2 <= x1 or y2 <= y1:
            print("控制区域参数错误，请检查 CONTROL_MARGIN_X / CONTROL_MARGIN_Y / CONTROL_OFFSET_Y")
            break

        # 画出有效控制区域
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
        # 6. 手部检测
        # ==============================

        # OpenCV 默认是 BGR，MediaPipe 需要 RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 提高性能：告诉 MediaPipe 这张图不需要写入
        rgb_frame.flags.writeable = False
        results = hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        # ==============================
        # 7. 根据食指指尖控制鼠标
        # ==============================

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # 绘制手部骨架
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # 食指指尖关键点编号是 8
            index_tip = hand_landmarks.landmark[8]

            # MediaPipe 给的是归一化坐标，需要转成像素坐标
            finger_x = int(index_tip.x * frame_width)
            finger_y = int(index_tip.y * frame_height)

            # 在食指指尖画绿色圆点
            cv2.circle(
                frame,
                (finger_x, finger_y),
                12,
                (0, 255, 0),
                cv2.FILLED
            )

            # 判断食指是否在紫色控制区域内
            if x1 <= finger_x <= x2 and y1 <= finger_y <= y2:
                # 摄像头控制区域坐标 -> 屏幕坐标
                target_mouse_x = (finger_x - x1) / (x2 - x1) * screen_width
                target_mouse_y = (finger_y - y1) / (y2 - y1) * screen_height

                # 限制鼠标目标坐标范围
                target_mouse_x = clamp(target_mouse_x, 0, screen_width - 1)
                target_mouse_y = clamp(target_mouse_y, 0, screen_height - 1)

                # 平滑滤波，降低鼠标抖动
                current_mouse_x = prev_mouse_x + (target_mouse_x - prev_mouse_x) * SMOOTHING
                current_mouse_y = prev_mouse_y + (target_mouse_y - prev_mouse_y) * SMOOTHING

                # 移动鼠标
                pyautogui.moveTo(current_mouse_x, current_mouse_y)

                # 更新上一帧鼠标位置
                prev_mouse_x = current_mouse_x
                prev_mouse_y = current_mouse_y

                status_text = "Mouse Moving"
                status_color = (0, 255, 0)
            else:
                status_text = "Finger Outside Control Area"
                status_color = (0, 0, 255)

            # 显示食指坐标
            cv2.putText(
                frame,
                f"Finger: ({finger_x}, {finger_y})",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # 显示当前状态
            cv2.putText(
                frame,
                status_text,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2
            )

        else:
            cv2.putText(
                frame,
                "No Hand Detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # ==============================
        # 8. 显示画面
        # ==============================

        cv2.imshow("Gesture Mouse Move Test", frame)

        # 按 q 退出程序
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ==============================
    # 9. 释放资源
    # ==============================

    hands.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()