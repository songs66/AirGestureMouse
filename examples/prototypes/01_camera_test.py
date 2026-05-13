import cv2


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("无法读取摄像头画面")
            break

        # 镜像翻转，让画面更符合人的直觉
        frame = cv2.flip(frame, 1)

        cv2.imshow("Camera Test", frame)

        # 按 q 退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()