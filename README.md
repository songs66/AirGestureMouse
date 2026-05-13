# AirGestureMouse

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)
![OpenCV](https://img.shields.io/badge/OpenCV-Real--time-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-orange)
![License](https://img.shields.io/badge/License-MIT-black)

AirGestureMouse 是一个基于摄像头、OpenCV、MediaPipe 和 PyAutoGUI 的空气鼠标项目。它可以实时识别手部关键点，把食指移动映射为鼠标移动，并支持左键、右键、滚轮和暂停控制。

这个项目适合作为计算机视觉、人机交互、Python 桌面自动化的实践项目，也适合继续扩展成无接触控制、演示控制器、辅助交互工具或课程作品。

## 亮点

- 实时手部追踪：使用 MediaPipe Hands 获取 21 个手部关键点。
- 边界钳制映射：食指离开控制框时，鼠标仍会稳定停留在屏幕边缘。
- 防误触点击：捏合距离、确认帧数、释放阈值、点击冷却共同降低误触。
- 多手势控制：移动、左键、右键、滚轮、暂停/恢复一套完整闭环。
- 工程化结构：从单文件 Demo 重构为可安装 Python 包，便于阅读、调参和二次开发。

## 手势说明

| 手势 | 功能 |
| --- | --- |
| 食指移动 | 控制鼠标移动 |
| 拇指 + 中指捏合 | 左键点击 |
| 拇指 + 无名指捏合 | 右键点击 |
| 食指 + 中指 + 无名指伸出并上下移动 | 鼠标滚轮 |
| 握拳 | 暂停 / 恢复手势控制 |
| 按 `q` | 退出程序 |

## 快速开始

推荐使用 Python 3.10、3.11 或 3.12。Windows 体验通常最好，因为 PyAutoGUI 的鼠标控制权限最直接。

```powershell
git clone https://github.com/<你的用户名>/AirGestureMouse.git
cd AirGestureMouse

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e .
air-gesture-mouse
```

也可以不安装包，直接从项目根目录运行：

```powershell
python run.py
```

## 摄像头与权限

运行后会打开一个小型 OpenCV 预览窗口，并默认放在屏幕左下角。请确保：

- 摄像头没有被其他软件占用。
- 系统允许 Python / 终端访问摄像头。
- macOS 需要在系统设置中允许终端控制鼠标和访问摄像头。
- PyAutoGUI 默认开启 failsafe，鼠标快速移动到屏幕左上角会触发安全中断。

## 项目结构

```text
AirGestureMouse/
├── src/
│   └── air_gesture_mouse/
│       ├── app.py          # 主应用循环：摄像头、MediaPipe、OpenCV 窗口
│       ├── config.py       # 可调参数：摄像头、窗口、控制区、手势阈值
│       ├── geometry.py     # 坐标钳制、距离计算、控制区边界
│       ├── gestures.py     # 手指伸缩、握拳、三指滚轮、捏合比例
│       └── mouse.py        # 鼠标移动、点击、滚轮封装
├── examples/
│   └── prototypes/         # 项目早期编号测试脚本
├── tests/                  # 不依赖摄像头的纯逻辑测试
├── run.py                  # 免安装运行入口
├── pyproject.toml          # 包元数据与命令行入口
├── requirements.txt        # 运行依赖
└── README.md
```

## 调参入口

核心参数集中在 `src/air_gesture_mouse/config.py`：

- `ControlAreaConfig`：控制框大小和垂直偏移。
- `GestureConfig.smoothing`：鼠标移动平滑系数，越大越灵敏。
- `left_pinch_trigger_ratio` / `right_pinch_trigger_ratio`：捏合触发阈值。
- `pinch_confirm_frames`：点击需要连续确认的帧数。
- `scroll_trigger_delta` / `scroll_speed`：滚轮触发距离和速度。
- `fist_confirm_frames`：握拳暂停需要连续确认的帧数。

如果误触较多，可以提高 `pinch_confirm_frames` 或降低捏合触发阈值；如果移动延迟明显，可以适当提高 `smoothing`。

## 测试

当前测试覆盖坐标边界计算和手势纯逻辑，不需要摄像头即可运行：

```powershell
python -m unittest discover -s tests
```

## 开发路线

- 添加配置文件，让用户无需改代码即可保存自己的参数。
- 增加演示 GIF 和手势示意图，降低新用户理解成本。
- 支持多摄像头选择和命令行参数。
- 扩展更多手势场景的自动化测试。
- 增加拖拽、双击、演示翻页等扩展手势。

## 贡献

欢迎提交 Issue 和 Pull Request。比较适合贡献的方向包括：不同系统的兼容性测试、手势误触优化、文档截图、演示视频、更多交互手势。

## License

本项目基于 MIT License 开源。
