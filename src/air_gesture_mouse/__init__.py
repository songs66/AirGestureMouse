"""AirGestureMouse package."""

__version__ = "0.1.0"
__all__ = ["AppConfig", "GestureMouseApp", "main"]


def __getattr__(name: str):
    if name == "AppConfig":
        from .config import AppConfig

        return AppConfig

    if name in {"GestureMouseApp", "main"}:
        from .app import GestureMouseApp, main

        return {"GestureMouseApp": GestureMouseApp, "main": main}[name]

    raise AttributeError(f"module 'air_gesture_mouse' has no attribute {name!r}")
