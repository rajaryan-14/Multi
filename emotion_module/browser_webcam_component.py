from pathlib import Path

import streamlit.components.v1 as components


frontend_dir = (Path(__file__).parent / "browser_webcam_frontend").absolute()
_component_func = components.declare_component("browser_webcam", path=str(frontend_dir))


def browser_webcam(
    *,
    key: str,
    overlay: dict | None = None,
    width: int = 480,
    height: int = 360,
    capture_interval_ms: int = 450,
    jpeg_quality: float = 0.6,
    facing_mode: str = "user",
):
    return _component_func(
        key=key,
        overlay=overlay or {},
        width=width,
        height=height,
        capture_interval_ms=capture_interval_ms,
        jpeg_quality=jpeg_quality,
        facing_mode=facing_mode,
        default=None,
    )
