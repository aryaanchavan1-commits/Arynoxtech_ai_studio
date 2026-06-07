import cv2
import numpy as np
from pathlib import Path
import config


BUILTIN_THEMES = {
    "blue": {"name": "Blue Studio", "type": "builtin"},
    "red": {"name": "Red Studio", "type": "builtin"},
    "dark": {"name": "Dark Studio", "type": "builtin"},
}


def list_themes() -> dict:
    themes = dict(BUILTIN_THEMES)
    themes_dir = config.THEMES_DIR
    if themes_dir.exists():
        for f in sorted(themes_dir.iterdir()):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                themes[f.name] = {"name": f.stem.replace("_", " ").title(), "type": "image", "path": str(f)}
    return themes


def load_background(theme_id: str, width: int = 1920, height: int = 1080) -> np.ndarray:
    themes = list_themes()
    if theme_id not in themes:
        theme_id = "blue"

    info = themes[theme_id]
    if info["type"] == "image":
        img = cv2.imread(info["path"])
        if img is not None:
            return cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)

    return _generate_builtin(theme_id, width, height)


def _generate_builtin(theme_id: str, w: int, h: int) -> np.ndarray:
    bg = np.zeros((h, w, 3), dtype=np.uint8)

    base = {"blue": (50, 22, 14), "red": (30, 25, 25), "dark": (18, 18, 20)}
    accent = {"blue": (215, 120, 0), "red": (50, 50, 200), "dark": (180, 180, 180)}
    top_r, top_g, top_b = base.get(theme_id, base["blue"])
    ar, ag, ab = accent.get(theme_id, accent["blue"])

    for y in range(h):
        t = y / h
        r = int(top_r + 12 * t)
        g = int(top_g + 18 * t)
        b = int(top_b + 30 * t)
        cv2.line(bg, (0, y), (w, y), (min(b, 255), min(g, 255), min(r, 255)), 1)

    desk_h = 130
    desk_y = h - desk_h
    overlay = bg[desk_y:h].copy()
    desk = np.full_like(overlay, (35, 38, 55))
    bg[desk_y:h] = cv2.addWeighted(overlay, 0.5, desk, 0.5, 0)
    cv2.line(bg, (0, desk_y), (w, desk_y), (ab, ag, ar), 3)

    cx, cy = w // 2, h // 2
    for r in [400, 600, 800]:
        cv2.circle(bg, (w - 250, 220), r, (40, 30, 20), 1)

    for y_pos in range(h // 4, h, h // 4):
        cv2.line(bg, (0, y_pos), (w, y_pos), (60, 50, 40), 1)

    cv2.rectangle(bg, (0, 0), (5, h), (ab, ag, ar), -1)
    return bg


def get_ken_burns_frame(bg: np.ndarray, frame_idx: int, total_frames: int) -> np.ndarray:
    if total_frames <= 1:
        return bg

    progress = frame_idx / max(total_frames - 1, 1)
    zoom_start, zoom_end = 1.0, 1.04
    zoom = zoom_start + (zoom_end - zoom_start) * progress

    h, w = bg.shape[:2]
    new_w = int(w * zoom)
    new_h = int(h * zoom)

    pan_x = int((new_w - w) * 0.3 * progress)
    pan_y = int((new_h - h) * 0.15 * progress)

    scaled = cv2.resize(bg, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    frame = scaled[pan_y:pan_y + h, pan_x:pan_x + w]
    return frame


def add_lighting_overlay(frame: np.ndarray, frame_idx: int, fps: int = 25) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = np.zeros_like(frame, dtype=np.float32)

    phase = (frame_idx / fps) * 2 * np.pi * 0.05
    cx = w // 2 + int(np.sin(phase) * 30)
    cy = h // 3 + int(np.cos(phase * 0.7) * 15)

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    glow = np.exp(-dist / (w * 0.35))
    glow = (glow * 0.04 * (1 + 0.3 * np.sin(phase * 1.5))).clip(0, 0.06)

    for c in range(3):
        overlay[:, :, c] = glow * 255

    result = cv2.addWeighted(frame, 1.0, overlay.astype(np.uint8), 1.0, 0)
    return result


def apply_vignette(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    kernel_x = cv2.getGaussianKernel(w, w * 0.4)
    kernel_y = cv2.getGaussianKernel(h, h * 0.4)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    mask = np.clip(1 - (1 - mask) * 0.5, 0.5, 1.0)

    for c in range(3):
        frame[:, :, c] = (frame[:, :, c].astype(np.float32) * mask).astype(np.uint8)
    return frame
