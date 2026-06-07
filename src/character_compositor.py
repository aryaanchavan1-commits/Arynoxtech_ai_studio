import cv2
import numpy as np
from pathlib import Path
import config
from src.wav2lip.face_detection import FaceDetector
from src.theme_manager import load_background


_face_detector = FaceDetector()


def _remove_bg(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]

    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)

    rect = (3, 3, w - 6, h - 6)
    cv2.grabCut(image, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)

    face = _face_detector.get_largest_face(image)
    if face:
        fx, fy, fw, fh = face["bbox"]
        cx, cy = fx + fw // 2, fy + fh // 2
        body_r = int(max(fw, fh) * 3.5)
        cv2.circle(mask, (cx, cy), body_r, cv2.GC_FGD, -1)

    mask2 = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
    mask2 = cv2.GaussianBlur(mask2, (9, 9), 2)

    rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask2
    return rgba


def _composite(rgba_char: np.ndarray, bg: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    scaled = cv2.resize(rgba_char, (w, h), interpolation=cv2.INTER_LANCZOS4)
    alpha = scaled[:, :, 3].astype(np.float32) / 255.0
    result = bg.copy()

    y1, y2 = max(0, y), min(bg.shape[0], y + h)
    x1, x2 = max(0, x), min(bg.shape[1], x + w)
    sy1 = y1 - y if y < 0 else 0
    sx1 = x1 - x if x < 0 else 0

    cy = min(h, bg.shape[0] - y)
    cx = min(w, bg.shape[1] - x)

    region = result[y1:y1 + cy, x1:x1 + cx]
    char_region = scaled[sy1:sy1 + cy, sx1:sx1 + cx]
    alpha_region = alpha[sy1:sy1 + cy, sx1:sx1 + cx, np.newaxis]

    blended = (1 - alpha_region) * region.astype(np.float32) + alpha_region * char_region[:, :, :3].astype(np.float32)
    result[y1:y1 + cy, x1:x1 + cx] = blended.astype(np.uint8)
    return result


def prepare_character_scene(
    input_path: str,
    output_path: str,
    canvas_w: int = 1920,
    canvas_h: int = 1080,
    theme_id: str = "blue",
) -> str | None:
    img = cv2.imread(input_path)
    if img is None:
        return None

    orig_h, orig_w = img.shape[:2]

    max_char_h = int(canvas_h * 0.72)
    char_h = min(orig_h, max_char_h)
    scale = char_h / orig_h
    char_w = int(orig_w * scale)

    img_scaled = cv2.resize(img, (char_w, char_h))

    rgba = _remove_bg(img_scaled)

    bg = load_background(theme_id, canvas_w, canvas_h)

    pip_w = int(canvas_w * 0.30)
    aspect = char_h / char_w
    disp_w = pip_w
    disp_h = int(disp_w * aspect)
    if disp_h > canvas_h * 0.65:
        disp_h = int(canvas_h * 0.65)
        disp_w = int(disp_h / aspect)

    disp_x = canvas_w - disp_w - 70
    disp_y = canvas_h - disp_h - 130

    composed = _composite(rgba, bg, disp_x, disp_y, disp_w, disp_h)

    desk_h = 130
    desk_y = canvas_h - desk_h
    overlay = composed[desk_y:canvas_h].copy()
    desk = np.full_like(overlay, (35, 38, 55))
    blended = cv2.addWeighted(overlay, 0.5, desk, 0.5, 0)
    composed[desk_y:canvas_h] = blended
    cv2.line(composed, (0, desk_y), (canvas_w, desk_y), (0, 120, 215), 3)

    cv2.imwrite(output_path, composed)
    return output_path if Path(output_path).exists() else None
