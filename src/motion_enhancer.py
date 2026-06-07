import cv2
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
import config
from src.wav2lip.face_detection import FaceDetector


_face_detector = FaceDetector()
_blink_cache = {}


def _smooth_noise(length: int, sigma: float = 4.0) -> np.ndarray:
    raw = np.random.randn(length) * 0.5
    return gaussian_filter1d(raw, sigma=sigma, mode="wrap")


def _generate_motion(fps: int, total_frames: int) -> dict:
    t = np.arange(total_frames)
    phase = 2 * np.pi * (t / fps)

    # Natural head movement: multi-frequency biological model
    # Low freq = voluntary movement, mid = drift, high = tremor
    head_yaw = (
        _smooth_noise(total_frames, sigma=fps * 0.6) * 1.0
        + np.sin(phase * 0.15) * 0.3
        + np.sin(phase * 0.37) * 0.15
    )
    head_pitch = (
        _smooth_noise(total_frames, sigma=fps * 0.5) * 0.8
        + np.sin(phase * 0.12) * 0.2
    )
    head_roll = (
        _smooth_noise(total_frames, sigma=fps * 0.4) * 0.6
        + np.sin(phase * 0.22) * 0.1
    )

    # Breathing: two-component model (chest + diaphragm)
    breath_chest = 1.0 + np.sin(phase * 0.2) * 0.003
    breath_shoulder = 1.0 + np.sin(phase * 0.2 + 0.5) * 0.004

    # Micro-expressions: subtle random facial changes
    micro_smile = _smooth_noise(total_frames, sigma=fps * 0.8) * 0.15
    micro_eyebrow = _smooth_noise(total_frames, sigma=fps * 0.6) * 0.1

    return {
        "rot": head_roll,
        "tx": head_yaw,
        "ty": head_pitch,
        "scale": 1.0 + _smooth_noise(total_frames, sigma=fps * 0.5) * 0.008,
        "breath_chest": breath_chest,
        "breath_shoulder": breath_shoulder,
        "micro_smile": micro_smile,
        "micro_eyebrow": micro_eyebrow,
    }


def _detect_eyes(gray: np.ndarray, face_rect) -> tuple:
    fx, fy, fw, fh = face_rect
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )
    face_gray = gray[fy:fy + fh, fx:fx + fw]
    eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 3, minSize=(int(fw * 0.1), int(fh * 0.05)))
    return [(fx + ex, fy + ey, ew, eh) for (ex, ey, ew, eh) in eyes]


def _make_closed_eye(frame: np.ndarray, eye_rect, blink_progress: float = 1.0) -> np.ndarray:
    ex, ey, ew, eh = eye_rect
    region = frame[ey:ey + eh, ex:ex + ew].copy()
    skin = np.median(region.reshape(-1, 3), axis=0).astype(np.uint8)
    result = np.full_like(region, skin)
    # Natural blink: eyelid closes from top and bottom
    close_amount = int(eh * 0.45 * blink_progress)
    if close_amount > 0:
        cv2.rectangle(result, (0, 0), (ew, close_amount), (30, 20, 10), -1)
        cv2.rectangle(result, (0, eh - close_amount), (ew, eh), (30, 20, 10), -1)
    # Eyelid crease line
    line_y = eh // 2
    cv2.line(result, (0, line_y), (ew, line_y), (25, 18, 8), 1)
    return result


def _apply_blink(frame: np.ndarray, frame_idx: int, eye_rects: list, blink_schedule: list) -> np.ndarray:
    result = frame.copy()
    for blink_start, blink_end, duration in blink_schedule:
        if blink_start <= frame_idx < blink_end:
            rel = (frame_idx - blink_start) / duration
            # Fast close (0-0.3), slow open (0.3-1.0)
            progress = rel / 0.3 if rel < 0.3 else (1.0 - (rel - 0.3) / 0.7)
            progress = max(0.0, min(1.0, progress))
            for ex, ey, ew, eh in eye_rects:
                closed = _make_closed_eye(frame, (ex, ey, ew, eh), progress)
                # Feather edges
                mask = np.zeros((eh, ew), dtype=np.float32)
                cv2.circle(mask, (ew // 2, eh // 2), min(ew, eh) // 2, 1.0, -1)
                mask = cv2.GaussianBlur(mask, (5, 5), 2)
                for c in range(3):
                    result[ey:ey + eh, ex:ex + ew, c] = (
                        closed[:, :, c] * mask + frame[ey:ey + eh, ex:ex + ew, c] * (1 - mask)
                    ).astype(np.uint8)
            return result
    return frame


def enhance_motion(
    frames: list,
    fps: int = 25,
    enable_head_move: bool = True,
    enable_blink: bool = True,
    enable_breath: bool = True,
) -> list:
    if not frames:
        return frames

    total = len(frames)
    h, w = frames[0].shape[:2]
    motion = _generate_motion(fps, total)

    blink_schedule = []
    if enable_blink:
        np.random.seed(42)
        blink_times = np.cumsum(np.random.exponential(3.5, size=total // 100 + 15) * fps)
        blink_times = blink_times[blink_times < total - 8]
        for bt in blink_times.astype(int):
            blink_frames = max(3, int(fps * 0.08))
            blink_schedule.append((bt, bt + blink_frames, blink_frames))

    eye_rects = []
    if enable_blink:
        gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        face = _face_detector.get_largest_face(frames[0])
        if face:
            eye_rects = _detect_eyes(gray, face["bbox"])

    # Sharpen kernel for final pass
    sharpen_kernel = np.array([[-0.5, -0.5, -0.5],
                               [-0.5,  5.0, -0.5],
                               [-0.5, -0.5, -0.5]]) * 0.3

    out = []
    for i in range(total):
        frame = frames[i].copy()

        # Breathing: vertical stretch for chest
        if enable_breath:
            b = motion["breath_chest"][i]
            if abs(b - 1.0) > 0.0005:
                M = np.float32([[1, 0, 0], [0, b, 0]])
                frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # Micro-expression: subtle smile pull at mouth corners
        if abs(motion["micro_smile"][i]) > 0.05:
            sm = motion["micro_smile"][i]
            cx_mouth = w // 2
            cy_mouth = int(h * 0.6)
            # Apply subtle warping for smile
            sm_map_x, sm_map_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            dist = np.sqrt((sm_map_x - cx_mouth) ** 2 + (sm_map_y - cy_mouth) ** 2)
            influence = np.exp(-dist / (w * 0.15))
            sm_map_y = sm_map_y + influence * sm * 1.5
            frame = cv2.remap(frame, sm_map_x, sm_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        # Head movement (applied after micro-expressions)
        if enable_head_move:
            cx, cy = w // 2, int(h * 0.3)
            M = cv2.getRotationMatrix2D((cx, cy), motion["rot"][i], motion["scale"][i])
            M[0, 2] += motion["tx"][i]
            M[1, 2] += motion["ty"][i]
            frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # Blinking
        if enable_blink and eye_rects:
            frame = _apply_blink(frame, i, eye_rects, blink_schedule)

        # Subtle sharpening and contrast
        frame = cv2.filter2D(frame, -1, sharpen_kernel)
        frame = np.clip(frame, 0, 255).astype(np.uint8)

        out.append(frame)

    return out


def enhance_video(
    input_path: str,
    output_path: str,
    fps: int = 25,
    on_progress=None,
) -> str | None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25

    if total > 300:
        if on_progress:
            on_progress(0, f"Too many frames ({total}), skipping motion enhancement...")
        cap.release()
        return None

    frames = []
    for i in range(total):
        ret, f = cap.read()
        if ret:
            frames.append(f)
        if on_progress and i % 50 == 0:
            on_progress(int(i / total * 40), "Reading frames...")
    cap.release()

    if not frames:
        return None

    if on_progress:
        on_progress(40, "Applying motion enhancement...")

    enhanced = enhance_motion(frames, fps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    for i, f in enumerate(enhanced):
        writer.write(f)
        if on_progress and i % 50 == 0:
            on_progress(40 + int(60 * i / total), f"Writing frame {i}/{total}")

    writer.release()
    return str(output_path) if Path(output_path).exists() else None
