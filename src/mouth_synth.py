import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.vision import RunningMode
from pathlib import Path
import config


INNER_LIP = np.array([
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
    324, 318, 317, 14, 87, 178, 88, 95
])


class FaceLandmarkDetector:
    def __init__(self):
        model_path = Path(config.MODELS_DIR) / "mediapipe" / "face_landmarker.task"
        if not model_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            import requests
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            print("Downloading face landmarker model...")
            r = requests.get(url, stream=True, timeout=120)
            if r.status_code != 200:
                raise RuntimeError(f"Failed to download face_landmarker.task: {r.status_code}")
            with open(str(model_path), "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        options = FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.IMAGE,
            output_face_blendshapes=False,
            num_faces=1,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

    def detect(self, image_rgb: np.ndarray):
        h, w = image_rgb.shape[:2]
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self.landmarker.detect(mp_img)
        if not result.face_landmarks:
            return None
        lm = result.face_landmarks[0]
        pts = np.array([(lm[i].x * w, lm[i].y * h) for i in range(len(lm))], dtype=np.float32)
        return pts


class MouthSynthesizer:
    def __init__(self, max_darken=0.85, smooth_alpha=0.5):
        self.landmark_detector = FaceLandmarkDetector()
        self.max_darken = max_darken
        self._smooth_alpha = smooth_alpha
        self._smooth_openness = 0.0
        self._inner_pts_96 = None

    def initialize(self, image_rgb: np.ndarray, face_bbox):
        x1, y1, x2, y2 = face_bbox
        pts = self.landmark_detector.detect(image_rgb)
        if pts is None:
            raise ValueError("No face landmarks detected")

        inner = pts[INNER_LIP]

        self._inner_pts_96 = (inner - np.array([x1, y1])) * np.array([96.0 / (x2 - x1), 96.0 / (y2 - y1)])

    def synthesize(self, face_96: np.ndarray, energy: float, min_open: float = 0.0, max_open: float = 0.15):
        if self._inner_pts_96 is None:
            return face_96

        openness = np.clip((energy - min_open) / (max_open - min_open + 1e-8), 0.0, 1.0)
        openness = openness ** 0.5
        self._smooth_openness = self._smooth_openness * self._smooth_alpha + openness * (1.0 - self._smooth_alpha)
        openness = self._smooth_openness

        if openness < 0.02:
            return face_96

        h, w = 96, 96
        pts = np.clip(self._inner_pts_96, 0, [w - 1, h - 1]).astype(np.int32)
        contour = pts.reshape(-1, 1, 2)

        area = cv2.contourArea(contour)
        if area < 3.0:
            return face_96

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)
        mask_f = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.0) / 255.0

        alpha = mask_f * openness * self.max_darken

        dark_val = np.array([45.0, 30.0, 25.0], dtype=np.float32)
        result = face_96.astype(np.float32)
        for c in range(3):
            result[:, :, c] = result[:, :, c] * (1.0 - alpha) + dark_val[c] * alpha

        dilate_k = max(1, int(openness * 4))
        if dilate_k > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
            lip_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
            lip_mask = lip_mask.astype(bool) & ~mask.astype(bool)
            if lip_mask.any():
                lip_alpha = openness * 0.25
                for c in range(3):
                    result[:, :, c] = np.where(lip_mask,
                        np.clip(result[:, :, c] * (1.0 + lip_alpha), 0, 255),
                        result[:, :, c])

        return np.clip(result, 0, 255).astype(np.uint8)
