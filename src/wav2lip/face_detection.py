import cv2
import numpy as np


class FaceDetector:
    def __init__(self, min_detection_confidence: float = 0.5):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        self._min_neighbors = max(3, int(10 * (1 - min_detection_confidence)))

    def detect(self, image: np.ndarray) -> list[dict]:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        rects = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=self._min_neighbors, minSize=(30, 30)
        )
        h, w = image.shape[:2]
        faces = []
        for (x, y, bw, bh) in rects:
            faces.append({
                "bbox": (x, y, bw, bh),
                "score": 1.0,
                "keypoints": {},
            })
        return faces

    def get_largest_face(self, image: np.ndarray) -> dict | None:
        faces = self.detect(image)
        if not faces:
            return None
        return max(faces, key=lambda f: f["bbox"][2] * f["bbox"][3])


def align_face(image: np.ndarray, face: dict, target_size: int = 96, margin_ratio: float = 0.3) -> np.ndarray | None:
    h, w = image.shape[:2]
    x, y, bw, bh = face["bbox"]

    margin_x = int(bw * margin_ratio)
    margin_y = int(bh * margin_ratio)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(w, x + bw + margin_x)
    y2 = min(h, y + bh + margin_y)

    face_crop = image[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None

    face_crop = cv2.resize(face_crop, (target_size, target_size))
    return face_crop


def get_face_bbox(image: np.ndarray, face: dict, margin_ratio: float = 0.3) -> tuple:
    h, w = image.shape[:2]
    x, y, bw, bh = face["bbox"]
    margin_x = int(bw * margin_ratio)
    margin_y = int(bh * margin_ratio)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(w, x + bw + margin_x)
    y2 = min(h, y + bh + margin_y)
    return (x1, y1, x2, y2)
