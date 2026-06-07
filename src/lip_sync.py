import cv2
import torch
import numpy as np
from pathlib import Path
import config
from src.wav2lip.models import load_wav2lip
from src.wav2lip.audio import Wav2LipAudio
from src.wav2lip.face_detection import FaceDetector, align_face, get_face_bbox
from src.utils import resolve_device
from src.mouth_synth import MouthSynthesizer


class LipSyncEngine:
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.face_detector = FaceDetector()
        self._load_model()

    def _load_model(self):
        if not config.WAV2LIP_MODEL_PATH.exists():
            from src.utils import download_with_fallback
            print("Wav2Lip model not found. Attempting download from mirrors...")
            ok = download_with_fallback(
                config.WAV2LIP_MODEL_URLS,
                config.WAV2LIP_MODEL_PATH,
                desc="Wav2Lip GAN",
                min_size=100e6,
            )
            if not ok:
                raise FileNotFoundError(
                    f"Wav2Lip model not found at {config.WAV2LIP_MODEL_PATH}. "
                    "Download manually from one of:\n" +
                    "\n".join(f"  {u}" for u in config.WAV2LIP_MODEL_URLS)
                )
        self.model = load_wav2lip(str(config.WAV2LIP_MODEL_PATH), self.device)

    def _process_frame(self, face_prev: np.ndarray, face_current: np.ndarray, mel_chunk: np.ndarray, lip_boost: float = 1.5) -> np.ndarray:
        face_stacked = np.concatenate([face_prev, face_current], axis=-1)
        face_tensor = torch.tensor(face_stacked / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

        mel_tensor = torch.tensor(mel_chunk.T, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        face_tensor = face_tensor.to(self.device)
        mel_tensor = mel_tensor.to(self.device)

        with torch.no_grad():
            pred = self.model(mel_tensor, face_tensor)

        pred = pred.cpu().squeeze(0).permute(1, 2, 0).numpy()
        pred = np.clip(pred, 0.0, 1.0)
        pred_img = (pred * 255).astype(np.uint8)

        if lip_boost > 1.0 and abs(lip_boost - 1.0) > 0.01:
            face_float = face_current.astype(np.float32)
            pred_float = pred * 255.0
            diff = pred_float - face_float
            mouth_mask = self._get_mouth_mask(face_current.shape[1], face_current.shape[0])
            gain = mouth_mask[:, :, np.newaxis] * (lip_boost - 1.0) + (1.0 - mouth_mask[:, :, np.newaxis]) * 0.3
            amplified = np.clip(face_float + diff * gain, 0, 255).astype(np.uint8)
            return amplified

        return pred_img

    def _init_mouth_synth(self, image_rgb, face_bbox):
        try:
            self._mouth_synth = MouthSynthesizer()
            self._mouth_synth.initialize(image_rgb, face_bbox)
        except Exception as e:
            print(f"WARNING: Mouth synth init failed: {e}")
            self._mouth_synth = None

    def _get_mouth_mask(self, w: int, h: int) -> np.ndarray:
        ys, xs = np.mgrid[0:h, 0:w]
        mouth_cx, mouth_cy = w // 2, int(h * 0.62)
        rx, ry = int(w * 0.40), int(h * 0.18)
        dx = (xs - mouth_cx) / rx
        dy = (ys - mouth_cy) / ry
        d = dx * dx + dy * dy
        return np.maximum(1.0 - np.minimum(d, 1.0), 0.0, dtype=np.float32)

    def generate(self, input_image_path: str, audio_path: str, output_path: str) -> str:
        img = cv2.imread(input_image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {input_image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        mel_chunks, full_mel = Wav2LipAudio.get_mel_chunks(audio_path)
        if not mel_chunks:
            raise ValueError("No audio chunks generated - audio too short?")

        wav = Wav2LipAudio.load_and_preprocess(audio_path)
        frame_samples = int(config.WAV2LIP_FPS)
        hop = len(wav) // max(1, len(mel_chunks))
        energies = []
        for i in range(len(mel_chunks)):
            start = i * hop
            end = min(start + hop * 2, len(wav))
            chunk = wav[start:end]
            rms = float(np.sqrt(np.mean(chunk**2))) if len(chunk) > 0 else 0.0
            energies.append(rms)
        energies = np.array(energies, dtype=np.float32)
        e_min, e_max = energies.min(), energies.max()
        if e_max > e_min:
            energies = (energies - e_min) / (e_max - e_min)
        else:
            energies = np.zeros_like(energies)
        energy_smooth = np.convolve(energies, np.ones(7)/7, mode='same')

        face = self.face_detector.get_largest_face(img)
        if face is None:
            h, w = img.shape[:2]
            cx, cy = w // 2, h // 3
            x1, y1, x2, y2 = (cx - w // 4, cy - h // 4, cx + w // 4, cy + h // 4)
            print(f"WARNING: No face detected. Using center crop ({x1},{y1})-({x2},{y2})")
        else:
            x1, y1, x2, y2 = get_face_bbox(img, face)
            print(f"Face detected at ({x1},{y1})-({x2},{y2})")

        face_roi = img[y1:y2, x1:x2]
        if face_roi.size == 0:
            raise ValueError("Empty face region detected")

        self._init_mouth_synth(img, (x1, y1, x2, y2))

        temp_frames_dir = config.OUTPUT_DIR / "temp_frames"
        temp_frames_dir.mkdir(parents=True, exist_ok=True)

        fps = config.WAV2LIP_FPS
        out_path_obj = Path(output_path)
        out_path_obj.parent.mkdir(parents=True, exist_ok=True)

        temp_video = temp_frames_dir / "temp_no_audio.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(temp_video), fourcc, fps, (orig_w, orig_h))

        total = len(mel_chunks)
        background = img.copy()
        h, w = face_roi.shape[:2]
        face_static_96 = cv2.resize(face_roi, (config.WAV2LIP_IMG_SIZE, config.WAV2LIP_IMG_SIZE))
        face_prev_96 = face_static_96.copy()

        raw_preds_96 = []
        for i, mel_chunk in enumerate(mel_chunks):
            pred_face_96 = self._process_frame(face_prev_96, face_static_96, mel_chunk, lip_boost=2.5)
            raw_preds_96.append(pred_face_96)

            decay = max(0.25, 0.7 * (1.0 - i / total * 0.5))
            face_prev_96 = cv2.addWeighted(pred_face_96, decay, face_static_96, 1.0 - decay, 0)
            yield i + 1, total

        from src.temporal_attention import temporal_refine
        all_preds_96 = temporal_refine(raw_preds_96)

        for i, pred_face_96 in enumerate(all_preds_96):
            if self._mouth_synth is not None:
                pred_face_96 = self._mouth_synth.synthesize(pred_face_96, energy_smooth[min(i, len(energy_smooth)-1)], min_open=0.0, max_open=1.0)
            pred_face_resized = cv2.resize(pred_face_96, (w, h))
            result = background.copy()
            result[y1:y2, x1:x2] = pred_face_resized
            writer.write(cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

        writer.release()

        import subprocess
        cmd = [
            "ffmpeg", "-y",
            "-i", str(temp_video),
            "-i", audio_path,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-shortest",
            str(out_path_obj),
        ]
        subprocess.run(cmd, capture_output=True, check=True)

        import time
        for f in temp_frames_dir.glob("*"):
            for _ in range(3):
                try:
                    f.unlink()
                    break
                except PermissionError:
                    time.sleep(0.1)
        try:
            temp_frames_dir.rmdir()
        except Exception:
            pass
        if temp_video.exists():
            for _ in range(3):
                try:
                    temp_video.unlink()
                    break
                except PermissionError:
                    time.sleep(0.1)

        return str(out_path_obj)


def preprocess_image(image_path: str) -> str:
    img = cv2.imread(image_path)
    if img is None:
        return image_path
    h, w = img.shape[:2]
    max_dim = 1920
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h))
        cv2.imwrite(image_path, img)
    return image_path
