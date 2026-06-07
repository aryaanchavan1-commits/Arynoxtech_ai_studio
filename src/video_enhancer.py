import cv2
import numpy as np
from pathlib import Path
import subprocess
import config


class ColorGrader:
    @staticmethod
    def match_histograms(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
        matched = source_lab.copy()
        for i in range(3):
            src_ch = source_lab[:, :, i]
            tgt_ch = target_lab[:, :, i]
            src_hist, _ = np.histogram(src_ch, 256, [0, 256])
            tgt_hist, _ = np.histogram(tgt_ch, 256, [0, 256])
            src_cdf = src_hist.cumsum()
            src_cdf = (src_cdf / src_cdf[-1]) * 255
            tgt_cdf = tgt_hist.cumsum()
            tgt_cdf = (tgt_cdf / tgt_cdf[-1]) * 255
            mapping = np.zeros(256, dtype=np.uint8)
            for j in range(256):
                diff = np.abs(tgt_cdf - src_cdf[j])
                mapping[j] = np.argmin(diff)
            matched[:, :, i] = mapping[src_ch]
        return cv2.cvtColor(matched, cv2.COLOR_LAB2BGR)

    @staticmethod
    def adjust_gamma(frame: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)], dtype=np.uint8)
        return cv2.LUT(frame, table)

    @staticmethod
    def auto_contrast(frame: np.ndarray, clip_percent: float = 2.0) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_ch = lab[:, :, 0]
        low = np.percentile(l_ch, clip_percent)
        high = np.percentile(l_ch, 100 - clip_percent)
        if high > low:
            l_ch = np.clip((l_ch - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
            lab[:, :, 0] = l_ch
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def subtle_sharpen(frame: np.ndarray, amount: float = 0.3) -> np.ndarray:
        kernel = np.array([
            [-amount, -amount, -amount],
            [-amount, 1 + 4 * amount, -amount],
            [-amount, -amount, -amount],
        ])
        return cv2.filter2D(frame, -1, kernel)

    @staticmethod
    def denoise(frame: np.ndarray, strength: int = 3) -> np.ndarray:
        return cv2.fastNlMeansDenoisingColored(frame, None, strength, strength, 7, 21)


class VideoUpscaler:
    @staticmethod
    def upscale_frame(frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

    @staticmethod
    def upscale_video_ffmpeg(input_path: str, output_path: str, target_width: int, target_height: int,
                             fps: int = 25) -> str | None:
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-vf", f"scale={target_width}:{target_height}:flags=lanczos",
            "-c:v", "libx264", "-preset", "slow", "-crf", "16",
            "-c:a", "aac", "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            return str(output_path) if Path(output_path).exists() else None
        except subprocess.CalledProcessError as e:
            print(f"Upscale error: {e.stderr.decode() if e.stderr else ''}")
            return None

    @staticmethod
    def upscale_video_opencv(input_path: str, output_path: str, target_width: int, target_height: int,
                              fps: int = 25) -> str | None:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (target_width, target_height))
        for i in range(total):
            ret, frame = cap.read()
            if not ret:
                break
            upscaled = VideoUpscaler.upscale_frame(frame, target_width, target_height)
            writer.write(upscaled)
        cap.release()
        writer.release()
        return str(output_path) if Path(output_path).exists() else None


class QualityEnhancer:
    def __init__(self):
        self.color_grader = ColorGrader()
        self.upscaler = VideoUpscaler()

    def enhance_frame(self, frame: np.ndarray, reference: np.ndarray = None,
                      gamma: float = 1.0, sharpen: bool = True) -> np.ndarray:
        result = frame.copy()
        result = self.color_grader.auto_contrast(result)
        if reference is not None:
            result = self.color_grader.match_histograms(result, reference)
        result = self.color_grader.adjust_gamma(result, gamma)
        if sharpen:
            result = self.color_grader.subtle_sharpen(result)
        return result

    def enhance_video_pass(self, input_path: str, output_path: str, reference_path: str = None,
                           fps: int = 25, gamma: float = 1.0, upscale_to: tuple = None,
                           on_progress=None) -> str | None:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_fps = int(cap.get(cv2.CAP_PROP_FPS)) or fps
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        tgt_w, tgt_h = upscale_to or (src_w, src_h)
        ref_cap = cv2.VideoCapture(reference_path) if reference_path else None
        ref_total = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT)) if ref_cap else 0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, src_fps, (tgt_w, tgt_h))
        for i in range(total):
            ret, frame = cap.read()
            if not ret:
                break
            ref_frame = None
            if ref_cap:
                ref_ret, ref_frame = ref_cap.read()
                if not ref_ret:
                    ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ref_ret, ref_frame = ref_cap.read()
            enhanced = self.enhance_frame(frame, ref_frame, gamma)
            if upscale_to:
                enhanced = self.upscaler.upscale_frame(enhanced, tgt_w, tgt_h)
            writer.write(enhanced)
            if on_progress and i % 30 == 0:
                on_progress(int(95 * i / total), f"Enhancing: frame {i}/{total}")
        cap.release()
        if ref_cap:
            ref_cap.release()
        writer.release()
        if on_progress:
            on_progress(100, "Enhancement complete")
        return str(output_path) if Path(output_path).exists() else None


def combine_with_color_correction(
    main_video: str,
    bg_video: str,
    output_path: str,
    audio_path: str = None,
    target_width: int = 1920,
    target_height: int = 1080,
    fps: int = 25,
    on_progress=None,
) -> str | None:
    cap_main = cv2.VideoCapture(main_video)
    cap_bg = cv2.VideoCapture(bg_video)
    if not cap_main.isOpened() or not cap_bg.isOpened():
        return None
    main_total = int(cap_main.get(cv2.CAP_PROP_FRAME_COUNT))
    bg_total = int(cap_bg.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp = Path(output_path).with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp), fourcc, fps, (target_width, target_height))
    total = max(main_total, bg_total)
    for i in range(total):
        m_ret, m_frame = cap_main.read()
        if not m_ret:
            cap_main.set(cv2.CAP_PROP_POS_FRAMES, 0)
            m_ret, m_frame = cap_main.read()
        b_ret, b_frame = cap_bg.read()
        if not b_ret:
            cap_bg.set(cv2.CAP_PROP_POS_FRAMES, 0)
            b_ret, b_frame = cap_bg.read()
        m_frame = cv2.resize(m_frame, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        b_frame = cv2.resize(b_frame, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        graded = ColorGrader.match_histograms(m_frame, b_frame)
        graded = ColorGrader.subtle_sharpen(graded)
        blended = cv2.addWeighted(graded, 0.85, b_frame, 0.15, 0)
        writer.write(blended)
        if on_progress and i % 30 == 0:
            on_progress(int(90 * i / total), f"Compositing: frame {i}/{total}")
    cap_main.release()
    cap_bg.release()
    writer.release()
    if audio_path and Path(audio_path).exists():
        cmd = [
            "ffmpeg", "-y",
            "-i", str(tmp),
            "-i", audio_path,
            "-c:v", "libx264", "-preset", "slow", "-crf", "16",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            str(output_path),
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True)
        except subprocess.CalledProcessError:
            return None
        finally:
            if tmp.exists():
                tmp.unlink()
    else:
        if tmp.exists():
            tmp.replace(output_path)
    return str(output_path) if Path(output_path).exists() else None
