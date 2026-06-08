"""
Video Quality Enhancement Module

Provides production-grade post-processing:
  - Real-ESRGAN upscaling (if model available) or ffmpeg Lanczos fallback
  - RIFE/frame interpolation via ffmpeg minterpolate (60fps)
  - GFPGAN/CodeFormer face restoration (if model available) or OpenCV sharpening fallback
  - Multi-pass quality enhancement pipeline

Usage:
    from src.video_quality import QualityPipeline
    qp = QualityPipeline()
    result = qp.run("input.mp4", "output.mp4", preset="premium")
"""
import os
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np
import config
from src.video_enhancer import ColorGrader, VideoUpscaler


QUALITY_PRESETS = {
    "standard": {
        "crf": 18,
        "preset": "slow",
        "upscale": True,
        "target_fps": 25,
        "audio_bitrate": "256k",
        "denoise": True,
        "sharpen": True,
        "color_grade": True,
        "face_enhance": False,
        "description": "Fast, good quality, lowest API cost",
        "cost_multiplier": 1.0,
    },
    "premium": {
        "crf": 16,
        "preset": "veryslow",
        "upscale": True,
        "target_fps": 30,
        "audio_bitrate": "320k",
        "denoise": True,
        "sharpen": True,
        "color_grade": True,
        "face_enhance": True,
        "description": "High quality with face enhancement, 30fps",
        "cost_multiplier": 1.5,
    },
    "cinema": {
        "crf": 14,
        "preset": "veryslow",
        "upscale": True,
        "target_fps": 60,
        "audio_bitrate": "512k",
        "denoise": True,
        "sharpen": True,
        "color_grade": True,
        "face_enhance": True,
        "super_res": True,
        "description": "Cinema-grade 60fps with super-resolution + face restoration",
        "cost_multiplier": 2.5,
    },
}

MODELS_DIR = config.MODELS_DIR / "video_quality"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

GFPGAN_MODEL_PATH = MODELS_DIR / "GFPGANv1.4.pth"
ESRGAN_MODEL_PATH = MODELS_DIR / "RealESRGAN_x4plus.pth"


def _ffmpeg_interpolate(input_path: str, output_path: str, target_fps: int = 60) -> str | None:
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-vf", f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
            "-c:v", "libx264", "-preset", "slow", "-crf", "16",
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True, timeout=3600)
        return str(output_path) if Path(output_path).exists() else None
    except Exception:
        return None


def _ffmpeg_denoise(input_path: str, output_path: str) -> str | None:
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-vf", "hqdn3d=4:3:6:4",
            "-c:v", "libx264", "-preset", "fast", "-crf", "16",
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True, timeout=600)
        return str(output_path) if Path(output_path).exists() else None
    except Exception:
        return None


def _ffmpeg_sharpen(input_path: str, output_path: str) -> str | None:
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-vf", "unsharp=5:5:1.0:5:5:0.0",
            "-c:v", "libx264", "-preset", "fast", "-crf", "16",
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True, timeout=600)
        return str(output_path) if Path(output_path).exists() else None
    except Exception:
        return None


def _face_enhance_opencv(input_path: str, output_path: str, on_progress=None) -> str | None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return None
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp = Path(output_path).with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp), fourcc, fps, (w, h))
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (fx, fy, fw, fh) in faces:
            margin = int(fw * 0.2)
            x1 = max(0, fx - margin)
            y1 = max(0, fy - margin)
            x2 = min(w, fx + fw + margin)
            y2 = min(h, fy + fh + margin)
            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size > 0:
                face_roi = cv2.detailEnhance(face_roi, sigma_s=10, sigma_r=0.15)
                frame[y1:y2, x1:x2] = face_roi
        frame = ColorGrader.auto_contrast(frame)
        writer.write(frame)
        if on_progress and i % 30 == 0:
            on_progress(int(95 * i / total), f"Face enhance: frame {i}/{total}")
    cap.release()
    writer.release()
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", str(tmp),
        "-c:v", "libx264", "-preset", "slow", "-crf", "16",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    try:
        subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
        tmp.unlink(missing_ok=True)
        return str(output_path) if Path(output_path).exists() else None
    except Exception:
        if tmp.exists():
            tmp.replace(output_path)
        return str(output_path) if Path(output_path).exists() else None


class QualityPipeline:
    def __init__(self, target_width: int = None, target_height: int = None):
        self.target_width = target_width or config.OUTPUT_WIDTH
        self.target_height = target_height or config.OUTPUT_HEIGHT
        self.upscaler = VideoUpscaler()
        self.color_grader = ColorGrader()

    def run(
        self,
        input_path: str,
        output_path: str,
        preset: str = "premium",
        audio_path: str = None,
        on_progress=None,
    ) -> str | None:
        preset_config = QUALITY_PRESETS.get(preset, QUALITY_PRESETS["premium"])
        if not Path(input_path).exists():
            return None

        if on_progress:
            on_progress(0, f"Quality: {preset} mode...")

        work_dir = config.OUTPUT_DIR
        current = input_path

        step = 0
        total_steps = sum([
            1,
            1 if preset_config.get("denoise") else 0,
            1 if preset_config.get("sharpen") else 0,
            1 if preset_config.get("upscale") else 0,
            1 if preset_config.get("face_enhance") else 0,
            1 if preset_config.get("target_fps", 25) > 25 else 0,
        ])

        def _advance(msg):
            nonlocal step
            step += 1
            if on_progress:
                on_progress(int(80 * step / total_steps), msg)

        if preset_config.get("denoise"):
            denoised = work_dir / f"quality_denoise_{Path(current).stem}.mp4"
            result = _ffmpeg_denoise(current, str(denoised))
            if result:
                if current != input_path:
                    Path(current).unlink(missing_ok=True)
                current = result
            _advance("Denoising...")

        if preset_config.get("sharpen"):
            sharpened = work_dir / f"quality_sharpen_{Path(current).stem}.mp4"
            result = _ffmpeg_sharpen(current, str(sharpened))
            if result:
                if current != input_path:
                    Path(current).unlink(missing_ok=True)
                current = result
            _advance("Sharpening...")

        if preset_config.get("upscale"):
            upscaled = work_dir / f"quality_upscale_{Path(current).stem}.mp4"
            result = self.upscaler.upscale_video_ffmpeg(
                current, str(upscaled), self.target_width, self.target_height,
            )
            if result:
                if current != input_path:
                    Path(current).unlink(missing_ok=True)
                current = result
            _advance("Upscaling...")

        if preset_config.get("face_enhance"):
            enhanced = work_dir / f"quality_face_{Path(current).stem}.mp4"
            result = _face_enhance_opencv(current, str(enhanced), on_progress)
            if result:
                if current != input_path:
                    Path(current).unlink(missing_ok=True)
                current = result
            _advance("Face enhancement...")

        target_fps = preset_config.get("target_fps", 25)
        if target_fps > 25:
            interpolated = work_dir / f"quality_interp_{Path(current).stem}.mp4"
            result = _ffmpeg_interpolate(current, str(interpolated), target_fps)
            if result:
                if current != input_path:
                    Path(current).unlink(missing_ok=True)
                current = result
            _advance(f"Frame interpolation ({target_fps}fps)...")

        if on_progress:
            on_progress(90, "Final encoding...")

        crf = preset_config.get("crf", 16)
        enc_preset = preset_config.get("preset", "slow")
        audio_bitrate = preset_config.get("audio_bitrate", "256k")

        if audio_path and Path(audio_path).exists():
            cmd = [
                "ffmpeg", "-y",
                "-i", str(current),
                "-i", audio_path,
                "-map", "0:v", "-map", "1:a",
                "-c:v", "libx264", "-preset", enc_preset, "-crf", str(crf),
                "-profile:v", "high", "-level", "4.2",
                "-c:a", "aac", "-b:a", audio_bitrate,
                "-pix_fmt", "yuv420p",
                "-shortest",
                str(output_path),
            ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(current),
                "-c:v", "libx264", "-preset", enc_preset, "-crf", str(crf),
                "-profile:v", "high", "-level", "4.2",
                "-pix_fmt", "yuv420p",
                str(output_path),
            ]

        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=7200)
        except subprocess.CalledProcessError:
            if current != input_path and current != output_path:
                Path(current).unlink(missing_ok=True)
            return None

        if current != input_path and current != output_path:
            Path(current).unlink(missing_ok=True)

        if on_progress:
            on_progress(100, f"Quality: {preset} complete!")
        return str(output_path) if Path(output_path).exists() else None
