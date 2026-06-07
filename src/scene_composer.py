import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import subprocess
import os
import config
from datetime import datetime
import math
from src.theme_manager import load_background, get_ken_burns_frame, add_lighting_overlay, apply_vignette, BUILTIN_THEMES


THEMES = {
    "blue": {
        "accent_bgr": (215, 120, 0),
        "accent_rgb": (0, 120, 215),
        "lower_bg_rgb": (0, 70, 160),
        "ticker_bg_rgb": (20, 20, 30),
    },
    "red": {
        "accent_bgr": (50, 50, 200),
        "accent_rgb": (200, 50, 50),
        "lower_bg_rgb": (160, 30, 30),
        "ticker_bg_rgb": (20, 20, 30),
    },
    "dark": {
        "accent_bgr": (180, 180, 180),
        "accent_rgb": (180, 180, 180),
        "lower_bg_rgb": (30, 30, 40),
        "ticker_bg_rgb": (10, 10, 15),
    },
}


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "C:/Windows/Fonts/Nirmala.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _add_wave_animation(frame: np.ndarray, frame_idx: int, fps: int, theme: str) -> np.ndarray:
    h, w = frame.shape[:2]
    t = THEMES.get(theme, THEMES["blue"])
    accent = t["accent_bgr"]
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    time = frame_idx / fps
    num_bars = 48
    bar_w = w // num_bars
    base_amp = h * 0.04

    cx = w // 2
    cy = h // 2

    for i in range(num_bars):
        x = i * bar_w
        freq = 1.5 + 2.5 * (i / num_bars)
        phase_offset = 2.0 * math.sin(math.pi * i / num_bars)
        amp = base_amp * (1.0 + 0.6 * math.sin(time * freq + phase_offset) + 0.3 * math.sin(time * freq * 0.5 + phase_offset * 1.5))
        amp = max(2, int(amp))
        bar_h = int(amp)

        alpha = 0.08 + 0.05 * (0.5 + 0.5 * math.sin(time * freq * 0.7 + phase_offset))
        y_center = cy + int(20 * math.sin(time * 0.3 + i * 0.1))
        y_top = y_center - bar_h
        y_bot = y_center

        cv2.rectangle(overlay, (x + 1, y_top), (x + bar_w - 2, y_bot), accent, -1)

    result = cv2.addWeighted(frame, 1.0, overlay, 0.3, 0)

    accent_rgb = (accent[2], accent[1], accent[0])
    for i in range(num_bars):
        x = i * bar_w
        freq = 1.5 + 2.5 * (i / num_bars)
        phase_offset = 2.0 * math.sin(math.pi * i / num_bars)
        amp = base_amp * (1.0 + 0.6 * math.sin(time * freq + phase_offset) + 0.3 * math.sin(time * freq * 0.5 + phase_offset * 1.5))
        amp = max(2, int(amp))
        bar_h = int(amp)
        y_center = cy + int(20 * math.sin(time * 0.3 + i * 0.1))
        glow_alpha = 0.04 + 0.03 * (0.5 + 0.5 * math.sin(time * freq * 1.3 + phase_offset))
        if glow_alpha > 0.05:
            cv2.rectangle(result, (x + 3, y_center - bar_h), (x + bar_w - 4, y_center + 1), accent_rgb, 1)

    return result


def _add_scrolling_scanline(frame: np.ndarray, frame_idx: int, fps: int) -> np.ndarray:
    h, w = frame.shape[:2]
    phase = (frame_idx / fps) * 60
    scan_y = int((phase % h))
    overlay = frame.copy()
    alpha = np.zeros((h, 1), dtype=np.float32)
    for dy in range(-30, 31):
        y = scan_y + dy
        if 0 <= y < h:
            a = max(0, 1.0 - abs(dy) / 30)
            alpha[y, 0] = a * 0.03
    for c in range(3):
        overlay[:, :, c] = (overlay[:, :, c].astype(np.float32) * (1.0 - alpha) + 255 * alpha).astype(np.uint8)
    return overlay


class SceneComposer:
    def __init__(self, width: int = 1920, height: int = 1080):
        self.W = width
        self.H = height
        self.font_xs = _load_font(26)
        self.font_sm = _load_font(32)
        self.font_md = _load_font(52)
        self.font_lg = _load_font(80)
        self.font_xl = _load_font(110)

    def _pil(self, frame: np.ndarray) -> Image.Image:
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def _cv2(self, pil_img: Image.Image) -> np.ndarray:
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _get_bg(self, theme_id: str) -> np.ndarray:
        return load_background(theme_id, self.W, self.H)

    def _create_intro(self, show_name: str, channel: str, date_str: str, theme: str) -> np.ndarray:
        t = THEMES.get(theme, THEMES["blue"])
        frame = self._get_bg(theme)
        pil = self._pil(frame)
        draw = ImageDraw.Draw(pil)

        y0 = self.H // 3
        draw.text((self.W // 2, y0), show_name, fill=(255, 255, 255), font=self.font_xl, anchor="mm")
        line_y = y0 + 90
        draw.rectangle([(self.W // 2 - 250, line_y), (self.W // 2 + 250, line_y + 4)], fill=t["accent_rgb"])
        draw.text((self.W // 2, line_y + 60), channel, fill=(210, 210, 210), font=self.font_lg, anchor="mm")
        draw.text((self.W // 2, line_y + 130), date_str, fill=(150, 150, 150), font=self.font_sm, anchor="mm")
        draw.text((self.W // 2, self.H - 90), "YOUR TRUSTED NEWS SOURCE", fill=(100, 100, 100), font=self.font_xs, anchor="mm")

        return self._cv2(pil)

    def _create_outro(self, show_name: str, channel: str, theme: str) -> np.ndarray:
        frame = self._get_bg(theme)
        pil = self._pil(frame)
        draw = ImageDraw.Draw(pil)

        draw.text((self.W // 2, self.H // 2 - 90), "Thank You for Watching", fill=(255, 255, 255), font=self.font_lg, anchor="mm")
        draw.text((self.W // 2, self.H // 2 + 30), show_name, fill=(200, 200, 200), font=self.font_md, anchor="mm")
        draw.text((self.W // 2, self.H // 2 + 100), channel, fill=(150, 150, 150), font=self.font_sm, anchor="mm")

        return self._cv2(pil)

    def _add_lower_third(self, frame: np.ndarray, headline: str, anchor: str, theme: str) -> np.ndarray:
        t = THEMES.get(theme, THEMES["blue"])
        bar_h = 85
        bar_w = int(self.W * 0.55)
        bx, by = 50, self.H - 220

        sub = frame[by:by + bar_h, bx:bx + bar_w].copy()
        bar = np.full_like(sub, t["lower_bg_rgb"][::-1])
        blended = cv2.addWeighted(sub, 0.25, bar, 0.75, 0)
        frame[by:by + bar_h, bx:bx + bar_w] = blended

        cv2.rectangle(frame, (bx, by), (bx + 5, by + bar_h), t["accent_bgr"], -1)

        pil = self._pil(frame)
        draw = ImageDraw.Draw(pil)
        if anchor:
            draw.text((bx + 20, by + 6), anchor.upper(), fill=(190, 190, 190), font=self.font_xs)
            draw.text((bx + 20, by + 40), headline, fill=(255, 255, 255), font=self.font_sm)
        else:
            draw.text((bx + 20, by + 28), headline, fill=(255, 255, 255), font=self.font_sm)
        return self._cv2(pil)

    def _add_ticker(self, frame: np.ndarray, text: str, offset: int, theme: str) -> np.ndarray:
        t = THEMES.get(theme, THEMES["blue"])
        th = 48
        ty = self.H - th

        sub = frame[ty:self.H].copy()
        bg = np.full_like(sub, t["ticker_bg_rgb"][::-1])
        blended = cv2.addWeighted(sub, 0.15, bg, 0.85, 0)
        frame[ty:self.H] = blended

        cv2.line(frame, (0, ty), (self.W, ty), t["accent_bgr"], 2)

        sep = "    ●    "
        full = sep.join([text] * 4)
        pil = self._pil(frame)
        draw = ImageDraw.Draw(pil)
        tw = int(draw.textlength(full, font=self.font_xs))
        x = self.W - (offset % (self.W + tw // 2))
        draw.text((x, ty + 8), full, fill=(255, 255, 255), font=self.font_xs)
        return self._cv2(pil)

    def _composite_character_onto_bg(self, char_frame: np.ndarray, bg: np.ndarray) -> np.ndarray:
        char_h, char_w = char_frame.shape[:2]
        bg_h, bg_w = bg.shape[:2]

        disp_w = int(bg_w * 0.38)
        aspect = char_h / char_w
        disp_h = int(disp_w * aspect)
        if disp_h > bg_h * 0.7:
            disp_h = int(bg_h * 0.7)
            disp_w = int(disp_h / aspect)

        disp_x = bg_w - disp_w - 80
        disp_y = bg_h - disp_h - 140

        scaled = cv2.resize(char_frame, (disp_w, disp_h), interpolation=cv2.INTER_LANCZOS4)

        result = bg.copy()
        y1, y2 = disp_y, disp_y + disp_h
        x1, x2 = disp_x, disp_x + disp_w
        result[y1:y2, x1:x2] = scaled

        desk_h = 130
        desk_y = bg_h - desk_h
        overlay = result[desk_y:bg_h].copy()
        desk = np.full_like(overlay, (35, 38, 55))
        result[desk_y:bg_h] = cv2.addWeighted(overlay, 0.5, desk, 0.5, 0)
        t = THEMES.get("blue", THEMES["blue"])
        cv2.line(result, (0, desk_y), (bg_w, desk_y), t["accent_bgr"], 3)

        return result

    def compose(
        self,
        input_video: str,
        output_path: str,
        audio_path: str,
        anchor_name: str = "",
        channel_name: str = "AI News Studio",
        show_name: str = "The Daily Briefing",
        headlines: list = None,
        ticker_text: str = "",
        theme: str = "blue",
        enable_intro: bool = True,
        enable_outro: bool = True,
        enable_ticker: bool = True,
        enable_lower_third: bool = True,
        music_path: str = None,
        pip_composite: bool = False,
        on_progress=None,
    ) -> str | None:
        if not Path(input_video).exists():
            return None

        cap = cv2.VideoCapture(input_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return None

        headlines = headlines or []
        if not ticker_text and headlines:
            ticker_text = " ● ".join(headlines)

        date_str = datetime.now().strftime("%B %d, %Y")
        bg_image = self._get_bg(theme)
        intro_n = int(fps * 3) if enable_intro else 0
        outro_n = int(fps * 2.5) if enable_outro else 0
        out_total = intro_n + total + outro_n

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        tmp = Path(output_path).with_suffix(".tmp.mp4")
        writer = cv2.VideoWriter(str(tmp), fourcc, int(fps), (self.W, self.H))
        idx = 0

        def pct(): return int(min(100, idx / out_total * 100)) if out_total else 0
        def report(msg, p=None):
            if on_progress:
                on_progress(p if p is not None else pct(), msg)

        if enable_intro:
            for i in range(intro_n):
                alpha = min(1.0, (i + 1) / max(1, int(fps * 0.4)))
                f = self._create_intro(show_name, channel_name, date_str, theme)
                f = (f * alpha).astype(np.uint8)
                writer.write(f)
                idx += 1
                report("Composing: intro...")

        for i in range(total):
            ret, raw = cap.read()
            if not ret:
                break

            if pip_composite:
                frame = self._composite_character_onto_bg(raw, bg_image.copy())
            else:
                frame = raw.copy()

            frame = apply_vignette(add_lighting_overlay(frame, i, int(fps)))
            frame = _add_wave_animation(frame, i, int(fps), theme)
            if i % 2 == 0:
                frame = _add_scrolling_scanline(frame, i, int(fps))

            if enable_lower_third and headlines:
                hi = min(i // max(1, total // len(headlines)), len(headlines) - 1)
                frame = self._add_lower_third(frame, headlines[hi], anchor_name, theme)

            if enable_ticker and ticker_text:
                frame = self._add_ticker(frame, ticker_text, i * 2, theme)

            writer.write(frame)
            idx += 1
            report(f"Composing: frame {i + 1}/{total}")

        cap.release()

        if enable_outro:
            for i in range(outro_n):
                alpha = min(1.0, (i + 1) / max(1, int(fps * 0.4)))
                f = self._create_outro(show_name, channel_name, theme)
                f = (f * alpha).astype(np.uint8)
                writer.write(f)
                idx += 1
                report("Composing: outro...")

        writer.release()

        if music_path and Path(music_path).exists():
            cmd = [
                "ffmpeg", "-y",
                "-i", str(tmp),
                "-i", audio_path,
                "-i", music_path,
                "-filter_complex",
                "[1:a]volume=1.0[voice];[2:a]volume=0.12[music];[voice][music]amix=inputs=2:duration=first[a]",
                "-map", "0:v", "-map", "[a]",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                str(output_path),
            ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(tmp),
                "-i", audio_path,
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                str(output_path),
            ]

        try:
            subprocess.run(cmd, capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg mix error: {e.stderr.decode() if e.stderr else ''}")
            return None
        finally:
            if tmp.exists():
                for _ in range(3):
                    try:
                        os.unlink(str(tmp))
                        break
                    except PermissionError:
                        import time
                        time.sleep(0.1)

        report("Studio composition complete", 100)
        return str(output_path) if Path(output_path).exists() else None
