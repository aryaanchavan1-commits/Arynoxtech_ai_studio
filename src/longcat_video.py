import json
import os
import subprocess
import sys
import tempfile
import time
import shutil
from pathlib import Path
import config


QUALITY_PRESETS = {
    "standard": {"resolution": "480p", "num_inference_steps": 50, "use_distill": False, "description": "480p standard"},
    "high": {"resolution": "720p", "num_inference_steps": 50, "use_distill": False, "description": "720p high"},
    "ultra": {"resolution": "720p", "num_inference_steps": 100, "use_distill": False, "description": "720p 100 steps"},
}


def longcat_available() -> bool:
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        return vram >= 16
    except Exception:
        return False


def longcat_vram_gb() -> float:
    try:
        import torch
        return torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:
        return 0.0


def longcat_weights_exist() -> bool:
    d = config.LONGCAT_WEIGHTS_DIR
    return d.exists() and any(d.iterdir())


def longcat_repo_cloned() -> bool:
    return (config.LONGCAT_REPO_DIR / "run_demo_avatar_single_audio_to_video.py").exists()


def longcat_setup_status() -> dict:
    return {
        "gpu_ready": longcat_available(),
        "vram_gb": longcat_vram_gb(),
        "repo_cloned": longcat_repo_cloned(),
        "weights_downloaded": longcat_weights_exist(),
        "complete": longcat_repo_cloned() and longcat_weights_exist(),
    }


def _auto_quality_preset() -> str:
    vram = longcat_vram_gb()
    return "ultra" if vram >= 22 else "high" if vram >= 16 else "standard"


def _run_with_retry(func, max_retries=3, retry_delay=5, on_progress=None, label=""):
    for attempt in range(1, max_retries + 1):
        if attempt > 1 and on_progress:
            on_progress(0, f"{label}: retry {attempt}/{max_retries}...")
        ok, msg = func(on_progress)
        if ok:
            return True
        if attempt < max_retries:
            if on_progress:
                on_progress(0, f"{label}: {msg}, retrying in {retry_delay}s...")
            time.sleep(retry_delay)
    if on_progress:
        on_progress(0, f"{label}: failed after {max_retries} attempts")
    return False


def clone_repo(on_progress=None) -> bool:
    if longcat_repo_cloned():
        return True

    target = config.LONGCAT_REPO_DIR
    target.mkdir(parents=True, exist_ok=True)

    def _do_clone(prog):
        if prog:
            prog(0, "Downloading LongCat repo (2GB)...")
        cmd = ["git", "clone", "--progress",
               "https://github.com/meituan-longcat/LongCat-Video.git", str(target)]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1)
            for line in proc.stdout:
                if prog and line:
                    prog(50, line.strip()[:80])
            proc.wait(timeout=1800)
            ok = proc.returncode == 0 and longcat_repo_cloned()
            if prog:
                prog(100 if ok else 50, "Repo ready!" if ok else "Clone failed")
            return ok, "Clone failed" if not ok else "OK"
        except subprocess.TimeoutExpired:
            if prog:
                prog(0, "Clone timed out after 30 min")
            try:
                proc.kill()
            except Exception:
                pass
            return False, "Timed out"
        except FileNotFoundError:
            if prog:
                prog(0, "Git not found - install git-scm.com")
            return False, "Git not installed"
        except Exception as e:
            if prog:
                prog(0, f"Clone error: {e}")
            return False, str(e)

    return _run_with_retry(_do_clone, max_retries=2, retry_delay=10,
                          on_progress=on_progress, label="Repo clone")


def download_weights(on_progress=None) -> bool:
    target = config.LONGCAT_WEIGHTS_DIR
    target.mkdir(parents=True, exist_ok=True)

    def _try_hf_api(prog):
        if prog:
            prog(0, "Downloading LongCat weights (27GB)...")
        try:
            from huggingface_hub import snapshot_download, logging as hf_logging
            import logging

            hf_logging.set_verbosity_error()

            snapshot_download(
                repo_id="meituan-longcat/LongCat-Video-Avatar-1.5",
                local_dir=str(target),
                resume_download=True,
                ignore_patterns=["*.git*"],
                local_dir_use_symlinks=False,
            )
            ok = any(target.iterdir())
            if prog:
                prog(100 if ok else 50, "Weights ready!" if ok else "Download incomplete")
            return ok, "OK" if ok else "Incomplete"
        except ImportError:
            if prog:
                prog(0, "huggingface_hub not found, trying CLI...")
            return _cli_download(target, prog)
        except Exception as e:
            if prog:
                prog(0, f"Download error: {e}")
            return False, str(e)

    def _cli_download(target_path, prog):
        if prog:
            prog(0, "CLI fallback: downloading weights...")
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "huggingface_hub", "download",
                 "meituan-longcat/LongCat-Video-Avatar-1.5",
                 "--local-dir", str(target_path), "--resume-download"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in proc.stdout:
                if prog and line:
                    prog(50, line.strip()[:80])
            proc.wait(timeout=7200)
            ok = proc.returncode == 0 and any(target_path.iterdir())
            if prog:
                prog(100 if ok else 50, "Weights ready!" if ok else "Download failed")
            return ok, "OK" if ok else "CLI download failed"
        except subprocess.TimeoutExpired:
            if prog:
                prog(0, "Download timed out after 2 hours")
            try:
                proc.kill()
            except Exception:
                pass
            return False, "Timed out"
        except Exception as e:
            if prog:
                prog(0, f"CLI download error: {e}")
            return False, str(e)

    return _run_with_retry(_try_hf_api, max_retries=3, retry_delay=10,
                          on_progress=on_progress, label="Weight download")


def setup_longcat(on_progress=None) -> bool:
    if on_progress:
        on_progress(0, "Checking GPU...")
    if not longcat_available():
        if on_progress:
            on_progress(100, "Need NVIDIA GPU with 16GB+ VRAM")
        return False
    if longcat_repo_cloned() and longcat_weights_exist():
        if on_progress:
            on_progress(100, "LongCat already ready!")
        return True
    if not clone_repo(on_progress=on_progress):
        return False
    if not download_weights(on_progress=on_progress):
        return False
    ready = longcat_repo_cloned() and longcat_weights_exist()
    if on_progress:
        on_progress(100, "LongCat setup complete!" if ready else "Setup incomplete")
    return ready


def _post_process_video(input_path: str, output_path: str, preset: str) -> str | None:
    import cv2
    import numpy as np

    if preset != "ultra":
        shutil.copy2(input_path, output_path)
        return output_path

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return None
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    sharpen = np.array([[-0.3, -0.3, -0.3], [-0.3, 3.4, -0.3], [-0.3, -0.3, -0.3]])

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.fastNlMeansDenoisingColored(frame, None, 5, 5, 7, 21)
        frame = cv2.filter2D(frame, -1, sharpen)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) * 1.08, 0, 255).astype(np.uint8)
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        writer.write(frame)

    cap.release()
    writer.release()
    return str(output_path) if Path(output_path).exists() else None


def generate_video(
    image_path: str,
    audio_path: str,
    output_path: str,
    duration_minutes: int = 1,
    quality: str = "auto",
    on_progress=None,
) -> str | None:
    if not longcat_available():
        return None
    if not (longcat_weights_exist() and longcat_repo_cloned()):
        if on_progress:
            on_progress(20, "LongCat: auto-downloading missing files...")
        if not setup_longcat(on_progress=on_progress):
            if on_progress:
                on_progress(40, "LongCat: setup incomplete - use sidebar Download button")
            return None

    quality = _auto_quality_preset() if quality == "auto" else quality
    preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["high"])

    if on_progress:
        on_progress(40, f"LongCat: {preset['description']}...")

    try:
        import torch

        prompt = "Professional news anchor, natural talking, broadcast studio lighting, high quality"
        input_data = {
            "prompt": prompt,
            "cond_image": str(Path(image_path).resolve()),
            "cond_audio": str(Path(audio_path).resolve()),
        }

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8") as f:
            json.dump(input_data, f)
            input_json_path = f.name

        nproc = max(1, torch.cuda.device_count())
        total_seconds = duration_minutes * 60
        first_seg_dur = 5.8
        next_seg_dur = 5.0
        num_segments = 1 if total_seconds <= first_seg_dur else 1 + int((total_seconds - first_seg_dur + next_seg_dur - 1) // next_seg_dur)

        cmd = [
            "torchrun", f"--nproc_per_node={nproc}",
            str(config.LONGCAT_REPO_DIR / "run_demo_avatar_single_audio_to_video.py"),
            f"--context_parallel_size={nproc}",
            f"--checkpoint_dir={str(config.LONGCAT_WEIGHTS_DIR.resolve())}",
            "--stage_1=ai2v",
            f"--input_json={input_json_path}",
            f"--output_dir={str(Path(output_path).parent.resolve())}",
            f"--resolution={preset['resolution']}",
            f"--num_inference_steps={preset['num_inference_steps']}",
            f"--num_segments={num_segments}",
        ]

        if on_progress:
            on_progress(45, f"Generating {num_segments} segment(s) at {preset['resolution']}...")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=max(3600, duration_minutes * 600))

        try:
            os.unlink(input_json_path)
        except Exception:
            pass

        if result.returncode != 0:
            print(f"LongCat error: {result.stderr[:500]}")
            return None

        out_dir = Path(output_path).parent
        candidates = list(out_dir.glob("*.mp4")) + list(out_dir.glob("*.MP4"))
        latest = max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None
        if not latest:
            return None

        if on_progress:
            on_progress(90, "Post-processing...")
        processed = _post_process_video(str(latest), str(output_path), quality)
        return processed

    except subprocess.TimeoutExpired:
        if on_progress:
            on_progress(0, "LongCat: generation timed out")
        return None
    except Exception as e:
        print(f"LongCat error: {e}")
        return None
