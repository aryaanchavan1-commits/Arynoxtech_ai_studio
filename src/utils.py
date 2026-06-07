import os
import re
import shutil
import subprocess
from pathlib import Path
import config


def clean_text(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"#+ ", "", text)
    text = re.sub(r"\[.*?\]\(.*?\)", r"\1", text)
    text = text.replace("|", "").replace("---", "")
    return text.strip()


def resolve_device() -> str:
    if config.DEVICE == "cpu":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def download_file(url: str, dest: Path, desc: str = "file") -> bool:
    import requests
    from tqdm import tqdm
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            desc=desc, total=total, unit="B", unit_scale=True, unit_divisor=1024
        ) as bar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def download_with_fallback(urls: list[str], dest: Path, desc: str = "model", min_size: int = 0) -> bool:
    """Try multiple URLs in order until one succeeds."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in urls:
        print(f"  Trying: {url}")
        if download_file(url, dest, desc):
            size = dest.stat().st_size if dest.exists() else 0
            if size >= min_size:
                print(f"  Downloaded ({size / 1e6:.0f} MB)")
                return True
            print(f"  Too small ({size} bytes), trying next...")
            dest.unlink(missing_ok=True)
    print(f"  Could not download {desc} from any source.")
    return False


def get_voice_options() -> list[dict]:
    try:
        import asyncio
        import edge_tts
        voices = asyncio.run(edge_tts.list_voices())
        return [
            {"name": v["ShortName"], "gender": v.get("Gender", ""), "locale": v.get("Locale", "")}
            for v in voices
        ]
    except Exception:
        return []


def format_timestamp(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
