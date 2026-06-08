"""
Standalone script to download Wav2Lip GAN model.

Downloads the Wav2Lip GAN checkpoint (~150MB) with multiple fallback mirrors.

Usage:
    python download_wav2lip.py              # Full download with progress
    python download_wav2lip.py --verify     # Only verify existing download
"""
import argparse
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    import subprocess
    print("Installing requests...")
    subprocess.run([sys.executable, "-m", "pip", "install", "requests"], check=True)
    import requests

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "wav2lip_gan.pth"

WAV2LIP_MODEL_URLS = [
    ("HuggingFace (Nekochu)", "https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth"),
    ("GitHub (Easy-Wav2Lip)", "https://github.com/anothermartz/Easy-Wav2Lip/releases/download/Prerequesits/Wav2Lip_GAN.pth"),
    ("HuggingFace (rippertnt)", "https://huggingface.co/rippertnt/wav2lip/resolve/6e6f1d979d1106130d7da9a655795b18f320a735/checkpoints/wav2lip_gan.pth"),
    ("HuggingFace (gmk123)", "https://huggingface.co/gmk123/wav2lip/resolve/main/wav2lip_gan.pth"),
    ("HuggingFace (sparksammy)", "https://huggingface.co/sparksammy/wav2lip-gan/resolve/main/wav2lip_gan.pth"),
    ("GitHub (Rudrabha)", "https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0/wav2lip_gan.pth"),
]


def step(msg: str):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print('='*60)


def download_file(url: str, dest: Path, source_name: str = "") -> bool:
    try:
        print(f"\n  Downloading from {source_name or url}...")
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        start = time.time()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded * 100 // total
                    speed = downloaded / (time.time() - start) / 1024 / 1024 if (time.time() - start) > 0 else 0
                    print(f"\r    Progress: {pct}% ({downloaded/1e6:.0f}/{total/1e6:.0f} MB) @ {speed:.1f} MB/s", end="")
        elapsed = time.time() - start
        size = dest.stat().st_size if dest.exists() else 0
        print(f"\n    Downloaded {size/1e6:.0f} MB in {elapsed:.0f}s ({size/elapsed/1e6:.1f} MB/s)")
        return size >= 100e6
    except Exception as e:
        print(f"    Failed: {e}")
        return False


def verify_download() -> bool:
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 100e6:
        size_mb = MODEL_PATH.stat().st_size / 1e6
        print(f"  Wav2Lip model exists ({size_mb:.0f} MB)")
        return True
    print("  Wav2Lip model not found or too small")
    return False


def main():
    parser = argparse.ArgumentParser(description="Download Wav2Lip GAN model")
    parser.add_argument("--verify", action="store_true", help="Only verify existing download")
    args = parser.parse_args()

    print("=" * 60)
    print("  Wav2Lip GAN Model Downloader")
    print("  For Arynox AI Studio")
    print("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.verify:
        step("Verifying download")
        if verify_download():
            print("\n  Wav2Lip model is ready!")
            return
        print("\n  Model not found. Run without --verify to download.")
        sys.exit(1)

    if verify_download():
        print("\n  Already downloaded. Use --verify to check, or delete the file to re-download.")
        return

    step("Downloading Wav2Lip GAN model (~150 MB)")
    print("  Trying multiple mirrors in order...")

    for source_name, url in WAV2LIP_MODEL_URLS:
        if download_file(url, MODEL_PATH, source_name):
            print(f"  Successfully downloaded from {source_name}")
            verify_download()
            print("\n  Wav2Lip model ready!")
            print(f"  Location: {MODEL_PATH}")
            return
        MODEL_PATH.unlink(missing_ok=True)

    print("\n  ERROR: Could not download from any source.")
    print("  Please manually download:")
    print("    https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0/wav2lip_gan.pth")
    print(f"  And save to: {MODEL_PATH}")
    sys.exit(1)


if __name__ == "__main__":
    main()
