"""
Standalone script to download LongCat-Video-Avatar-1.5 model.

Downloads:
  1. Clones the LongCat-Video GitHub repo (~2GB) with inference scripts
  2. Downloads model weights from ModelScope (~27GB)

Usage:
    python download_longcat.py          # Full download
    python download_longcat.py --weights-only  # Skip repo clone
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

try:
    from modelscope.hub.snapshot_download import snapshot_download
except ImportError:
    print("Installing modelscope...")
    subprocess.run([sys.executable, "-m", "pip", "install", "modelscope"], check=True)
    from modelscope.hub.snapshot_download import snapshot_download

MODELSCOPE_API_KEY = os.getenv("MODELSCOPE_API_KEY", "")

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
REPO_DIR = MODELS_DIR / "LongCat-Video"
WEIGHTS_DIR = MODELS_DIR / "LongCat-Video-Avatar-1.5"


def step(msg: str):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print('='*60)


def run_cmd(cmd, desc: str = "", timeout: int = 3600):
    print(f"  {desc or ' '.join(cmd[:3])}...")
    start = time.time()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, encoding="utf-8", errors="replace",
    )
    for line in proc.stdout:
        line = line.strip()
        if line:
            print(f"    {line[:200]}")
    proc.wait()
    elapsed = time.time() - start
    if proc.returncode != 0:
        print(f"  FAILED after {elapsed:.0f}s (code {proc.returncode})")
        return False
    print(f"  Done in {elapsed:.0f}s")
    return True


def clone_repo() -> bool:
    if REPO_DIR.exists() and (REPO_DIR / "run_demo_avatar_single_audio_to_video.py").exists():
        size_mb = sum(f.stat().st_size for f in REPO_DIR.rglob("*") if f.is_file()) / 1e6
        print(f"  Repo already cloned ({size_mb:.0f} MB)")
        return True
    step("Cloning LongCat-Video repo (~2GB)...")
    REPO_DIR.mkdir(parents=True, exist_ok=True)
    return run_cmd(
        ["git", "clone", "https://github.com/meituan-longcat/LongCat-Video.git", str(REPO_DIR)],
        desc="git clone LongCat-Video",
        timeout=600,
    )


def download_weights() -> bool:
    if WEIGHTS_DIR.exists() and any(WEIGHTS_DIR.iterdir()):
        size_gb = sum(f.stat().st_size for f in WEIGHTS_DIR.rglob("*") if f.is_file()) / 1e9
        print(f"  Weights already downloaded ({size_gb:.1f} GB)")
        return True
    step("Downloading LongCat-Video-Avatar-1.5 weights (~27GB) from ModelScope...")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    print("  This will take 15-60 minutes depending on internet speed.")
    try:
        if MODELSCOPE_API_KEY:
            from modelscope.hub.api import HubApi
            api = HubApi()
            api.login(MODELSCOPE_API_KEY)
        snapshot_download(
            "meituan-longcat/LongCat-Video-Avatar-1.5",
            local_dir=str(WEIGHTS_DIR),
            resume_download=True,
        )
        return True
    except Exception as e:
        print(f"  ModelScope download failed: {e}")
        print("  Falling back to CLI download...")
        return run_cmd(
            [sys.executable, "-m", "modelscope", "download",
             "meituan-longcat/LongCat-Video-Avatar-1.5",
             "--local-dir", str(WEIGHTS_DIR)],
            desc="modelscope CLI download weights",
            timeout=7200,
        )


def verify() -> bool:
    step("Verifying download...")
    ok = True
    if (REPO_DIR / "run_demo_avatar_single_audio_to_video.py").exists():
        print("  ✅ Repo: cloned")
    else:
        print("  ❌ Repo: missing")
        ok = False
    if WEIGHTS_DIR.exists() and any(WEIGHTS_DIR.iterdir()):
        gb = sum(f.stat().st_size for f in WEIGHTS_DIR.rglob("*") if f.is_file()) / 1e9
        print(f"  ✅ Weights: {gb:.1f} GB")
    else:
        print("  ❌ Weights: missing")
        ok = False
    if ok:
        print("\n  ✅ LongCat ready! Run: streamlit run app.py")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Download LongCat-Video-Avatar model")
    parser.add_argument("--weights-only", action="store_true", help="Skip repo clone")
    args = parser.parse_args()

    print("=" * 60)
    print("  LongCat-Video-Avatar-1.5 Downloader")
    print("=" * 60)

    if not shutil.which("git"):
        print("ERROR: git not found. Install git from https://git-scm.com/")
        sys.exit(1)

    if not args.weights_only:
        if not clone_repo():
            sys.exit(1)
    else:
        if (REPO_DIR / "run_demo_avatar_single_audio_to_video.py").exists():
            print("  Repo already cloned, skipping.")
        else:
            print("  ERROR: Repo not cloned and --weights-only specified.")
            sys.exit(1)

    if not download_weights():
        sys.exit(1)

    if not verify():
        sys.exit(1)


if __name__ == "__main__":
    main()
