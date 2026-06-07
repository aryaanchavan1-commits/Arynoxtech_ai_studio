"""
Setup script for AI News Studio on a new PC.

Run this on your new PC after copying the project folder:
    python setup_new_pc.py

This will:
  1. Create a virtual environment
  2. Install all Python dependencies
  3. Download Wav2Lip model (from HuggingFace mirror)
  4. Clone LongCat-Video repo & download weights (if CUDA GPU detected)
"""
import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
VENV_DIR = BASE_DIR / "venv"
REQUIREMENTS = [
    "streamlit>=1.28",
    "opencv-python-headless",
    "torch>=2.0",
    "torchvision",
    "numpy<2.0",
    "librosa",
    "scipy",
    "edge-tts>=7.2",
    "gtts",
    "elevenlabs",
    "python-dotenv",
    "groq",
    "openai>=1.0",
    "requests",
    "pillow",
    "huggingface-hub",
    "sarvamai",
]


def run(cmd, cwd=None):
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
    return result


def step(msg):
    print(f"\n{'='*60}")
    print(f"STEP: {msg}")
    print('='*60)


def main():
    print("=" * 60)
    print("  AI NEWS STUDIO - New PC Setup")
    print("=" * 60)

    # Step 1: Create virtual environment
    step("Creating virtual environment...")
    if not (VENV_DIR / "Scripts" / "python.exe").exists():
        run([sys.executable, "-m", "venv", str(VENV_DIR)])
    else:
        print("  Already exists, skipping.")

    pip = str(VENV_DIR / "Scripts" / "pip.exe")
    python = str(VENV_DIR / "Scripts" / "python.exe")

    # Step 2: Upgrade pip
    step("Upgrading pip...")
    run([pip, "install", "--upgrade", "pip"])

    # Step 3: Install requirements
    step("Installing Python packages (this takes a few minutes)...")
    for pkg in REQUIREMENTS:
        run([pip, "install", pkg])

    # Step 4: Check for CUDA GPU
    step("Checking for CUDA GPU...")
    result = run([python, "-c", "import torch; print(torch.cuda.is_available())"])
    has_cuda = "True" in result.stdout

    if has_cuda:
        print("  CUDA GPU detected!")
    else:
        print("  No CUDA detected. PyTorch will run on CPU.")
        print("  If you have an NVIDIA GPU, install CUDA toolkit from:")
        print("  https://developer.nvidia.com/cuda-downloads")

    # Step 5: Download Wav2Lip model from HuggingFace mirror
    step("Downloading Wav2Lip model (436 MB)...")
    models_dir = BASE_DIR / "models"
    models_dir.mkdir(exist_ok=True)
    wav2lip_path = models_dir / "wav2lip_gan.pth"

    if wav2lip_path.exists() and wav2lip_path.stat().st_size > 100e6:
        print(f"  Wav2Lip model already exists ({wav2lip_path.stat().st_size / 1e6:.0f} MB)")
    else:
        urls = [
            "https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth",
            "https://github.com/anothermartz/Easy-Wav2Lip/releases/download/Prerequesits/Wav2Lip_GAN.pth",
            "https://huggingface.co/rippertnt/wav2lip/resolve/6e6f1d979d1106130d7da9a655795b18f320a735/checkpoints/wav2lip_gan.pth",
            "https://huggingface.co/gmk123/wav2lip/resolve/main/wav2lip_gan.pth",
        ]
        for url in urls:
            print(f"  Trying: {url}")
            result = run([python, "-c", f"""
import urllib.request, os
try:
    urllib.request.urlretrieve('{url}', r'{wav2lip_path}')
    size = os.path.getsize(r'{wav2lip_path}')
    print(f'Downloaded: {{size / 1e6:.0f}} MB')
    if size > 100e6:
        exit(0)
except Exception as e:
    print(f'Failed: {{e}}')
    exit(1)
"""])
            if result.returncode == 0 and wav2lip_path.exists() and wav2lip_path.stat().st_size > 100e6:
                break
        else:
            print("  WARNING: Could not download Wav2Lip model from any source.")
            print("  Upload manually to: models/wav2lip_gan.pth")

    # Step 6: Clone LongCat-Video repo & download weights (if CUDA)
    if has_cuda:
        step("Cloning LongCat-Video repo (~2GB)...")
        lc_repo = models_dir / "LongCat-Video"
        if (lc_repo / "run_demo_avatar_single_audio_to_video.py").exists():
            print("  LongCat-Video repo already cloned.")
        else:
            lc_repo.mkdir(parents=True, exist_ok=True)
            run(["git", "clone", "https://github.com/meituan-longcat/LongCat-Video.git",
                 str(lc_repo)])

        step("Downloading LongCat-Video-Avatar-1.5 weights (~27GB)...")
        lc_weights = models_dir / "LongCat-Video-Avatar-1.5"
        if lc_weights.exists() and any(lc_weights.iterdir()):
            print("  LongCat weights already downloaded.")
        else:
            print("  This will take 15-30 minutes depending on your internet speed.")
            lc_weights.mkdir(parents=True, exist_ok=True)
            run([python, "-m", "huggingface_hub", "download",
                 "meituan-longcat/LongCat-Video-Avatar-1.5",
                 "--local-dir", str(lc_weights),
                 "--resume-download"])
    else:
        print("\n  Skipping LongCat setup (requires CUDA GPU).")
        print("  When you get a GPU PC, run:")
        print("    git clone https://github.com/meituan-longcat/LongCat-Video.git models/LongCat-Video")
        print("    huggingface-cli download meituan-longcat/LongCat-Video-Avatar-1.5 --local-dir models/LongCat-Video-Avatar-1.5")

    # Step 8: Verify installation
    step("Verifying installation...")
    test_code = """
import sys
sys.path.insert(0, r'{base}')
import config
print('  config.py OK')
from src.longcat_video import longcat_available, longcat_weights_exist
print('  longcat_video.py OK')
from src.elevenlabs_tts import elevenlabs_available
print('  elevenlabs_tts.py OK')
from src.text_to_speech import resolve_voice
print('  text_to_speech.py OK')
from src.video_pipeline import run_full_pipeline
print('  video_pipeline.py OK')
from src.wav2lip.models import load_wav2lip
print('  wav2lip.models.py OK')
from src.longcat_chat import generate_script
print('  longcat_chat.py OK')
print('ALL MODULES VERIFIED')
""".format(base=BASE_DIR)
    run([python, "-c", test_code])

    # Done
    print("\n" + "=" * 60)
    print("  SETUP COMPLETE!")
    print("=" * 60)
    print()
    print("To run the app:")
    print(f"  {VENV_DIR}\\Scripts\\streamlit run app.py")
    print()
    print("Edit .env file to add your API keys:")
    print("  GROQ_API_KEY  - https://console.groq.com")
    print("  LONGCAT_CHAT_API_KEY - https://longcat.chat/platform/api_keys")
    print("  SARVAM_API_KEY - https://sarvam.ai")
    print()
    print("For LongCat AI avatar (GPU only):")
    print("  Already set up if CUDA was detected.")
    print()


if __name__ == "__main__":
    main()
