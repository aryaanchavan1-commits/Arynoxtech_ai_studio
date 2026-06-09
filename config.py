import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

# Paths
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
VOICES_DIR = DATA_DIR / "voices"
ASSETS_DIR = BASE_DIR / "assets"

THEMES_DIR = BASE_DIR / "themes"

for d in [MODELS_DIR, DATA_DIR, UPLOADS_DIR, OUTPUT_DIR, VOICES_DIR, ASSETS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
LONGCAT_CHAT_API_KEY = os.getenv("LONGCAT_CHAT_API_KEY", "")
MODELSCOPE_API_KEY = os.getenv("MODELSCOPE_API_KEY", "")

# Device
DEVICE = os.getenv("DEVICE", "auto")

# Wav2Lip model — primary URL + fallback mirrors
WAV2LIP_MODEL_URLS = [
    "https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth",
    "https://github.com/anothermartz/Easy-Wav2Lip/releases/download/Prerequesits/Wav2Lip_GAN.pth",
    "https://huggingface.co/rippertnt/wav2lip/resolve/6e6f1d979d1106130d7da9a655795b18f320a735/checkpoints/wav2lip_gan.pth",
    "https://huggingface.co/gmk123/wav2lip/resolve/main/wav2lip_gan.pth",
]
WAV2LIP_MODEL_PATH = MODELS_DIR / "wav2lip_gan.pth"

# Face detection — uses OpenCV Haar cascade (built-in, no download needed)

# DiT Video Generation (Diffusion Transformer)
DIT_MODELS_DIR = MODELS_DIR / "dit"
DIT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
DIT_COGVIDEOX_PATH = DIT_MODELS_DIR / "CogVideoX-5B-I2V"
DIT_LTX_PATH = DIT_MODELS_DIR / "LTX-Video"
DIT_BACKEND = os.getenv("DIT_BACKEND", "auto")
DIT_ENABLE = os.getenv("DIT_ENABLE", "auto")
DIT_MAX_RESOLUTION = os.getenv("DIT_MAX_RESOLUTION", "720p")
DIT_SCENE_DURATION = int(os.getenv("DIT_SCENE_DURATION", "4"))
DIT_USE_CPU_OFFLOAD = os.getenv("DIT_USE_CPU_OFFLOAD", "true") == "true"
DIT_GUIDANCE_SCALE = float(os.getenv("DIT_GUIDANCE_SCALE", "7.0"))
DIT_NUM_INFERENCE_STEPS = int(os.getenv("DIT_NUM_INFERENCE_STEPS", "50"))
DIT_USE_GENERATED_B_ROLL = os.getenv("DIT_USE_GENERATED_B_ROLL", "true") == "true"

# Video Enhancement
VIDEO_UPSCALE = os.getenv("VIDEO_UPSCALE", "true") == "true"
VIDEO_UPSCALE_TARGET = os.getenv("VIDEO_UPSCALE_TARGET", "3840x2160")
VIDEO_ENABLE_COLOR_GRADE = os.getenv("VIDEO_ENABLE_COLOR_GRADE", "true") == "true"
VIDEO_ENABLE_TRANSITIONS = os.getenv("VIDEO_ENABLE_TRANSITIONS", "crossfade")
VIDEO_TRANSITION_FRAMES = int(os.getenv("VIDEO_TRANSITION_FRAMES", "30"))
VIDEO_CRF = int(os.getenv("VIDEO_CRF", "14"))
VIDEO_PRESET = os.getenv("VIDEO_PRESET", "veryslow")
VIDEO_AUDIO_BITRATE = os.getenv("VIDEO_AUDIO_BITRATE", "320k")
VIDEO_ENABLE_DENOISE = os.getenv("VIDEO_ENABLE_DENOISE", "true") == "true"
VIDEO_ENABLE_SHARPEN = os.getenv("VIDEO_ENABLE_SHARPEN", "true") == "true"
VIDEO_ENABLE_GAMMA_CORRECTION = os.getenv("VIDEO_ENABLE_GAMMA_CORRECTION", "true") == "true"
VIDEO_GAMMA = float(os.getenv("VIDEO_GAMMA", "1.0"))
CINEMATIC_STYLE = os.getenv("CINEMATIC_STYLE", "evening")

# Quality Presets
QUALITY_PRESET = os.getenv("QUALITY_PRESET", "cinema")
ENABLE_FACE_ENHANCE = os.getenv("ENABLE_FACE_ENHANCE", "true") == "true"
ENABLE_FRAME_INTERPOLATION = os.getenv("ENABLE_FRAME_INTERPOLATION", "true") == "true"
ENABLE_SUPER_RES = os.getenv("ENABLE_SUPER_RES", "false") == "true"
FRAME_INTERPOLATION_TARGET = int(os.getenv("FRAME_INTERPOLATION_TARGET", "60"))

# LongCat-Video (local GPU inference)
LONGCAT_REPO_DIR = MODELS_DIR / "LongCat-Video"
LONGCAT_WEIGHTS_DIR = MODELS_DIR / "LongCat-Video-Avatar-1.5"

# Kling AI Video API
KLING_ACCESS_KEY = os.getenv("KLING_ACCESS_KEY", "")
KLING_SECRET_KEY = os.getenv("KLING_SECRET_KEY", "")
KLING_MODEL = os.getenv("KLING_MODEL", "kling-v3")
KLING_ENABLE = os.getenv("KLING_ENABLE", "auto")
KLING_EXTEND_IMAGE = os.getenv("KLING_EXTEND_IMAGE", "")

# Video settings
OUTPUT_FPS = int(os.getenv("OUTPUT_FPS", "60"))
OUTPUT_WIDTH, OUTPUT_HEIGHT = map(int, os.getenv("OUTPUT_RESOLUTION", "3840x2160").split("x"))

# Language
LANGUAGE = os.getenv("LANGUAGE", "english")

# TTS
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-AriaNeural")
TTS_SPEED = float(os.getenv("TTS_SPEED", "1.0"))

LANGUAGE_VOICES = {
    "english": {
        "female": {
            "indian": "en-IN-NeerjaNeural",
            "us": "en-US-AriaNeural",
            "uk": "en-GB-SoniaNeural",
        },
        "male": {
            "indian": "en-IN-PrabhatNeural",
            "us": "en-US-GuyNeural",
            "uk": "en-GB-RyanNeural",
        },
    },
    "hindi": {
        "female": {"default": "hi-IN-SwaraNeural"},
        "male":   {"default": "hi-IN-MadhurNeural"},
    },
    "marathi": {
        "female": {"default": "mr-IN-AarohiNeural"},
        "male":   {"default": "mr-IN-ManoharNeural"},
    },
    "tamil": {
        "female": {"default": "ta-IN-PallaviNeural"},
        "male":   {"default": "ta-IN-ValluvarNeural"},
    },
    "telugu": {
        "female": {"default": "te-IN-ShrutiNeural"},
        "male":   {"default": "te-IN-MohanNeural"},
    },
    "kannada": {
        "female": {"default": "kn-IN-SapnaNeural"},
        "male":   {"default": "kn-IN-GaganNeural"},
    },
    "malayalam": {
        "female": {"default": "ml-IN-SobhanaNeural"},
        "male":   {"default": "ml-IN-MidhunNeural"},
    },
    "bengali": {
        "female": {"default": "bn-IN-TanishaNeural"},
        "male":   {"default": "bn-IN-BashkarNeural"},
    },
    "gujarati": {
        "female": {"default": "gu-IN-NishaNeural"},
        "male":   {"default": "gu-IN-DharmeshNeural"},
    },
    "punjabi": {
        "female": {"default": "pa-IN-SanaNeural"},
        "male":   {"default": "pa-IN-UnknownNeural"},
    },
    "odia": {
        "female": {"default": "or-IN-SubhasiniNeural"},
        "male":   {"default": "or-IN-SukantNeural"},
    },
}

# Groq model
GROQ_MODEL = "llama-3.3-70b-versatile"

WAV2LIP_IMG_SIZE = 96
WAV2LIP_MEL_STEP = 16
WAV2LIP_FPS = 25
