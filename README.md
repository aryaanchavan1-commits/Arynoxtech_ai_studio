# Arynoxtech AI Studio&#x20;

Turn any portrait photo into a **professional Marathi/English news anchor**. Full broadcast pipeline: AI script → Sarvam AI Marathi voice → LongCat talking avatar + Kling AI cinematic B-roll → interleaved broadcast edit with crossfades → studio production overlays.

**New in v3.0:**
- **NLP Amplifier** — Automatically enhances brief user prompts into cinematic production-quality descriptions. Better first-generation results = fewer retries = lower API spend. Turn "a man speaking about tech" into "cinematic neon-lit tech studio with volumetric lighting..."
- **NLP Amplifier toggle** in Scene Description — enable/disable AI prompt enhancement per session
- **`download_wav2lip.py`** — Standalone script to download the Wav2Lip GAN model with progress bars and 6 fallback mirrors
- **Quality Presets** — Choose from Standard / Premium / Cinema modes (in sidebar)
- **60fps Frame Interpolation** — Smooth motion via ffmpeg minterpolate (bidirectional motion compensation)
- **Face Enhancement** — OpenCV-based face detail enhancement + edge-preserving filter on detected faces
- **Multi-pass Post-Processing** — Denoise → Sharpen → Face enhance → Upscale → Frame interpolation → Final encode
- **4K Output** — Native 3840x2160 upscale with Lanczos + CRF 14 veryslow encoding

**New in v2.0:** Professional interleaved broadcast editing — anchor speaks, then cuts to cinematic B-roll, then back to anchor — like a real news channel. Kling AI generates 5-10s clips from scene-matched prompts. Sarvam AI Bulbul v3 with explicit voice selection (sumit/ishita/amit/priya).

**Use cases:** Marathi news broadcasts, daily news shows, corporate announcements, educational content, social media videos.

Created by **Aryan Chavan** for **Arynoxtech**

---

## What This Does (Non-Technical Explanation)

Imagine you have a photo of yourself. This software makes that photo come alive — it talks, moves its head, blinks, and presents your script like a real TV news anchor. It can:

- Write a Marathi/English news script automatically from any topic (or use your own text)
- Generate a natural Marathi voiceover using Sarvam AI Bulbul v3 (voices: sumit, ishita, amit, priya, plus auto)
- **NLP Amplifier** — Auto-enhances your scene descriptions into cinematic production-quality prompts using Groq AI. Better prompts = better outputs = less API spend
- Make the photo speak with perfectly synced lip movements via LongCat full-body avatar (16GB+ GPU) or Wav2Lip (any GPU)
- Generate cinematic B-roll clips using **Kling AI** API — scene-matched visuals for each news segment
- **Interleave anchor segments with B-roll clips** in a professional broadcast edit with crossfade transitions
- Add studio overlays, branded titles, ticker, lower thirds, intro/outro animations
- Accept custom themes/backgrounds you upload
- Produce a professional 1080p MP4 video ready for any platform

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8GB | 16GB+ |
| GPU | Any NVIDIA (4GB+) | RTX 4080/4090 (16GB+) |
| Storage | 10GB free | 50GB free (for AI models) |
| OS | Windows 10/11 | Windows 11 |
| Internet | Required for setup | Broadband |

### What Works on Your Current Laptop (RTX 3050 4GB)
- Full Wav2Lip pipeline (lip sync + motion enhancement + studio production)
- AI news script generation
- Edge-TTS and Sarvam AI voiceovers
- All TTS engines (English + Marathi)
- **Kling AI B-roll generation** (API-based, no GPU needed)
- **Wav2Lip + Kling mode** (interleaved broadcast edit with Kling B-roll)

### What Requires a Better GPU (RTX 4080/4090 with 16GB+ VRAM)
- DiT AI video generation (CogVideoX) — needs 12GB+ VRAM
- LongCat avatar animation — needs 16GB+ VRAM
- **LongCat + Kling Broadcast Production mode** — full interleaved anchor/B-roll edit

### API Requirements

* **Kling AI** — requires API credits at app.klingai.com (\~\$0.35/generation for v2.5 Turbo)
- **Sarvam AI** — free tier available for Indian TTS
- **Groq** — free tier for AI script generation

The system auto-detects your GPU and enables only what your hardware can handle.

## Algorithms & AI Models Used

### Speech & Language
| Model | Purpose | Source |
|-------|---------|--------|
| **Llama 3.3 70B** (via Groq API) | News script generation, **NLP prompt amplification** | groq.com |
| **Edge-TTS** | Neural text-to-speech (75+ voices, 50+ languages) | Microsoft (local) |
| **Sarvam AI** | Indian-accented TTS (Hindi, Marathi, Tamil, etc.) | sarvam.ai |
| **ElevenLabs** | Premium voice cloning & TTS | elevenlabs.io |

### Video Generation
| Model | Purpose | When Used |
|-------|---------|-----------|
| **Wav2Lip GAN** | Lip-sync audio to video frames | Always (fallback) |
| **CogVideoX-5B** (diffusers) | Text-to-video scene generation | DiT mode (12GB+ GPU) |
| **LongCat-Video-Avatar-1.5** | Full body talking avatar | LongCat mode (16GB+ GPU) |
| **Kling AI** (v1/v2.5 Turbo/v3) | Cinematic B-roll generation via API | Kling mode (any GPU) |
| **Kling + LongCat** | Interleaved anchor (LongCat) ↔ B-roll (Kling) broadcast edit | Production mode (16GB+ GPU) |
| **Kling + Wav2Lip** | Talking head (Wav2Lip) + Kling B-roll interleaved | Fallback production (any GPU) |

### NLP & Prompt Engineering
| Algorithm | Purpose | Implementation |
|-----------|---------|----------------|
| **NLP Amplifier** | Expands brief user prompts into cinematic production-quality descriptions using Groq LLM | Custom (src/nlp_amplifier.py) |
| **Cinematic Terminology Injection** | Adds professional cinematography terms (lighting, camera, mood) for better AI generation | Custom (src/nlp_amplifier.py) |

### Computer Vision & Processing
| **Haar Cascade** | Face detection | OpenCV (built-in) |
| **OpenCV FaceMesh** | Facial landmark detection | MediaPipe |
| **Lanczos Interpolation** | Video upscaling to 1080p | OpenCV `INTER_LANCZOS4` |
| **Fast NL-Means Denoising** | Frame noise reduction | OpenCV `fastNlMeansDenoising` |
| **LAB Histogram Matching** | Color grading between layers | Custom (src/video_enhancer.py) |
| **HSV Saturation Boost** | Color enhancement | Custom (src/video_enhancer.py) |
| **Unsharp Mask (kernel)** | Frame sharpening | Custom 3x3 convolution |
| **Gamma Correction** | Brightness/contrast adjustment | Power-law transform |
| **ffmpeg xfade** | Scene transitions (crossfade/slide/fade) | ffmpeg |
| **Motion Enhancement** | Head movement, blinking, breathing simulation | Custom (src/motion_enhancer.py) |

### Audio Processing
| Algorithm | Purpose |
|-----------|---------|
| **Librosa STFT/Mel** | Audio feature extraction for lip-sync |
| **ffmpeg amix** | Voice + background music mixing |
| **AAC encoding (256kbps)** | Final audio compression |

## Project Structure

```
Arynox-AI-Studio/
  app.py                  Main Streamlit UI
  config.py               All settings and paths
  requirements.txt        Python dependencies
  setup.bat               One-click Windows setup
  setup_new_pc.py         Python setup script
  download_longcat.py     Downloads LongCat AI avatar model (27GB)
  download_wav2lip.py     Downloads Wav2Lip GAN model (~150MB) with 6 fallback mirrors
  .env                    API keys (you edit this)
  .gitignore              Files to exclude from git

  src/                    All source code
    video_pipeline.py     Main pipeline (routes to correct engine)
    lip_sync.py           Wav2Lip engine wrapper
    motion_enhancer.py    Head movement, blinking, breathing
    news_script.py        AI news script generation (Groq)
    nlp_amplifier.py      NLP Amplifier — enhances prompts into cinematic quality
    text_to_speech.py     Edge-TTS voice generation
    sarvam_tts.py         Sarvam AI TTS integration (voice override)
    elevenlabs_tts.py     ElevenLabs TTS integration
    longcat_video.py      LongCat avatar setup & generation
    longcat_chat.py       LongCat Chat API for scripts
    kling_video.py        Kling AI client (JWT auth, text2video, image2video, polling)
    dit_video.py          DiT (CogVideoX) video engine
    dit_pipeline.py       DiT end-to-end pipeline
    scene_director.py     Converts scripts to anchor/broll scenes with topic-matched prompts
    scene_composer.py     Studio overlay (ticker, lower third, intro)
    scene_transition.py   ffmpeg crossfade/slide/wipe transitions
    video_enhancer.py     Upscaling, color grading, compositing
    character_compositor.py  Place character into studio scene
    theme_manager.py      Studio theme management
    utils.py              Shared utilities
    video_quality.py       Quality presets: denoise, sharpen, face enhance, 60fps interpolation, 4K upscale
    nlp_amplifier.py       NLP prompt enhancer (Groq-based)

    wav2lip/              Wav2Lip model code & face detection
  models/                 Downloaded AI models (gitignored)
  output/                 Generated videos (gitignored)
  data/                   Uploads, temporary files
  themes/                 Studio background themes
  assets/                 Static assets
```

## Setup Guide (Step by Step)

### For Non-Technical Users

#### Step 1: Install Python
1. Go to https://www.python.org/downloads/
2. Click "Download Python 3.10 or 3.11" (not 3.13 — some libraries need 3.10/3.11)
3. Run the installer
4. **IMPORTANT:** Check "Add Python to PATH" at the bottom
5. Click "Install Now"

#### Step 2: Install Git
1. Go to https://git-scm.com/download/win
2. Download and run the installer (default settings are fine)

#### Step 3: Install FFmpeg
Open Command Prompt (Win+R, type `cmd`, press Enter) and run:
```
winget install ffmpeg
```

#### Step 4: Get the Project
Open Command Prompt and run:
```
cd Desktop
git clone https://github.com/YOUR_USERNAME/Arynox-AI-News-Studio.git
cd Arynox-AI-News-Studio
```

#### Step 5: Run Setup
Double-click `setup.bat` in the project folder.
This will:
- Create a virtual environment
- Install all Python packages
* Download the Wav2Lip model (\~150MB)
- Check for FFmpeg

#### Step 6: Configure `.env` File

Open the `.env` file in the project folder (right-click, open with Notepad) and set these values:

```ini
# === REQUIRED: Groq API (free AI script generation) ===
# Get key at https://console.groq.com
GROQ_API_KEY=gsk_your_groq_api_key_here

# === OPTIONAL: Sarvam AI (best for Marathi/Indian TTS) ===
# Get key at https://sarvam.ai
# Voices: sumit (male, Marathi), ishita (female, Marathi),
#         amit (male, Hindi), priya (female, Hindi)
SARVAM_API_KEY=your_sarvam_api_key_here
SARVAM_VOICE=sumit

# === OPTIONAL: Kling AI (cinematic B-roll video scenes) ===
# Get keys at https://app.klingai.com
KLING_ACCESS_KEY=your_kling_access_key_here
KLING_SECRET_KEY=your_kling_secret_key_here
KLING_MODEL=kling-v2-5-turbo

# === OPTIONAL: ElevenLabs (premium English TTS) ===
# ELEVENLABS_API_KEY=your_key_here
# ELEVENLABS_VOICE_ID=your_voice_id_here

# === VIDEO SETTINGS ===
DEVICE=auto          # auto, cuda, or cpu
LANGUAGE=marathi     # english, marathi, hindi, tamil, telugu, etc.
TTS_VOICE=mr-IN-ManoharNeural   # Edge-TTS voice (fallback)
TTS_SPEED=1.0
OUTPUT_FPS=25
OUTPUT_RESOLUTION=1920x1080
```
Save and close.

#### Step 7: Run the App
Double-click `run.bat` (or run `streamlit run app.py` in the command prompt).
The app opens in your browser at `http://localhost:8501`.

### For Technical Users (Manual Setup)

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/Arynox-AI-News-Studio.git
cd Arynox-AI-News-Studio

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download Wav2Lip model (uses download_wav2lip.py with 6 fallback mirrors)
python download_wav2lip.py

# Or manually:
# python -c "
# from huggingface_hub import snapshot_download
# import requests
# url = 'https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth'
# resp = requests.get(url, stream=True, timeout=120)
# resp.raise_for_status()
# with open('models/wav2lip_gan.pth', 'wb') as f:
#     for chunk in resp.iter_content(8192):
#         f.write(chunk)
# print('Wav2Lip model downloaded')
# "

# Edit .env with your API keys
notepad .env

# Run
streamlit run app.py
```

## How to Use (User Guide)

### Generating Your First Video

1. **Upload a Character Image**
   - Click "Browse files" in the Upload section
   - Choose a front-facing portrait photo (JPG or PNG)
   - The photo should show the person looking at the camera
2. **Enter Your Content**
   - **AI Mode:** Type a topic like "Birthday greeting for my friend" or "New product launch announcement"
   - **Manual Mode:** Paste your own script text directly
   - **Visual Style (optional):** Describe the scene/background, e.g. "A tropical beach at sunset" or "Futuristic cyberpunk city"
3. **Select Language & Voice**
    - Choose from 11 Indian languages (English, Hindi, Marathi, Tamil, Telugu, Kannada, Malayalam, Bengali, Gujarati, Punjabi, Odia)
    - For Sarvam AI: pick explicit voice (sumit/ishita/amit/priya) for consistent anchor personality
    - For Edge-TTS: select male or female voice
    - For English, pick Indian accent for natural Indian English
4. **Configure Video Settings**
   - Duration: 1-5 minutes
   - **Kling AI:** Enable for cinematic B-roll clips between anchor segments
   - **LongCat:** Enable for full-body talking avatar (needs 16GB+ VRAM)
   - **Both LongCat + Kling:** Broadcast PRO mode — interleaved anchor ↔ B-roll edit
   - Enable Studio Overlay for professional TV look
   - Upload your own background theme image
   - Enable ticker for scrolling text at bottom (news-style)
5. **Click "Generate Video"**
   - The progress bar shows each step
   * Processing time varies: \~2 minutes per minute of video
   - The final video appears in the preview player
6. **Download**
   - Click the download button to save the MP4 file

### Engine Modes Explained

Your system automatically picks the best available engine based on your GPU and selected options:

| Mode | What It Does | GPU Needed |
|------|-------------|------------|
| **Standard (Wav2Lip)** | Lip-syncs your photo to audio + motion enhancement | Any GPU (4GB+) |
| **Wav2Lip + Kling** | Talking head + Kling cinematic B-roll, interleaved edit | Any GPU (Kling is API) |
| **DiT Video Engine** | Generates cinematic background scenes with AI | 12GB+ VRAM |
| **LongCat Avatar** | Creates full body talking avatar from photo | 16GB+ VRAM |
| **LongCat + Kling** (Broadcast PRO) | Full anchor avatar + Kling B-roll, interleaved with crossfades | 16GB+ VRAM (Kling is API) |
| **LongCat + DiT** | Avatar + AI backgrounds combined | 16GB+ VRAM + 12GB+ VRAM |

### Broadcast Production Mode (LongCat + Kling)

This is the flagship feature — produces professional interleaved news broadcasts:

1. **Script** → Groq AI generates a structured news script with multiple segments
2. **Voice** → Sarvam AI Bulbul v3 generates Marathi/English voiceover (select sumit/ishita/amit/priya)
3. **Avatar** → LongCat creates full-body talking anchor lip-synced to the audio
4. **Scene Planning** → `scene_director.py` classifies topic (tech/business/sports/etc.) and generates scene-matched prompts
5. **B-roll** → Kling AI generates 5-10s cinematic clips for each B-roll scene
6. **Interleaved Edit** → Anchor segments are trimmed from LongCat video, B-roll clips from Kling are sequenced alternately: Anchor → crossfade → B-roll → crossfade → Anchor → ...
7. **Audio Mixing** → Final audio track is overlaid on the interleaved video
8. **Studio Production** → Intro/outro, ticker, lower thirds, branded overlays are composited on top

The result looks like a real news broadcast — the anchor delivers a segment, then the video cuts to relevant visuals, then back to the anchor.

**Cost per 2-minute news video:** \~$2.10 (Kling API, ~6 clips at $0.35 each) + free (LongCat, Sarvam, Groq)

### Studio Production Features

When "Enable Studio Overlay" is on:
- **Intro/Outro** — Animated opening and closing segments with your branding
- **Ticker** — Scrolling text at the bottom (great for news, announcements)
- **Lower Third** — Presenter name and branding overlay
- **Background Themes** — Choose from built-in studio themes or upload your own
- **Music** — Add background music track

### Custom Visual Prompts

You can describe exactly what the video background should look like:

| Prompt | Result |
|--------|--------|
| *(empty)* | AI auto-generates based on your topic |
| `"Tropical beach at sunset` | Warm beach scene background |
| `"Futuristic cyberpunk city with neon lights"` | Neon cityscape |
| `"Elegant corporate boardroom"` | Professional office setting |
| `"Cozy living room with fireplace"` | Warm indoor setting |

### NLP Amplifier (Prompt Enhancement)

The **NLP Amplifier** uses Groq AI (Llama 3.3 70B) to automatically transform your brief scene descriptions into rich, cinematic production-quality prompts:

| You Type | NLP Amplifier Produces |
|----------|----------------------|
| `"tech studio"` | `"Futuristic technology visualization, advanced AI neural network animation, glowing digital circuits, blue and purple neon cinematic lighting, holographic interfaces, smooth camera dolly movement, 8K tech documentary quality"` |
| `"beach sunset"` | `"Tropical beach scene at golden hour, warm amber sunlight reflecting on gentle waves, cinematic wide shot, soft bokeh in foreground, dramatic cloud formations, professional travel documentary quality"` |
| `"corporate boardroom"` | `"Elegant corporate boardroom with floor-to-ceiling windows, modern minimalist design, soft natural lighting through blinds, mahogany conference table, professional business atmosphere, cinematic shallow depth of field"` |

**How it reduces API spending:** Better prompts mean AI generators (Kling, DiT) produce usable results on the **first attempt**, eliminating costly retries. Each Kling generation costs ~$0.35 — one retry saved per scene can cut costs by 30-50%.

**To enable/disable:** Toggle "NLP Amplifier" checkbox in the scene description section of the sidebar.

### Tips for Best Results

- Use a well-lit, front-facing photo with neutral expression
- For Marathi content, use Sarvam AI voice with "sumit" (male) or "ishita" (female)
- Keep videos under 5 minutes for fastest processing
- Upload your own background theme for custom branding
- Describe visual scenes in detail for better AI-generated backgrounds
* On a laptop GPU (4GB), expect \~2min processing per 1min of video
- On a high-end GPU (24GB), LongCat + Kling Broadcast mode delivers the most professional output
- **Without LongCat (any GPU):** Enable Wav2Lip + Kling mode — you still get cinematic B-roll interleaved with your talking head
- **Kling credits:** Each 2-minute news video uses about 4-8 Kling generations ($1.40-$2.80 at v2.5 Turbo pricing)
- **Sarvam voices:** Use explicit voice names (sumit/ishita) for consistent anchor personality across episodes

## Configuration

All settings are in `config.py` or overrideable via `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | API key for news script generation |
| `SARVAM_API_KEY` | — | API key for Indian TTS |
| `SARVAM_VOICE` | auto | Default Sarvam voice: `sumit`, `ishita`, `amit`, `priya`, `meera`, `vijay`, `kavya`, `arjun`, `auto` |
| `NEWSAPI_KEY` | — | API key for live news fetch (optional) |
| `ELEVENLABS_API_KEY` | — | API key for premium TTS (optional) |
| `KLING_ACCESS_KEY` | — | Kling AI API access key (app.klingai.com) |
| `KLING_SECRET_KEY` | — | Kling AI API secret key |
| `KLING_MODEL` | kling-v3 | Model: `kling-v1`, `kling-v2-5-turbo`, `kling-v3` |
| `DEVICE` | auto | `auto`, `cuda`, or `cpu` |
| `LANGUAGE` | english | `english`, `hindi`, `marathi`, `tamil`, `telugu`, `kannada`, `malayalam`, `bengali`, `gujarati`, `punjabi`, `odia` |
| `OUTPUT_RESOLUTION` | 1920x1080 | Final video resolution |
| `DIT_ENABLE` | auto | Enable DiT when GPU has 12GB+ VRAM |
| `DIT_NUM_INFERENCE_STEPS` | 50 | Quality vs speed (higher = better) |

## Advanced: Setting Up DiT (CogVideoX)

DiT auto-enables when you have a GPU with 12GB+ VRAM. On first use, it downloads \~12GB of model weights. This happens automatically when you click "Generate Video" with DiT enabled.

To force download ahead of time:
```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('THUDM/CogVideoX-5B-I2V',
    local_dir='models/dit/CogVideoX-5B-I2V',
    resume_download=True)
"
```

## Advanced: Setting Up LongCat Avatar

LongCat requires an NVIDIA GPU with 16GB+ VRAM. When you enable it in the sidebar, the system:

1. Checks if the LongCat repo is cloned (\~2GB)
2. Checks if the weight files are downloaded (\~27GB)
3. Auto-downloads anything missing

You can also click "Download LongCat AI (27GB)" in the sidebar to pre-download.

## Advanced: Setting Up Kling AI

Kling AI is an API-based service — no GPU needed. You need credits on your Kling account:

1. Go to https://app.klingai.com and sign up
2. Navigate to API management and create an access key/secret key pair
3. Add credits to your account (starts at \~\$10)
4. Add the keys to your `.env` file:
   ```
KLING_ACCESS_KEY=your_access_key_here
KLING_SECRET_KEY=your_secret_key_here
KLING_MODEL=kling-v3
   ```

**Pricing:** Kling v2.5 Turbo = \~$0.35/generation (5-10s clip), v3 = ~$0.84/generation. For a 2-minute news video with \~6 B-roll clips, expect \~\$2-5 per video total.

**Available models:** `kling-v1` ($0.20), `kling-v2-5-turbo` ($0.35, best value), `kling-v3` (\$0.84, latest quality).

## Troubleshooting

### "No module named X"
```bash
venv\Scripts\activate
pip install -r requirements.txt
```

### "FFmpeg not found"
Install: `winget install ffmpeg` or download from https://ffmpeg.org

### "Groq API key not set"
Open `.env` and add your key from https://console.groq.com

### "Out of memory" / "CUDA out of memory"
Your GPU doesn't have enough VRAM for the selected engine. The system will fall back to Wav2Lip. Use CPU mode if no GPU available:
```bash
echo DEVICE=cpu >> .env
```

### "LongCat generation failed"
- Ensure you have a GPU with 16GB+ VRAM
* The download is \~27GB and will take 15-30 minutes
- Check that CUDA toolkit is installed: `nvidia-smi`

### "Kling API: Account balance not enough"
- Go to https://app.klingai.com and purchase credits
* Kling v2.5 Turbo costs \~\$0.35 per generation (5-10s clip)
- Add credits and retry

### "Kling generation failed" / "Invalid model name"
- Check that `KLING_ACCESS_KEY` and `KLING_SECRET_KEY` are set in `.env`
- Valid models: `kling-v1`, `kling-v2-5-turbo`, `kling-v3`
- Invalid models (will fail): `kling-v2`, `kling-v2.6-pro`
- API mode uses `"std"` not `"standard"`

### "PyJWT import error"
- Install: `pip install PyJWT>=2.8.0`
- Required for Kling API authentication (JWT token generation)

## License

MIT License — see [LICENSE](LICENSE).

Copyright (c) 2026 Aryan Chavan, Arynoxtech

## Creator

**Aryan Chavan** — Arynoxtech

Built with OpenCode AI assistance.
