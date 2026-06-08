import os
import traceback
from pathlib import Path
import streamlit as st
import config
from src.video_pipeline import run_full_pipeline

for key in ["dit_ready", "use_dit", "use_generated_b_roll", "cinematic_style",
             "image_loaded", "image_path", "selected_news", "visual_prompt", "use_runway"]:
    if key not in st.session_state:
        st.session_state[key] = False if key in ("dit_ready", "use_dit", "use_generated_b_roll", "image_loaded", "use_runway") else ""

st.set_page_config(page_title="Arynox AI Studio", page_icon="🎬", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background: #0e1117; }
    .main-header { text-align: center; padding: 1.5rem 0; }
    .main-header h1 { font-size: 2.5rem; font-weight: 700; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .error-box { background: #2d1b1b; border: 1px solid #ff4444; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; }
    .error-box p { color: #ff6666; margin: 0; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>Arynox AI Studio</h1><p style="color: #8892b0;">Turn any portrait into a talking AI video — news, greetings, presentations & more</p></div>', unsafe_allow_html=True)

def load_utils():
    from src.utils import resolve_device as _rd, ffmpeg_available as _fa, get_voice_options as _gvo
    return _rd(), _fa(), _gvo

try:
    device, has_ffmpeg, voice_options = load_utils()
except Exception as e:
    device, has_ffmpeg, voice_options = "cpu", False, []
    st.sidebar.error(f"System detection failed: {e}")

with st.sidebar:
    st.markdown("### Configuration")

    lang = st.selectbox("Language", options=[
        "english", "hindi", "marathi", "tamil", "telugu", "kannada",
        "malayalam", "bengali", "gujarati", "punjabi", "odia",
    ], format_func=lambda x: {
        "english":"English","hindi":"हिन्दी (Hindi)","marathi":"मराठी (Marathi)",
        "tamil":"தமிழ் (Tamil)","telugu":"తెలుగు (Telugu)","kannada":"ಕನ್ನಡ (Kannada)",
        "malayalam":"മലയാളം (Malayalam)","bengali":"বাংলা (Bengali)",
        "gujarati":"ગુજરાતી (Gujarati)","punjabi":"ਪੰਜਾਬੀ (Punjabi)",
        "odia":"ଓଡ଼ିଆ (Odia)",
    }[x], index=0)

    api_key = st.text_input("Groq API Key", type="password",
        value=config.GROQ_API_KEY if config.GROQ_API_KEY and config.GROQ_API_KEY != "gsk_your_groq_api_key_here" else "",
        help="Get a free key at https://console.groq.com")
    st.caption("Without a key, uses a fallback script template.")

    st.markdown("### Voice Settings")
    gender = st.radio("Anchor Gender", ["female", "male"], horizontal=True, index=0)

    if lang != "english":
        accent = "default"
        preferred_voice = config.LANGUAGE_VOICES[lang][gender]["default"]
    else:
        accent_options = {"indian": "Indian English", "us": "US English", "uk": "UK English"}
        accent = st.selectbox("Accent", options=list(accent_options.keys()),
            format_func=lambda x: accent_options[x], index=0)
        preferred_voice = config.LANGUAGE_VOICES["english"][gender][accent]

    _lang_prefixes = {
        "english":"en-","hindi":"hi-","marathi":"mr-","tamil":"ta-","telugu":"te-",
        "kannada":"kn-","malayalam":"ml-","bengali":"bn-","gujarati":"gu-",
        "punjabi":"pa-","odia":"or-",
    }
    if voice_options:
        prefix = _lang_prefixes.get(lang, "en-")
        filtered_voices = [v for v in voice_options if v["name"].startswith(prefix)]
    else:
        filtered_voices = []

    if filtered_voices:
        voice_names = [v["name"] for v in filtered_voices]
        default_idx = 0
        for i, v in enumerate(filtered_voices):
            if v["name"] == preferred_voice:
                default_idx = i; break
        selected_voice = st.selectbox("Voice", voice_names, index=default_idx)
    else:
        selected_voice = st.text_input("Voice", value=preferred_voice)
        st.caption("Run `edge-tts --list-voices` to see all options")

    speed = st.slider("Speech Speed", 0.5, 2.0, 1.0, 0.1)

    st.markdown("### Voice Engine")
    use_sarvam = st.checkbox("Use Sarvam AI", value=True,
        help="Indian-optimized TTS with natural Marathi/Hindi voices.")
    if use_sarvam:
        sarvam_key = st.text_input("Sarvam AI API Key", type="password",
            value=config.SARVAM_API_KEY if config.SARVAM_API_KEY else "", key="sarvam_key_input")
        sarvam_voice = st.selectbox("Sarvam Voice",
            options=["auto", "sumit", "ishita", "amit", "priya", "neel", "shubh", "anushka", "arjun", "kavya"],
            format_func=lambda x: {
                "auto": f"Auto ({'sumit' if gender == 'male' else 'ishita'} for Marathi)",
                "sumit": "Sumit (Male - Marathi/Hindi)",
                "ishita": "Ishita (Female - Marathi/Hindi)",
                "amit": "Amit (Male)",
                "priya": "Priya (Female)",
                "neel": "Neel (Male)",
                "shubh": "Shubh (Male)",
                "anushka": "Anushka (Female)",
                "arjun": "Arjun (Male)",
                "kavya": "Kavya (Female)",
            }[x], index=0,
            help="Select 'auto' to use gender-based default. Sumit and Ishita are optimized for Marathi.")
        if sarvam_key:
            voice_display = sarvam_voice if sarvam_voice != "auto" else ("sumit" if gender == "male" else "ishita")
            st.caption(f"Ready - voice: {voice_display}")
        else:
            st.caption("Enter your Sarvam API key above")
    else:
        sarvam_voice = None
        st.caption("Using Edge-TTS (free)")

    st.markdown("### Duration")
    duration_minutes = st.slider("Video Length", 1, 5, 1, 1, format="%d min")

    st.markdown("### System")
    st.code(f"Device: {device.upper()}\nFFmpeg: {'OK' if has_ffmpeg else 'Not found'}")
    if not has_ffmpeg:
        st.warning("Install FFmpeg: `winget install ffmpeg` or add to PATH")

    st.markdown("### Video Engine")
    try:
        from src.longcat_video import longcat_setup_status, setup_longcat
        lc = longcat_setup_status()
        if lc.get("gpu_ready") and lc.get("complete"):
            use_longcat = st.checkbox("Use LongCat AI", value=False,
                help="Full body avatar with natural movement. Combine with DiT below for AI-generated backgrounds.")
            if use_longcat:
                vram = lc.get("vram_gb", 0)
                qual = "Ultra" if vram >= 22 else "High" if vram >= 16 else "Standard"
                st.caption(f"LongCat ready ({qual}, {vram:.0f}GB VRAM)")
        elif lc.get("gpu_ready"):
            use_longcat = False
            st.info("LongCat AI Avatar")
            if not lc.get("repo_cloned"):
                st.caption("Repo: not cloned (~2GB)")
            if not lc.get("weights_downloaded"):
                st.caption("Weights: not found (~27GB)")
            if st.button("Download LongCat AI (27GB)", type="primary", use_container_width=True, key="dl_lc"):
                dl_p = st.progress(0, text="Starting..."); dl_s = st.empty()
                def _dl(pct, msg): dl_p.progress(pct/100, text=msg); dl_s.info(msg)
                if setup_longcat(on_progress=_dl):
                    dl_s.success("LongCat ready!"); st.rerun()
                else:
                    dl_s.error("Download failed")
        else:
            use_longcat = False
            if lc.get("repo_cloned") or lc.get("weights_downloaded"):
                st.caption("LongCat files found but no GPU (needs 16GB+ VRAM)")
    except Exception as e:
        use_longcat = False
        st.caption("LongCat: not available")

    if device == "cpu":
        st.info("Using CPU. A GPU makes processing 20-50x faster.")

    st.markdown("### DiT AI Video Engine")
    try:
        from src.dit_video import get_capability_report
        dit_status = get_capability_report()
        if dit_status["dit_ready"]:
            st.session_state.dit_ready = True
            st.session_state.use_dit = st.checkbox("Use DiT AI Video Generation", value=True,
                help="Generates cinematic backgrounds/B-roll. Combine with LongCat above for full avatar + background.")
            if st.session_state.use_dit:
                vram = dit_status["vram_gb"]
                qual = "960p Ultra" if vram >= 24 else "720p High" if vram >= 16 else "480p Standard"
                st.caption(f"DiT: {qual} - upscaled to 1080p")
                st.session_state.use_generated_b_roll = st.checkbox("Generate B-Roll from news topic", value=True)
                st.markdown("#### Cinematic Style")
                st.session_state.cinematic_style = st.selectbox("Lighting & Mood",
                    options=["evening", "morning", "night", "breaking"],
                    format_func=lambda x: {"evening":"Evening Studio","morning":"Morning Bright",
                        "night":"Night Broadcast","breaking":"Breaking News"}[x],
                    index=0, label_visibility="collapsed")
            else:
                st.session_state.use_generated_b_roll = False
                st.session_state.cinematic_style = "evening"
        elif dit_status["gpu_available"]:
            st.session_state.dit_ready = False; st.session_state.use_dit = False
            st.session_state.use_generated_b_roll = False; st.session_state.cinematic_style = "evening"
            st.info("DiT Video Engine")
            st.caption(f"GPU: {dit_status['gpu_name']} ({dit_status['vram_gb']:.0f}GB) - needs 12GB+ VRAM")
        else:
            st.session_state.dit_ready = False; st.session_state.use_dit = False
            st.session_state.use_generated_b_roll = False; st.session_state.cinematic_style = "evening"
            st.caption("DiT needs NVIDIA GPU with 12GB+ VRAM (RTX 4080/4090)")
    except Exception as e:
        st.session_state.dit_ready = False; st.session_state.use_dit = False
        st.session_state.use_generated_b_roll = False; st.session_state.cinematic_style = "evening"
        st.caption(f"DiT detection failed")

    st.markdown("### Runway ML Video Engine")
    try:
        from src.runway_video import runway_available as runway_avail, runway_api_configured
        if runway_avail():
            use_runway = st.checkbox("Use Runway ML (Cinematic Scenes)", value=True,
                help="Runway Gen-4 Turbo generates cinema-quality B-roll. Better than Kling, same or lower cost.")
            st.caption("Runway API: connected")
            if use_runway:
                runway_model = st.selectbox("Runway Model",
                    options=["gen4_turbo", "gen3a_turbo"],
                    format_func=lambda x: {
                        "gen4_turbo": "Gen-4 Turbo (best value, $0.05/s)",
                        "gen3a_turbo": "Gen-3a Turbo (fastest, $0.05/s)",
                    }[x], index=0)
                os.environ["RUNWAY_MODEL"] = runway_model
                config.RUNWAY_MODEL = runway_model
                st.session_state.use_runway = True
            else:
                st.session_state.use_runway = False
        else:
            use_runway = False
            st.session_state.use_runway = False
            st.info("Runway ML Video")
            st.caption("Set RUNWAY_API_KEY in .env")
    except Exception as e:
        use_runway = False
        st.session_state.use_runway = False
        st.caption(f"Runway: not available ({e})")

    if st.session_state.get("use_runway", False) and use_longcat:
        st.success("BROADCAST PRO: LongCat avatar + Runway cinematic scenes (interleaved edit with crossfades)")
    elif st.session_state.get("use_dit", False) and use_longcat:
        st.success("Combined Mode: LongCat + DiT scenes")
    elif st.session_state.get("use_dit", False):
        st.info("DiT generates full video scenes")
    elif use_longcat:
        st.info("LongCat generates full avatar video")
    elif st.session_state.get("use_runway", False):
        st.info("Runway generates cinematic broadcast scenes")

    if st.session_state.get("use_dit", False):
        with st.expander("DiT Advanced Settings", expanded=False):
            st.session_state["video_crf"] = st.slider("Video Quality (lower=better)", 14, 28, 16, 1)
            st.session_state["video_preset"] = st.select_slider("Encoding Speed/Quality",
                options=["ultrafast","veryfast","faster","fast","medium","slow","veryslow"], value="slow")
            st.session_state["enable_denoise"] = st.checkbox("Denoise", value=True)
            st.session_state["enable_sharpen"] = st.checkbox("Sharpening", value=True)
            st.session_state["enable_color_grade"] = st.checkbox("Color Grading", value=True)
            st.session_state["transition_type"] = st.selectbox("Scene Transition",
                options=["crossfade","fadeblack","slideleft","slideright"],
                format_func=lambda x: {"crossfade":"Crossfade","fadeblack":"Fade to Black",
                    "slideleft":"Slide Left","slideright":"Slide Right"}[x], index=0)

    st.markdown("### Quality Preset")
    quality_preset = st.selectbox("Output Quality",
        options=["cinema", "premium", "standard"],
        format_func=lambda x: {
            "standard": "Standard — fast, good quality, $2-3/video",
            "premium": "Premium — high quality, face enhance, 30fps, $3-5/video",
            "cinema": "Cinema — 60fps, 4K upscale, face enhance, denoise, $5-8/video",
        }[x], index=0,
        help="Higher presets apply multi-pass post-processing: denoise, sharpen, face enhancement, upscale, frame interpolation.")
    st.caption({
        "standard": "CRF 18, 25fps, 1080p",
        "premium": "CRF 16, 30fps, face enhance, 1080p",
        "cinema": "CRF 14, 60fps, 4K, face enhance, denoise + sharpen",
    }[quality_preset])

    st.markdown("### Studio Production")
    studio_production = st.checkbox("Enable Studio Overlay", value=True)
    if studio_production:
        anchor_name = st.text_input("Presenter Name", value="Aryan")
        channel_name = st.text_input("Brand / Channel Name", value="Arynoxtech")
        show_name = st.text_input("Video Title", value="AI Studio")
        try:
            from src.theme_manager import list_themes
            themes = list_themes()
            theme_options = {}
            for tid, tinfo in themes.items():
                theme_options[tid] = f"{tinfo['name']}"
            theme_ids = list(theme_options.keys())
            default_theme_idx = theme_ids.index("blue") if "blue" in theme_ids else 0
            studio_theme = theme_ids[st.selectbox("Theme Background", [theme_options[t] for t in theme_ids],
                index=default_theme_idx)]
        except Exception:
            studio_theme = "blue"

        uploaded_theme = st.file_uploader("Upload Custom Background", type=["jpg","jpeg","png"])
        if uploaded_theme:
            (config.THEMES_DIR / uploaded_theme.name).write_bytes(uploaded_theme.getvalue())
            st.caption(f"Saved: {uploaded_theme.name}")

        enable_intro = st.checkbox("Show Intro/Outro", value=True)
        enable_ticker = st.checkbox("Show News Ticker", value=True)
        _me_disabled = st.session_state.get("use_dit", False) or use_longcat
        me_default = not _me_disabled
        motion_enhancement = st.checkbox("Motion Enhancement", value=me_default,
            disabled=_me_disabled,
            help="Head movement, blinking, breathing. Not needed when LongCat or DiT is active.")
        music_file = st.file_uploader("Background Music (optional)", type=["mp3","wav","m4a"])
    else:
        anchor_name = ""; channel_name = ""; show_name = ""
        studio_theme = "blue"; enable_intro = False; enable_ticker = False
        motion_enhancement = not (st.session_state.get("use_dit", False) or use_longcat); music_file = None

    st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Create Video", "Live News", "How It Works"])

with tab1:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### 1. Upload Character Image")
        uploaded_img = st.file_uploader("Upload a front-facing portrait (JPG/PNG)", type=["jpg","jpeg","png"], label_visibility="collapsed")
        if uploaded_img:
            img_path = config.UPLOADS_DIR / "character.png"
            img_path.write_bytes(uploaded_img.getvalue())
            st.image(uploaded_img, caption="Character", use_container_width=True)
            st.session_state.image_loaded = True
            st.session_state.image_path = str(img_path)
        elif st.session_state.get("image_path"):
            st.image(st.session_state.image_path, caption="Character", use_container_width=True)
        else:
            st.info("Upload a portrait photo to begin")

    with col2:
        st.markdown("### 2. Script Content")
        input_mode = st.radio("Input mode:", ["AI Generate from Topic", "Write Manually"], horizontal=True)
        if input_mode == "AI Generate from Topic":
            preselected = st.session_state.pop("selected_news", "")
            topic = st.text_input("Topic / Prompt", placeholder="e.g. AI breakthrough in medicine, birthday greeting, product launch...", value=preselected)
            news_style = st.select_slider("Tone", options=["neutral", "professional", "casual", "dramatic"], value="neutral")
            use_ai = True; manual_text = ""
        else:
            manual_text = st.text_area("Write your script", height=200, placeholder="Write the text your presenter will speak...")
            topic = ""; news_style = "neutral"; use_ai = False

        st.markdown("### Visual Style")
        use_nlp = st.checkbox("NLP Amplifier (auto-enhance prompts)", value=True,
            help="Uses AI to turn brief descriptions into cinematic production-quality prompts. Better images, less API spending.")
        vis_prompt = st.text_area("Scene description (optional)", height=80,
            placeholder="e.g. A sleek modern studio with blue lighting, or a tropical beach background...",
            key="visual_prompt_input",
            help="Describe the visual scene / background. Leave empty for AI to auto-generate based on topic.")
        if vis_prompt:
            st.session_state.visual_prompt = vis_prompt

        st.markdown("### 3. Generate")
        can_gen = (st.session_state.get("image_loaded", False)) and ((use_ai and topic.strip()) or (not use_ai and manual_text.strip()))

        if st.button("Generate Video", type="primary", use_container_width=True, disabled=not can_gen):
            if not st.session_state.get("image_path"):
                st.error("Please upload a character image first")
            else:
                progress_ph = st.empty(); status_ph = st.empty(); video_ph = st.empty()
                progress_bar = progress_ph.progress(0, text="Starting...")

                def on_progress(pct, msg):
                    try:
                        progress_bar.progress(pct/100, text=msg)
                        status_ph.info(msg)
                    except Exception:
                        pass

                if api_key:
                    os.environ["GROQ_API_KEY"] = api_key; config.GROQ_API_KEY = api_key
                if use_sarvam:
                    sk = st.session_state.get("sarvam_key_input", "") or config.SARVAM_API_KEY
                    if sk:
                        config.SARVAM_API_KEY = sk; os.environ["SARVAM_API_KEY"] = sk

                music_path = None
                if studio_production and music_file:
                    music_path = str(config.ASSETS_DIR / "bg_music.mp3")
                    Path(music_path).write_bytes(music_file.getvalue())

                try:
                    sarvam_voice_selected = sarvam_voice if sarvam_voice and sarvam_voice != "auto" else None
                    use_nlp_val = use_nlp and bool(api_key or config.GROQ_API_KEY)
                    nlp_style = st.session_state.get("cinematic_style", "evening")
                    if use_nlp_val and vis_prompt:
                        try:
                            from src.nlp_amplifier import amplify_prompt
                            enhanced = amplify_prompt(vis_prompt, "cinematic" if nlp_style != "breaking" else "dramatic")
                            if enhanced and enhanced != vis_prompt:
                                st.session_state.visual_prompt = enhanced
                                if on_progress:
                                    on_progress(1, f"NLP: prompt enhanced ({len(vis_prompt)}→{len(enhanced)} chars)")
                        except Exception:
                            pass
                    result = run_full_pipeline(
                        character_image_path=st.session_state.image_path,
                        topic=topic, manual_text=manual_text, news_style=news_style,
                        voice=selected_voice, speed=speed, use_ai_script=use_ai,
                        language=lang, gender=gender, accent=accent,
                        duration_minutes=duration_minutes, use_elevenlabs=False,
                        use_sarvam=use_sarvam, sarvam_voice=sarvam_voice_selected,
                        use_longcat=use_longcat,
                        use_dit=st.session_state.get("use_dit", False),
                        use_runway=st.session_state.get("use_runway", False),
                        on_progress=on_progress, studio_production=studio_production,
                        anchor_name=anchor_name, channel_name=channel_name,
                        show_name=show_name, studio_theme=studio_theme,
                        enable_intro=enable_intro, enable_ticker=enable_ticker,
                        motion_enhancement=motion_enhancement, music_path=music_path,
                        use_generated_b_roll=st.session_state.get("use_generated_b_roll", True),
                        cinematic_style=st.session_state.get("cinematic_style", "evening"),
                        visual_prompt=st.session_state.get("visual_prompt", ""),
                        quality_preset=quality_preset,
                    )
                    progress_bar.progress(1.0, text="Complete!")
                    status_ph.success(f"Video generated: {result['title']}")
                    with video_ph.container():
                        st.markdown("### Your News Video")
                        if Path(result["video_path"]).exists():
                            st.video(result["video_path"])
                            st.download_button("Download Video", data=Path(result["video_path"]).read_bytes(),
                                file_name=f"{result['title'][:30].replace(' ','_')}.mp4", mime="video/mp4",
                                use_container_width=True)
                except Exception as e:
                    progress_bar.progress(1.0)
                    msg = str(e) if str(e) else "Unknown error during generation"
                    status_ph.error(f"Generation failed: {msg}")
                    if on_progress:
                        on_progress(100, f"Failed: {msg}")

with tab2:
    st.markdown("### Live News Fetch")
    st.caption("Requires NEWSAPI_KEY in `.env`. Free tier: 100 requests/day.")
    if config.NEWSAPI_KEY:
        try:
            from src.news_script import fetch_live_news
            news_cat = st.selectbox("Category", ["general","business","technology","sports","entertainment","health","science"])
            if st.button("Fetch Latest News", use_container_width=True):
                articles = fetch_live_news(news_cat)
                if articles:
                    for a in articles:
                        with st.expander(a["title"]):
                            st.write(a.get("description", "No description"))
                            if st.button("Use This Story", key=a["title"]):
                                st.session_state["selected_news"] = a["title"]
                                st.success(f"Selected: {a['title'][:50]}...")
                else:
                    st.warning("No articles found or API limit reached")
        except Exception as e:
            st.warning(f"News fetch unavailable: {e}")
    else:
        st.info("Add NEWSAPI_KEY to `.env` to enable live news fetching")

with tab3:
    st.markdown("""
    **Pipeline (Production Mode - LongCat + Runway):**
    1. **Script** - Groq AI generates news script from topic
    2. **Voice** - Sarvam AI Bulbul v3 generates natural voiceover
    3. **Avatar** - LongCat creates full-body talking anchor with lip-sync from audio
    4. **B-roll** - Runway Gen-4 Turbo generates cinematic scene clips matching each story segment
    5. **Edit** - Anchor segments + B-roll clips interleaved with crossfade transitions (broadcast style)
    6. **Quality** - Multi-pass post-processing: denoise, sharpen, face enhance, 4K upscale, 60fps interpolation
    7. **Studio** - Intro/outro, ticker, lower thirds, branded overlays added
    8. **Master** - CRF 14 veryslow encode with AAC 320kbps audio, 3840x2160 output

    **Quality Presets:**
    - **Standard** - CRF 18, 25fps, 1080p - $2-3/video
    - **Premium** - CRF 16, 30fps, face enhance, 1080p - $3-5/video
    - **Cinema** - CRF 14, 60fps, 4K, face enhance, denoise + sharpen - $5-8/video

    **Fallback modes:** Wav2Lip (any GPU) + Runway, or Wav2Lip standalone on RTX 3050

    **Tips:** LongCat needs 16GB+ VRAM. On RTX 3050, use Wav2Lip + Runway for cinematic B-roll with talking head.
    """)
