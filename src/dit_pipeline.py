import uuid
from pathlib import Path
from typing import List, Optional
import subprocess
import config
from src.dit_video import DiTVideoEngine, get_capability_report, download_model_if_needed
from src.scene_director import Scene, plan_scenes
from src.text_to_speech import generate_speech, generate_speech_from_text
from src.sarvam_tts import generate_speech_sarvam, sarvam_available
from src.video_enhancer import VideoUpscaler, ColorGrader, QualityEnhancer, combine_with_color_correction
from src.scene_transition import stitch_with_transitions


DIT_AVAILABLE = False
_dit_engine = None


def _get_dit_engine(on_progress=None):
    global _dit_engine
    if _dit_engine is None:
        _dit_engine = DiTVideoEngine()
        _dit_engine.initialize(on_progress=on_progress)
    return _dit_engine


def dit_available() -> bool:
    global DIT_AVAILABLE
    try:
        report = get_capability_report()
        DIT_AVAILABLE = report["dit_ready"]
        return DIT_AVAILABLE
    except Exception:
        return False


def _generate_audio(script: dict, voice: str, speed: float, language: str, gender: str, accent: str, use_sarvam: bool, on_progress=None) -> Path:
    full_text = " ".join(s["text"] for s in script["segments"])
    audio_path = None

    if use_sarvam and sarvam_available():
        if on_progress:
            on_progress(5, "Sarvam AI voice generation...")
        audio_path = generate_speech_sarvam(
            full_text, config.OUTPUT_DIR / "temp_audio.wav", language, gender,
        )

    if audio_path is None:
        if on_progress:
            on_progress(5, "Edge-TTS voice generation...")
        audio_path = generate_speech(script, voice, speed, language, gender, accent)
    if audio_path is None:
        audio_path = generate_speech_from_text(
            full_text, voice, speed, language, gender, accent,
        )
    if audio_path is None or not Path(audio_path).exists():
        raise RuntimeError("Failed to generate audio")
    return audio_path


def run_dit_pipeline(
    character_image_path: str = None,
    topic: str = "",
    manual_text: str = "",
    news_style: str = "neutral",
    voice: str = None,
    speed: float = 1.0,
    use_ai_script: bool = True,
    language: str = "english",
    gender: str = "female",
    accent: str = "indian",
    duration_minutes: int = 1,
    use_sarvam: bool = True,
    on_progress=None,
    studio_production: bool = True,
    anchor_name: str = "",
    channel_name: str = "",
    show_name: str = "",
    studio_theme: str = "blue",
    enable_intro: bool = True,
    enable_ticker: bool = True,
    music_path: str = None,
    use_generated_b_roll: bool = True,
    cinematic_style: str = "evening",
    visual_prompt: str = "",
) -> dict:
    output_id = uuid.uuid4().hex[:8]
    final_video = config.OUTPUT_DIR / f"dit_news_{output_id}.mp4"

    if on_progress:
        on_progress(0, "Planning scenes...")

    from src.news_script import generate_news_script, generate_manual_script

    if use_ai_script and topic.strip():
        script = generate_news_script(topic, news_style, language, duration_minutes)
    elif manual_text.strip():
        script = generate_manual_script("Custom News", manual_text)
    else:
        script = generate_manual_script("News", "This is a test broadcast.")

    scenes = plan_scenes(script, character_image_path, duration_minutes,
                         use_generated_b_roll, cinematic_style,
                         user_visual_prompt=visual_prompt)

    audio_path = _generate_audio(script, voice, speed, language, gender, accent, use_sarvam, on_progress)

    if on_progress:
        on_progress(10, "Initializing DiT video engine...")

    engine = _get_dit_engine(on_progress)
    if not engine.is_ready():
        raise RuntimeError("DiT engine requires GPU with 12GB+ VRAM")

    clip_paths = []
    total = len(scenes)

    for i, scene in enumerate(scenes):
        if on_progress:
            pct = 10 + int(75 * (i / total))
            on_progress(pct, f"Generating scene {i+1}/{total}: {scene.scene_type}...")

        clip_path = config.OUTPUT_DIR / f"dit_scene_{output_id}_{i}.mp4"
        result = engine.generate_clip(
            prompt=scene.prompt,
            duration_sec=scene.duration_sec,
            output_path=str(clip_path),
            image_path=scene.image_condition,
            on_progress=on_progress,
        )
        if result:
            clip_paths.append(result)
            upscaled_path = config.OUTPUT_DIR / f"dit_upscaled_{output_id}_{i}.mp4"
            up_result = VideoUpscaler.upscale_video_ffmpeg(
                result, str(upscaled_path),
                config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT,
                fps=12,
            )
            if up_result:
                clip_paths[-1] = up_result
        else:
            print(f"Scene {i} failed, skipping")

    if not clip_paths:
        raise RuntimeError("No clips were generated")

    if on_progress:
        on_progress(88, "Stitching scenes with professional transitions...")

    stitched_path = config.OUTPUT_DIR / f"dit_stitched_{output_id}.mp4"
    stitched_result = stitch_with_transitions(
        clip_paths, str(stitched_path),
        transition_type="crossfade",
        transition_frames=30,
        fps=25,
        target_width=config.OUTPUT_WIDTH,
        target_height=config.OUTPUT_HEIGHT,
        on_progress=on_progress,
    )

    if not stitched_result:
        raise RuntimeError("Failed to stitch scenes")

    if on_progress:
        on_progress(95, "Adding audio and final rendering...")

    render_cmd = [
        "ffmpeg", "-y",
        "-i", str(stitched_path),
        "-i", str(audio_path),
    ]

    filter_complex = "[1:a]volume=1.0[voice]"
    if music_path and Path(music_path).exists():
        render_cmd.extend(["-i", str(music_path)])
        filter_complex += f";[2:a]volume=0.10[music];[voice][music]amix=inputs=2:duration=first[a]"
        map_audio = "[a]"
    else:
        map_audio = "[voice]"

    render_cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "0:v",
        "-map", map_audio if music_path and Path(music_path).exists() else "1:a",
        "-c:v", "libx264", "-preset", "slow", "-crf", "16",
        "-c:a", "aac", "-b:a", "256k",
        "-pix_fmt", "yuv420p",
        "-shortest",
        str(final_video),
    ])

    try:
        subprocess.run(render_cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Final render error: {e.stderr.decode() if e.stderr else ''}")
        raise

    for p in clip_paths:
        try:
            Path(p).unlink()
        except Exception:
            pass
    try:
        stitched_path.unlink()
    except Exception:
        pass

    if on_progress:
        on_progress(100, "Complete!")

    return {
        "video_path": str(final_video),
        "script": script,
        "title": script.get("title", "DiT News Video"),
    }
