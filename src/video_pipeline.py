import subprocess
import time
import uuid
from pathlib import Path
import config
from src.news_script import generate_news_script, generate_manual_script
from src.text_to_speech import generate_speech, generate_speech_from_text
from src.elevenlabs_tts import generate_speech_elevenlabs, elevenlabs_available
from src.longcat_video import generate_video as generate_longcat_video, longcat_available
from src.lip_sync import LipSyncEngine, preprocess_image
from src.scene_composer import SceneComposer
from src.sarvam_tts import generate_speech_sarvam, sarvam_available
from src.character_compositor import prepare_character_scene
from src.motion_enhancer import enhance_video as enhance_motion_video
from src.dit_pipeline import run_dit_pipeline, dit_available
from src.kling_video import kling_available, run_kling_scene_generation, generate_scene_video
from src.video_enhancer import VideoUpscaler, ColorGrader, combine_with_color_correction
from src.scene_transition import stitch_with_transitions
from src.scene_director import plan_scenes


_pipeline_state = {"engine": None, "composer": None}


def _get_engine():
    if _pipeline_state["engine"] is None:
        _pipeline_state["engine"] = LipSyncEngine()
    return _pipeline_state["engine"]


def _get_composer():
    if _pipeline_state["composer"] is None:
        _pipeline_state["composer"] = SceneComposer(
            config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT
        )
    return _pipeline_state["composer"]


def cleanup_temp():
    for p in config.OUTPUT_DIR.glob("temp_*"):
        for _ in range(3):
            try:
                p.unlink()
                break
            except (PermissionError, OSError):
                time.sleep(0.1)
    temp_frames = config.OUTPUT_DIR / "temp_frames"
    if temp_frames.is_dir():
        for f in temp_frames.glob("*"):
            for _ in range(3):
                try:
                    f.unlink()
                    break
                except (PermissionError, OSError):
                    time.sleep(0.1)
        try:
            temp_frames.rmdir()
        except Exception:
            pass


def run_full_pipeline(
    character_image_path: str,
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
    use_elevenlabs: bool = False,
    use_longcat: bool = False,
    use_sarvam: bool = True,
    sarvam_voice: str = None,
    use_dit: bool = False,
    use_kling: bool = False,
    on_progress=None,
    studio_production: bool = False,
    anchor_name: str = "",
    channel_name: str = "Arynoxtech",
    show_name: str = "AI Studio",
    studio_theme: str = "blue",
    enable_intro: bool = True,
    enable_ticker: bool = True,
    music_path: str = None,
    motion_enhancement: bool = True,
    use_generated_b_roll: bool = True,
    cinematic_style: str = "evening",
    visual_prompt: str = "",
) -> dict:
    use_dit_actual = use_dit and dit_available()
    use_longcat_actual = use_longcat and longcat_available()
    use_kling_actual = use_kling and kling_available()

    if use_dit_actual and not use_longcat_actual and not use_kling_actual:
        return run_dit_pipeline(
            character_image_path=character_image_path, topic=topic,
            manual_text=manual_text, news_style=news_style,
            voice=voice, speed=speed, use_ai_script=use_ai_script,
            language=language, gender=gender, accent=accent,
            duration_minutes=duration_minutes, use_sarvam=use_sarvam,
            on_progress=on_progress, studio_production=studio_production,
            anchor_name=anchor_name, channel_name=channel_name,
            show_name=show_name, studio_theme=studio_theme,
            enable_intro=enable_intro, enable_ticker=enable_ticker,
            music_path=music_path, use_generated_b_roll=use_generated_b_roll,
            cinematic_style=cinematic_style, visual_prompt=visual_prompt,
        )

    output_id = uuid.uuid4().hex[:8]
    raw_video = config.OUTPUT_DIR / f"raw_{output_id}.mp4"
    final_video = config.OUTPUT_DIR / f"news_video_{output_id}.mp4"

    if on_progress:
        on_progress(0, "Preparing...")
    preprocess_image(character_image_path)

    if on_progress:
        on_progress(5, "Generating script...")

    if use_ai_script and topic.strip():
        script = generate_news_script(topic, news_style, language, duration_minutes)
    elif manual_text.strip():
        script = generate_manual_script("Custom News", manual_text)
    else:
        script = generate_manual_script("News", "This is a test broadcast.")

    if on_progress:
        on_progress(20, "Generating voiceover...")

    full_text = " ".join(s["text"] for s in script["segments"])
    audio_path = None

    if use_sarvam and sarvam_available():
        if on_progress:
            on_progress(20, "Sarvam AI voice...")
        audio_path = generate_speech_sarvam(
            full_text, config.OUTPUT_DIR / "temp_audio.wav", language, gender,
            voice=sarvam_voice,
        )

    if audio_path is None and use_elevenlabs and elevenlabs_available():
        audio_path = generate_speech_elevenlabs(
            full_text, config.OUTPUT_DIR / "temp_audio.mp3", language
        )

    if audio_path is None:
        audio_path = generate_speech(script, voice, speed, language, gender, accent)
    if audio_path is None:
        audio_path = generate_speech_from_text(
            full_text, voice, speed, language, gender, accent,
        )
    if audio_path is None or not audio_path.exists():
        raise RuntimeError("Failed to generate audio")

    # === HELPERS: audio timing & segment extraction ===
    def _get_audio_duration_sec(audio_p: Path) -> float:
        try:
            cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                   "-of", "csv=p=0", str(audio_p)]
            r = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(r.stdout.strip())
        except Exception:
            return duration_minutes * 60.0

    def _stage_interleaved_edit(scenes_list, lc_video_path, audio_dur):
        """Build broadcast edit: anchor (LongCat) ↔ broll (Kling) interleaved."""
        segs = script.get("segments", [])
        if not segs:
            return None

        total_chars = sum(len(s["text"]) for s in segs)
        cum_time = 0.0
        seg_times = []
        for s in segs:
            ratio = len(s["text"]) / total_chars if total_chars else 1.0 / max(len(segs), 1)
            dur = audio_dur * ratio
            seg_times.append({"start": cum_time, "end": cum_time + dur})
            cum_time += dur

        scene_clips = []
        scene_index = 0

        for i, sc in enumerate(scenes_list):
            scene_index = min(i, len(seg_times) - 1)
            st_info = seg_times[scene_index]
            seg_dur = st_info["end"] - st_info["start"]

            if sc.scene_type == "broll":
                if on_progress:
                    on_progress(60, f"Kling: B-roll scene {i+1}/{len(scenes_list)}...")
                clip_p = config.OUTPUT_DIR / f"seg_broll_{output_id}_{i}.mp4"
                result = generate_scene_video(
                    prompt=sc.prompt,
                    duration_sec=int(min(seg_dur, 10)),
                    output_path=str(clip_p),
                    on_progress=on_progress,
                )
                if result:
                    up_p = config.OUTPUT_DIR / f"seg_broll_up_{output_id}_{i}.mp4"
                    up = VideoUpscaler.upscale_video_ffmpeg(
                        result, str(up_p),
                        config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT,
                    )
                    if up:
                        Path(result).unlink(missing_ok=True)
                        scene_clips.append(up)
                    else:
                        scene_clips.append(result)
            else:
                if lc_video_path and Path(lc_video_path).exists():
                    seg_p = config.OUTPUT_DIR / f"seg_anchor_{output_id}_{i}.mp4"
                    if on_progress:
                        on_progress(65, f"Editing anchor scene {i+1}/{len(scenes_list)}...")
                    trim_cmd = [
                        "ffmpeg", "-y",
                        "-i", str(lc_video_path),
                        "-ss", str(st_info["start"]),
                        "-t", str(max(seg_dur, 1.0)),
                        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                        "-pix_fmt", "yuv420p",
                        "-an",
                        str(seg_p),
                    ]
                    try:
                        subprocess.run(trim_cmd, capture_output=True, check=True)
                        if seg_p.exists():
                            scene_clips.append(str(seg_p))
                    except Exception:
                        pass

        if len(scene_clips) < 1:
            return None

        if on_progress:
            on_progress(82, "Stitching broadcast segments with crossfades...")

        stitched = config.OUTPUT_DIR / f"interleaved_{output_id}.mp4"
        final_edit = stitch_with_transitions(
            scene_clips, str(stitched),
            transition_type="crossfade",
            transition_frames=20,
            fps=25,
            target_width=config.OUTPUT_WIDTH,
            target_height=config.OUTPUT_HEIGHT,
            on_progress=on_progress,
        )
        if not final_edit or not Path(final_edit).exists():
            return None

        if on_progress:
            on_progress(90, "Mixing audio track...")
        audio_out = config.OUTPUT_DIR / f"interleaved_audio_{output_id}.mp4"
        audio_cmd = [
            "ffmpeg", "-y",
            "-i", final_edit,
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "256k",
            "-map", "0:v", "-map", "1:a",
            "-shortest",
            str(audio_out),
        ]
        try:
            subprocess.run(audio_cmd, capture_output=True, check=True)
            for p in scene_clips:
                Path(p).unlink(missing_ok=True)
            Path(final_edit).unlink(missing_ok=True)
            return str(audio_out)
        except Exception:
            if Path(final_edit).exists():
                return final_edit
            return None

    # === MODE 1: LongCat + Kling (professional broadcast edit) ===
    if use_longcat_actual and use_kling_actual:
        if on_progress:
            on_progress(30, "LongCat: generating avatar video...")
        lc_video = generate_longcat_video(
            character_image_path, str(audio_path),
            str(raw_video), duration_minutes, "auto", on_progress,
        )
        if lc_video:
            lc_path = Path(lc_video)
            if on_progress:
                on_progress(50, "Planning broadcast scene sequence...")
            scenes = plan_scenes(
                script=script,
                character_image_path=character_image_path,
                duration_minutes=duration_minutes,
                use_generated_b_roll=True,
                cinematic_style=cinematic_style,
                user_visual_prompt=visual_prompt,
            )
            audio_dur = _get_audio_duration_sec(audio_path)
            interleaved = _stage_interleaved_edit(scenes, lc_path, audio_dur)
            if interleaved and Path(interleaved).exists():
                raw_video = Path(interleaved)
            else:
                print("Interleaved edit failed, using raw LongCat video")
                raw_video = lc_path
        else:
            print("LongCat failed, falling back to Wav2Lip")
            use_longcat_actual = False

    # === MODE 1b: LongCat + DiT combined (existing) ===
    elif use_longcat_actual and use_dit_actual:
        if on_progress:
            on_progress(30, "LongCat: generating anchor video...")
        lc_output = generate_longcat_video(
            character_image_path, str(audio_path),
            str(raw_video), duration_minutes, "auto", on_progress,
        )
        if not lc_output:
            use_longcat_actual = False
        else:
            raw_video = Path(lc_output)
            if on_progress:
                on_progress(70, "DiT: generating background scenes...")
            dit_result = run_dit_pipeline(
                character_image_path=None, topic=topic,
                manual_text=manual_text, news_style=news_style,
                voice=voice, speed=speed, use_ai_script=use_ai_script,
                language=language, gender=gender, accent=accent,
                duration_minutes=duration_minutes, use_sarvam=use_sarvam,
                on_progress=on_progress, studio_production=False,
                use_generated_b_roll=use_generated_b_roll,
                cinematic_style=cinematic_style,
                visual_prompt=visual_prompt,
            )
            dit_bg_path = dit_result.get("video_path") if dit_result else None
            if dit_bg_path and Path(dit_bg_path).exists():
                if on_progress:
                    on_progress(85, "Compositing LongCat + DiT with color grading...")
                composite_out = config.OUTPUT_DIR / f"composite_{output_id}.mp4"
                combined = combine_with_color_correction(
                    str(raw_video), dit_bg_path, str(composite_out),
                    audio_path=str(audio_path),
                    target_width=config.OUTPUT_WIDTH,
                    target_height=config.OUTPUT_HEIGHT,
                    on_progress=on_progress,
                )
                if combined:
                    raw_video = Path(combined)

    # === MODE 2: LongCat only ===
    elif use_longcat_actual:
        if on_progress:
            on_progress(40, "LongCat: generating avatar video...")
        lc_result = generate_longcat_video(
            character_image_path, str(audio_path),
            str(raw_video), duration_minutes, "auto", on_progress,
        )
        if not lc_result:
            print("LongCat failed, falling back to Wav2Lip")
            use_longcat_actual = False

    # === MODE 3: Wav2Lip (default fallback) ===
    if not use_longcat_actual:
        if studio_production:
            if on_progress:
                on_progress(2, "Compositing character into scene...")
            composited = prepare_character_scene(
                character_image_path,
                str(config.OUTPUT_DIR / f"composited_{output_id}.png"),
                config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT,
                theme_id=studio_theme,
            )
            if composited:
                character_image_path = composited

        if on_progress:
            on_progress(40, "Running lip sync (Wav2Lip)...")
        engine = _get_engine()
        for progress, total in engine.generate(
            character_image_path, str(audio_path), str(raw_video),
        ):
            pct = 40 + int(60 * progress / total)
            if on_progress:
                on_progress(pct, f"Lip sync: frame {progress}/{total}")

        if motion_enhancement and raw_video.exists():
            if on_progress:
                on_progress(92, "Enhancing motion...")
            motion_video = config.OUTPUT_DIR / f"motion_{output_id}.mp4"
            enhanced = enhance_motion_video(str(raw_video), str(motion_video),
                fps=config.WAV2LIP_FPS, on_progress=on_progress)
            if enhanced:
                raw_video.unlink(missing_ok=True)
                raw_video = Path(enhanced)

    # === Studio Production ===
    if studio_production and raw_video.exists():
        if on_progress:
            on_progress(95, "Composing studio scene...")
        headlines = [s["text"][:80] for s in script.get("segments", []) if s["text"]]
        composer = _get_composer()
        composed = composer.compose(
            input_video=str(raw_video), output_path=str(final_video),
            audio_path=str(audio_path),
            anchor_name=anchor_name or script.get("title", "News")[:40],
            channel_name=channel_name, show_name=show_name,
            headlines=headlines, ticker_text="", theme=studio_theme,
            enable_intro=enable_intro, enable_outro=True,
            enable_ticker=enable_ticker, enable_lower_third=True,
            music_path=music_path, pip_composite=use_longcat_actual,
            on_progress=on_progress,
        )
        if composed:
            raw_video.unlink(missing_ok=True)
            cleanup_temp()
            if on_progress:
                on_progress(100, "Done!")
            return {"video_path": composed, "script": script, "title": script.get("title", "News Video")}

    if raw_video != final_video:
        import shutil
        shutil.move(str(raw_video), str(final_video))

    cleanup_temp()
    if on_progress:
        on_progress(100, "Done!")
    return {"video_path": str(final_video), "script": script, "title": script.get("title", "News Video")}
