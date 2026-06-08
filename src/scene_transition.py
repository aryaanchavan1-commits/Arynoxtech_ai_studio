import subprocess
from pathlib import Path
import config
import uuid


TRANSITION_LABELS = {
    "crossfade": "fade",
    "fadeblack": "fadeblack",
    "fadewhite": "fadewhite",
    "slideleft": "slideleft",
    "slideright": "slideright",
    "slidetop": "slidetop",
    "slidebottom": "slidebottom",
}


def apply_transition_between_clips(
    clip_a: str,
    clip_b: str,
    output_path: str,
    transition_type: str = "crossfade",
    duration_frames: int = 30,
    fps: int = 25,
    target_width: int = 1920,
    target_height: int = 1080,
) -> str | None:
    ffmpeg_transition = TRANSITION_LABELS.get(transition_type, "fade")
    transition_sec = duration_frames / fps
    filter_complex = (
        f"[0:v]format=yuv420p,scale={target_width}:{target_height}:flags=lanczos,settb=AVTB[v0];"
        f"[1:v]format=yuv420p,scale={target_width}:{target_height}:flags=lanczos,settb=AVTB[v1];"
        f"[v0][v1]xfade=transition={ffmpeg_transition}:duration={transition_sec}:offset=offset"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", clip_a,
        "-i", clip_b,
        "-filter_complex", filter_complex,
        "-c:v", "libx264", "-preset", "slow", "-crf", "16",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return str(output_path) if Path(output_path).exists() else None
    except subprocess.CalledProcessError as e:
        print(f"Transition error: {e.stderr.decode() if e.stderr else ''}")
        return None


def stitch_with_transitions(
    clip_paths: list,
    output_path: str,
    transition_type: str = "crossfade",
    transition_frames: int = 30,
    fps: int = 25,
    target_width: int = 1920,
    target_height: int = 1080,
    on_progress=None,
) -> str | None:
    if not clip_paths:
        return None
    if len(clip_paths) == 1:
        inp = Path(clip_paths[0])
        if inp.exists():
            import shutil
            shutil.copy2(str(inp), str(output_path))
            return str(output_path) if Path(output_path).exists() else None

    current = clip_paths[0]
    for i in range(1, len(clip_paths)):
        if on_progress:
            on_progress(int(90 * (i - 1) / (len(clip_paths) - 1)),
                        f"Transition: stitching scene {i}/{len(clip_paths)}")
        segment_out = config.OUTPUT_DIR / f"transition_seg_{uuid.uuid4().hex[:8]}.mp4"
        result = apply_transition_between_clips(
            current, clip_paths[i], str(segment_out),
            transition_type, transition_frames,
            fps, target_width, target_height,
        )
        if result:
            current = str(segment_out)
        else:
            if on_progress:
                on_progress(0, f"Transition {i} failed, using concat fallback")
            fallback = config.OUTPUT_DIR / f"transition_fb_{uuid.uuid4().hex[:8]}.mp4"
            concat_list = config.OUTPUT_DIR / f"concat_list_{uuid.uuid4().hex[:8]}.txt"
            try:
                with open(str(concat_list), "w") as f:
                    f.write(f"file '{Path(current).as_posix()}'\n")
                    f.write(f"file '{Path(clip_paths[i]).as_posix()}'\n")
                concat_cmd = [
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", str(concat_list),
                    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                    "-pix_fmt", "yuv420p",
                    str(fallback),
                ]
                subprocess.run(concat_cmd, capture_output=True, check=True)
                if fallback.exists():
                    current = str(fallback)
            except Exception as ce:
                print(f"Concat fallback also failed: {ce}")
            finally:
                try:
                    concat_list.unlink()
                except Exception:
                    pass

    final = Path(output_path)
    final.parent.mkdir(parents=True, exist_ok=True)
    inp = Path(current)
    if inp.exists():
        import shutil
        shutil.copy2(str(inp), str(output_path))
    cleanup_paths = []
    for p in config.OUTPUT_DIR.glob("transition_seg_*"):
        cleanup_paths.append(p)
    for p in cleanup_paths:
        try:
            p.unlink()
        except Exception:
            pass
    for p in config.OUTPUT_DIR.glob("transition_fb_*"):
        try:
            p.unlink()
        except Exception:
            pass
    return str(output_path) if Path(output_path).exists() else None
