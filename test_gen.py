import time
from pathlib import Path
import config
from src.video_pipeline import run_full_pipeline

def on_progress(pct, msg):
    print(f"  [{int(pct)}%] {msg}")

script_text = """Good evening and welcome to today's broadcast. We are covering a major breakthrough in artificial intelligence that promises to transform our world.

Researchers have developed a new neural network that can learn from far less data than traditional models.

That concludes our update. Thank you for watching.
"""

print("Starting test generation...")
print(f"Character: data/uploads/avatar.png")
print(f"Using manual script (~30 seconds speech)")
print(f"Using Wav2Lip pipeline (GPU: RTX 3050 4GB)")
print()

start = time.time()
result = run_full_pipeline(
    character_image_path=str(config.UPLOADS_DIR / "avatar.png"),
    topic="",
    manual_text=script_text,
    news_style="neutral",
    voice=None,
    speed=1.0,
    use_ai_script=False,
    language="english",
    gender="female",
    accent="indian",
    duration_minutes=1,
    use_elevenlabs=False,
    use_longcat=False,
    use_sarvam=True,
    use_dit=False,
    on_progress=on_progress,
    studio_production=False,
    motion_enhancement=False,
    use_generated_b_roll=False,
    cinematic_style="evening",
)
elapsed = time.time() - start
print(f"\nDone in {elapsed:.0f}s")
print(f"Video: {result['video_path']}")
print(f"Title: {result['title']}")
print(f"File size: {Path(result['video_path']).stat().st_size / 1e6:.1f} MB")
