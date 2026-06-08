"""
Runway Gen-4 Turbo API wrapper for Arynox AI Studio.

Replaces Kling AI backend with Runway's superior video generation at same/lower cost.
Uses gen4_turbo ($0.05/s) as default — cheaper than Kling v3 with better quality.

API Docs: https://docs.runwayml.com/
"""
import time
import uuid
from pathlib import Path

import requests

import config


def runway_available() -> bool:
    return bool(config.RUNWAY_API_KEY)


def runway_api_configured() -> bool:
    return runway_available()


class RunwayVideoEngine:
    BASE_URL = "https://api.runwayml.com/v1"

    def __init__(self):
        self.api_key = config.RUNWAY_API_KEY
        self.model = config.RUNWAY_MODEL
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def text_to_video(
        self,
        prompt: str,
        duration: int = 5,
        aspect_ratio: str = "16:9",
        model_name: str = None,
    ) -> dict:
        model = model_name or self.model
        payload = {
            "model": model,
            "prompt": prompt,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
            "mode": "text-to-video",
        }
        resp = requests.post(
            f"{self.BASE_URL}/generations",
            headers=self._headers,
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            raise RuntimeError(f"Runway API error: {data['error']}")
        return data

    def image_to_video(
        self,
        image_path: str,
        prompt: str,
        duration: int = 5,
        model_name: str = None,
    ) -> dict:
        model = model_name or self.model
        import base64
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        payload = {
            "model": model,
            "prompt": prompt,
            "image": img_b64,
            "duration": duration,
            "mode": "image-to-video",
        }
        resp = requests.post(
            f"{self.BASE_URL}/generations",
            headers=self._headers,
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            raise RuntimeError(f"Runway image2video error: {data['error']}")
        return data

    def poll_task(self, generation_id: str) -> dict:
        url = f"{self.BASE_URL}/generations/{generation_id}"
        while True:
            resp = requests.get(url, headers=self._headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if data.get("error"):
                raise RuntimeError(f"Runway poll error: {data['error']}")
            status = data.get("status", "pending")
            if status == "completed":
                return data
            if status == "failed":
                raise RuntimeError(f"Runway task failed: {data.get('failure_reason', 'unknown')}")
            time.sleep(5)

    def download_video(self, url: str, output_path: str) -> str:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(url, timeout=300, stream=True)
        resp.raise_for_status()
        with open(str(out), "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        return str(out)

    def generate_clip(
        self,
        prompt: str,
        duration_sec: int = 5,
        output_path: str = None,
        image_path: str = None,
        on_progress=None,
    ) -> str | None:
        try:
            if output_path is None:
                output_path = str(config.OUTPUT_DIR / f"runway_{uuid.uuid4().hex[:8]}.mp4")

            if image_path and Path(image_path).exists():
                if on_progress:
                    on_progress(0, "Runway: image-to-video...")
                task_data = self.image_to_video(
                    image_path=image_path,
                    prompt=prompt,
                    duration=min(duration_sec, 10),
                )
            else:
                if on_progress:
                    on_progress(0, "Runway: text-to-video...")
                task_data = self.text_to_video(
                    prompt=prompt,
                    duration=min(duration_sec, 10),
                )

            generation_id = task_data.get("id")
            if not generation_id:
                raise RuntimeError("No generation ID in Runway response")

            if on_progress:
                on_progress(30, "Runway: generating...")

            result = self.poll_task(generation_id)
            video_url = result.get("output", [{}])[0].get("url") if result.get("output") else None
            if not video_url:
                video_url = result.get("video_url")
            if not video_url:
                raise RuntimeError("No video URL in Runway response")

            if on_progress:
                on_progress(80, "Runway: downloading...")

            return self.download_video(video_url, output_path)

        except Exception:
            if on_progress:
                on_progress(0, "Runway clip generation failed")
            return None


_engine = None


def _get_engine() -> RunwayVideoEngine:
    global _engine
    if _engine is None:
        _engine = RunwayVideoEngine()
    return _engine


_runway_spend_counter = 0
RUNWAY_MAX_SPEND = 15


def generate_scene_video(
    prompt: str,
    duration_sec: int = 5,
    output_path: str = None,
    image_path: str = None,
    on_progress=None,
    max_retries: int = 3,
) -> str | None:
    global _runway_spend_counter
    if _runway_spend_counter >= RUNWAY_MAX_SPEND:
        if on_progress:
            on_progress(0, "Runway spend cap reached, skipping remaining scenes")
        return None

    engine = _get_engine()
    for attempt in range(1, max_retries + 1):
        if _runway_spend_counter >= RUNWAY_MAX_SPEND:
            return None
        if on_progress and attempt > 1:
            on_progress(0, f"Runway retry {attempt}/{max_retries}...")
        result = engine.generate_clip(
            prompt=prompt,
            duration_sec=duration_sec,
            output_path=output_path,
            image_path=image_path,
            on_progress=on_progress,
        )
        if result:
            _runway_spend_counter += 1
            return result
        if on_progress:
            on_progress(0, f"Runway attempt {attempt} failed, retrying...")
        time.sleep(3)
    if on_progress:
        on_progress(0, f"Runway gave up on: {prompt[:50]}...")
    return None


def reset_runway_spend():
    global _runway_spend_counter
    _runway_spend_counter = 0


def run_runway_scene_generation(
    script: dict,
    character_image_path: str = None,
    duration_minutes: int = 1,
    use_generated_b_roll: bool = True,
    cinematic_style: str = "evening",
    visual_prompt: str = "",
    on_progress=None,
) -> list[dict]:
    from src.scene_director import plan_scenes

    scenes = plan_scenes(
        script=script,
        character_image_path=character_image_path,
        duration_minutes=duration_minutes,
        use_generated_b_roll=use_generated_b_roll,
        cinematic_style=cinematic_style,
        user_visual_prompt=visual_prompt,
    )

    clip_results = []
    total = len(scenes)

    for i, scene in enumerate(scenes):
        if on_progress:
            pct = 10 + int(80 * (i / total))
            on_progress(pct, f"Runway scene {i+1}/{total}: {scene.scene_type}...")

        clip_path = config.OUTPUT_DIR / f"runway_scene_{uuid.uuid4().hex[:8]}.mp4"

        sc_type = scene.scene_type
        img_cond = scene.image_condition if sc_type == "anchor" else None

        result = generate_scene_video(
            prompt=scene.prompt,
            duration_sec=int(scene.duration_sec),
            output_path=str(clip_path),
            image_path=img_cond,
            on_progress=on_progress,
        )
        if result:
            clip_results.append({
                "path": result,
                "scene_type": scene.scene_type,
                "duration_sec": scene.duration_sec,
                "text": scene.text,
            })
        else:
            if on_progress:
                on_progress(0, f"Runway scene {i} failed, skipping")

    return clip_results
