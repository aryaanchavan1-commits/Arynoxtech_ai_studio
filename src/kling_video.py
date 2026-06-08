import base64
import json
import time
import uuid
from pathlib import Path

import requests

import config


def kling_available() -> bool:
    return bool(config.KLING_ACCESS_KEY and config.KLING_SECRET_KEY)


def kling_api_configured() -> bool:
    return kling_available()


class KlingVideoEngine:
    BASE_URL = "https://api.klingai.com"

    def __init__(self):
        self.access_key = config.KLING_ACCESS_KEY
        self.secret_key = config.KLING_SECRET_KEY
        self._token = None
        self._token_expiry = 0

    def _generate_token(self) -> str:
        try:
            import jwt as pyjwt
        except ImportError:
            raise ImportError("PyJWT is required. Install: pip install PyJWT")

        payload = {
            "iss": self.access_key,
            "exp": int(time.time()) + 1800,
            "nbf": int(time.time()) - 5,
        }
        return pyjwt.encode(payload, self.secret_key, algorithm="HS256")

    def _get_headers(self) -> dict:
        now = time.time()
        if now > self._token_expiry - 60:
            self._token = self._generate_token()
            self._token_expiry = now + 1800
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

    def text_to_video(
        self,
        prompt: str,
        duration: int = 5,
        aspect_ratio: str = "16:9",
        model_name: str = None,
        mode: str = "std",
        negative_prompt: str = None,
    ) -> dict:
        model_name = model_name or config.KLING_MODEL
        payload = {
            "model_name": model_name,
            "prompt": prompt,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
            "mode": mode,
        }
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        if config.KLING_EXTEND_IMAGE:
            payload["extend_image"] = config.KLING_EXTEND_IMAGE

        resp = requests.post(
            f"{self.BASE_URL}/v1/videos/text2video",
            headers=self._get_headers(),
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"Kling API error: {data.get('message', str(data))}")
        return data["data"]

    def image_to_video(
        self,
        image_path: str,
        prompt: str,
        duration: int = 5,
        model_name: str = None,
        mode: str = "std",
    ) -> dict:
        model_name = model_name or config.KLING_MODEL
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "model_name": model_name,
            "image": img_b64,
            "prompt": prompt,
            "duration": duration,
            "mode": mode,
        }

        resp = requests.post(
            f"{self.BASE_URL}/v1/videos/image2video",
            headers=self._get_headers(),
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"Kling image2video error: {data.get('message', str(data))}")
        return data["data"]

    def poll_task(self, task_id: str, task_type: str = "text2video") -> dict:
        url = f"{self.BASE_URL}/v1/videos/{task_type}/{task_id}"
        while True:
            resp = requests.get(url, headers=self._get_headers(), timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 0:
                raise RuntimeError(f"Kling poll error: {data.get('message', str(data))}")
            task_data = data["data"]
            status = task_data.get("task_status")
            if status == "succeed":
                return task_data
            if status == "failed":
                raise RuntimeError(f"Kling task failed: {task_data.get('task_status_msg', 'unknown')}")
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
                output_path = str(config.OUTPUT_DIR / f"kling_{uuid.uuid4().hex[:8]}.mp4")

            if image_path and Path(image_path).exists():
                if on_progress:
                    on_progress(0, "Kling: image-to-video...")
                task_data = self.image_to_video(
                    image_path=image_path,
                    prompt=prompt,
                    duration=min(duration_sec, 10),
                )
                task_type = "image2video"
            else:
                if on_progress:
                    on_progress(0, "Kling: text-to-video...")
                task_data = self.text_to_video(
                    prompt=prompt,
                    duration=min(duration_sec, 10),
                )
                task_type = "text2video"

            task_id = task_data["task_id"]
            if on_progress:
                on_progress(30, "Kling: waiting for generation...")

            result = self.poll_task(task_id, task_type)
            videos = result.get("task_result", {}).get("videos", [])
            if not videos:
                raise RuntimeError("No videos in Kling response")

            video_url = videos[0].get("url")
            if not video_url:
                raise RuntimeError("No video URL in Kling response")

            if on_progress:
                on_progress(80, "Kling: downloading video...")

            return self.download_video(video_url, output_path)

        except Exception:
            if on_progress:
                on_progress(0, "Kling clip generation failed")
            return None


_engine = None


def _get_engine() -> KlingVideoEngine:
    global _engine
    if _engine is None:
        _engine = KlingVideoEngine()
    return _engine


_kling_spend_counter = 0
KLING_MAX_SPEND = 15

def generate_scene_video(
    prompt: str,
    duration_sec: int = 5,
    output_path: str = None,
    image_path: str = None,
    on_progress=None,
    max_retries: int = 3,
) -> str | None:
    global _kling_spend_counter
    if _kling_spend_counter >= KLING_MAX_SPEND:
        if on_progress:
            on_progress(0, "Kling spend cap reached, skipping remaining scenes")
        return None

    engine = _get_engine()
    for attempt in range(1, max_retries + 1):
        if _kling_spend_counter >= KLING_MAX_SPEND:
            return None
        if on_progress and attempt > 1:
            on_progress(0, f"Kling retry {attempt}/{max_retries}...")
        result = engine.generate_clip(
            prompt=prompt,
            duration_sec=duration_sec,
            output_path=output_path,
            image_path=image_path,
            on_progress=on_progress,
        )
        if result:
            _kling_spend_counter += 1
            return result
        if on_progress:
            on_progress(0, f"Kling attempt {attempt} failed, retrying...")
        time.sleep(3)
    if on_progress:
        on_progress(0, f"Kling gave up on: {prompt[:50]}...")
    return None

def reset_kling_spend():
    global _kling_spend_counter
    _kling_spend_counter = 0


def run_kling_scene_generation(
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
            on_progress(pct, f"Kling scene {i+1}/{total}: {scene.scene_type}...")

        clip_path = config.OUTPUT_DIR / f"kling_scene_{uuid.uuid4().hex[:8]}.mp4"

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
                on_progress(0, f"Kling scene {i} failed, skipping")

    return clip_results
