import gc
import uuid
import torch
from pathlib import Path
import config


def resolve_device():
    if config.DEVICE == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def get_vram_gb() -> float:
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:
        pass
    return 0.0


def detect_best_backend() -> str:
    vram = get_vram_gb()
    if torch.cuda.is_available():
        if vram >= 20:
            return "cogvideox"
        elif vram >= 12:
            return "cogvideox"
    return "disabled"


def backend_available(name: str) -> bool:
    try:
        if name == "cogvideox":
            from diffusers import CogVideoXPipeline
            vram = get_vram_gb()
            return vram >= 12
        return False
    except Exception:
        return False


def get_capability_report() -> dict:
    vram = get_vram_gb()
    cuda = torch.cuda.is_available()
    gpu_name = ""
    if cuda:
        try:
            gpu_name = torch.cuda.get_device_properties(0).name
        except Exception:
            gpu_name = "Unknown GPU"
    backends = {}
    for b in ["cogvideox"]:
        backends[b] = backend_available(b)
    best = detect_best_backend() if any(backends.values()) else "disabled"
    return {
        "gpu_available": cuda,
        "gpu_name": gpu_name,
        "vram_gb": vram,
        "backends": backends,
        "best_backend": best,
        "dit_ready": best != "disabled" and config.DIT_ENABLE in ("auto", "true"),
    }


def download_model_if_needed(model_type: str = "cogvideox", on_progress=None) -> bool:
    if model_type == "cogvideox":
        target = config.DIT_COGVIDEOX_PATH
        repo_id = "THUDM/CogVideoX-5B-I2V"
    else:
        return False

    if any(target.iterdir()):
        if on_progress:
            on_progress(100, f"DiT: {model_type} model already exists")
        return True

    if on_progress:
        on_progress(0, f"DiT: downloading {repo_id} (~10GB)...")

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target),
            resume_download=True,
            ignore_patterns=["*.safetensors"] if get_vram_gb() < 16 else [],
        )
        ok = any(target.iterdir())
        if on_progress:
            on_progress(100, "DiT: download complete!" if ok else "DiT: download incomplete")
        return ok
    except Exception as e:
        if on_progress:
            on_progress(0, f"DiT: download error - {e}")
        return False


class CogVideoXBackend:
    def __init__(self):
        self.pipeline = None
        self.device = resolve_device()
        self.loaded = False

    def load(self, on_progress=None) -> bool:
        if self.loaded:
            return True
        try:
            from diffusers import CogVideoXPipeline, CogVideoXImageToVideoPipeline
            from diffusers.utils import export_to_video
            import torch

            if on_progress:
                on_progress(0, "DiT: loading CogVideoX model...")

            model_path = str(config.DIT_COGVIDEOX_PATH)
            if not Path(model_path).exists() or not any(Path(model_path).iterdir()):
                model_path = "THUDM/CogVideoX-5B-I2V"

            vram = get_vram_gb()
            dtype = torch.float16 if vram < 24 else torch.bfloat16

            if on_progress:
                on_progress(20, "DiT: loading pipeline...")

            self.pipeline = CogVideoXPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype,
                variant="fp16" if vram < 24 else None,
            )

            if config.DIT_USE_CPU_OFFLOAD or vram < 16:
                self.pipeline.enable_model_cpu_offload()
            else:
                self.pipeline.to(self.device)

            self.pipeline.vae.enable_tiling()

            if vram >= 16:
                self.pipeline.enable_sequential_cpu_offload()

            self.loaded = True
            if on_progress:
                on_progress(100, "DiT: CogVideoX ready!")
            return True

        except Exception as e:
            print(f"CogVideoX load error: {e}")
            if on_progress:
                on_progress(0, f"DiT: load failed - {e}")
            return False

    def generate_clip(
        self,
        prompt: str,
        num_frames: int = 49,
        output_path: str = None,
        image_path: str = None,
        guidance_scale: float = None,
        num_inference_steps: int = None,
        on_progress=None,
    ) -> str | None:
        if not self.loaded and not self.load(on_progress):
            return None

        if output_path is None:
            from pathlib import Path
            output_path = str(config.OUTPUT_DIR / f"dit_clip_{uuid.uuid4().hex[:8]}.mp4")

        guidance_scale = guidance_scale or config.DIT_GUIDANCE_SCALE
        num_inference_steps = num_inference_steps or config.DIT_NUM_INFERENCE_STEPS

        try:
            from diffusers.utils import export_to_video
            import torch

            if on_progress:
                on_progress(50, "DiT: generating video frames...")

            generator = torch.Generator(device=self.device).manual_seed(42)

            vram = get_vram_gb()
            if vram >= 24:
                width, height = 960, 640
                num_inference_steps = max(num_inference_steps, 100)
            elif vram >= 16:
                width, height = 720, 480
                num_inference_steps = max(num_inference_steps, 75)
            else:
                width, height = 480, 320

            extra_kwargs = {}
            if image_path and Path(image_path).exists():
                try:
                    from PIL import Image
                    cond_img = Image.open(image_path).convert("RGB").resize((width, height))
                    extra_kwargs["image"] = cond_img
                except Exception:
                    pass

            with torch.autocast(device_type="cuda" if "cuda" in self.device else "cpu", dtype=torch.float16):
                frames = self.pipeline(
                    prompt=prompt,
                    num_videos_per_prompt=1,
                    num_inference_steps=num_inference_steps,
                    num_frames=num_frames,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    width=width,
                    height=height,
                    **extra_kwargs,
                ).frames[0]

            if on_progress:
                on_progress(85, "DiT: exporting to video...")

            export_to_video(frames, output_path, fps=12)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            if on_progress:
                on_progress(100, "DiT: clip generated!")

            return str(output_path) if Path(output_path).exists() else None

        except Exception as e:
            print(f"CogVideoX generation error: {e}")
            if on_progress:
                on_progress(0, f"DiT: generation failed - {e}")
            return None

    def unload(self):
        self.pipeline = None
        self.loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class DiTVideoEngine:
    def __init__(self):
        self.backend = None
        self.backend_name = None
        self.capability = get_capability_report()

    def initialize(self, force_backend: str = None, on_progress=None) -> bool:
        backend_name = force_backend or config.DIT_BACKEND
        if backend_name == "disabled":
            return False
        if backend_name == "auto":
            backend_name = detect_best_backend()

        if backend_name == "cogvideox":
            self.backend = CogVideoXBackend()
            self.backend_name = "cogvideox"
            return self.backend.load(on_progress)

        return False

    def generate_clip(
        self,
        prompt: str,
        duration_sec: float = 4.0,
        output_path: str = None,
        image_path: str = None,
        on_progress=None,
    ) -> str | None:
        if self.backend is None:
            return None
        num_frames = max(13, int(duration_sec * 8))
        return self.backend.generate_clip(
            prompt=prompt,
            num_frames=num_frames,
            output_path=output_path,
            image_path=image_path,
            on_progress=on_progress,
        )

    def is_ready(self) -> bool:
        return self.backend is not None and self.backend.loaded

    def capability_report(self) -> dict:
        return self.capability

    def shutdown(self):
        if self.backend:
            self.backend.unload()
