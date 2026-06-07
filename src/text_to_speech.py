import asyncio
from pathlib import Path
import edge_tts
import config


def resolve_voice(voice: str | None, language: str = "english", gender: str = "female", accent: str = "indian") -> str:
    if voice and voice != "auto":
        return voice
    lang_voices = config.LANGUAGE_VOICES.get(language, config.LANGUAGE_VOICES["english"])
    gender_map = lang_voices.get(gender, lang_voices.get("female", {}))
    return gender_map.get(accent, gender_map.get("default", gender_map.get("indian", list(gender_map.values())[0] if gender_map else "en-US-AriaNeural")))


async def _generate_audio(text: str, voice: str, output_path: Path, speed: float = 1.0):
    communicate = edge_tts.Communicate(text, voice, rate=f"{int((speed - 1.0) * 100):+d}%")
    await communicate.save(str(output_path))


def _fallback_gtts(text: str, output_path: Path, language: str = "english") -> bool:
    try:
        from gtts import gTTS
        lang_map = {
            "english":"en","hindi":"hi","marathi":"mr","tamil":"ta","telugu":"te",
            "kannada":"kn","malayalam":"ml","bengali":"bn","gujarati":"gu",
            "punjabi":"pa","odia":"or",
        }
        lang_code = lang_map.get(language, "en")
        tts = gTTS(text, lang=lang_code, slow=False)
        tts.save(str(output_path))
        return output_path.exists()
    except Exception as e:
        print(f"gTTS fallback error: {e}")
        return False


def generate_speech(script: dict, voice: str = None, speed: float = 1.0, language: str = "english", gender: str = "female", accent: str = "indian") -> Path | None:
    voice = resolve_voice(voice, language, gender, accent)
    full_text = "\n\n".join(seg["text"] for seg in script.get("segments", []))
    if not full_text.strip():
        return None

    output_path = config.OUTPUT_DIR / "temp_audio.wav"
    try:
        asyncio.run(_generate_audio(full_text, voice, output_path, speed))
        if output_path.exists():
            return output_path
        print("edge-tts produced no output, trying gTTS fallback")
    except Exception as e:
        print(f"edge-tts error: {e}, trying gTTS fallback")

    if _fallback_gtts(full_text, output_path.with_suffix(".mp3"), language):
        return output_path.with_suffix(".mp3")
    return None


def generate_speech_from_text(text: str, voice: str = None, speed: float = 1.0, language: str = "english", gender: str = "female", accent: str = "indian") -> Path | None:
    voice = resolve_voice(voice, language, gender, accent)
    if not text.strip():
        return None
    output_path = config.OUTPUT_DIR / "temp_audio.wav"
    try:
        asyncio.run(_generate_audio(text, voice, output_path, speed))
        if output_path.exists():
            return output_path
        print("edge-tts produced no output, trying gTTS fallback")
    except Exception as e:
        print(f"edge-tts error: {e}, trying gTTS fallback")

    if _fallback_gtts(text, output_path.with_suffix(".mp3"), language):
        return output_path.with_suffix(".mp3")
    return None
