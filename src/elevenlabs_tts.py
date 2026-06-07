import os
from pathlib import Path
import config


def elevenlabs_available() -> bool:
    key = os.environ.get("ELEVENLABS_API_KEY") or config.ELEVENLABS_API_KEY
    voice = os.environ.get("ELEVENLABS_VOICE_ID") or config.ELEVENLABS_VOICE_ID
    return bool(key and voice and key != "your_api_key_here")


def generate_speech_elevenlabs(text: str, output_path: Path, language: str = "english") -> Path | None:
    try:
        from elevenlabs import ElevenLabs

        api_key = os.environ.get("ELEVENLABS_API_KEY") or config.ELEVENLABS_API_KEY
        voice_id = os.environ.get("ELEVENLABS_VOICE_ID") or config.ELEVENLABS_VOICE_ID

        if not api_key or not voice_id or api_key == "your_api_key_here":
            return None

        client = ElevenLabs(api_key=api_key)
        audio = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
        with open(str(output_path), "wb") as f:
            for chunk in audio:
                f.write(chunk)
        return output_path if output_path.exists() else None
    except Exception as e:
        print(f"ElevenLabs TTS error: {e}")
        return None
