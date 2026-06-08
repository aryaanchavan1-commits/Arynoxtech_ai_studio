import os
import base64
from pathlib import Path
import config


SARVAM_VOICES = {
    "male": "amit",
    "female": "priya",
}

SARVAM_VOICE_OPTIONS = [
    "amit", "priya", "sumit", "ishita",
    "neel", "shubh", "anushka", "arjun", "kavya",
]

LANG_TO_SARVAM = {
    "english":  "en-IN",
    "hindi":    "hi-IN",
    "marathi":  "mr-IN",
    "tamil":    "ta-IN",
    "telugu":   "te-IN",
    "kannada":  "kn-IN",
    "malayalam":"ml-IN",
    "bengali":  "bn-IN",
    "gujarati": "gu-IN",
    "punjabi":  "pa-IN",
    "odia":     "or-IN",
}


def sarvam_available() -> bool:
    if not config.SARVAM_API_KEY:
        return False
    try:
        import sarvamai
        return True
    except ImportError:
        return False


def generate_speech_sarvam(
    text: str,
    output_path: Path,
    language: str = "english",
    gender: str = "female",
    sample_rate: int = 48000,
    voice: str = None,
) -> Path | None:
    try:
        from sarvamai import SarvamAI

        api_key = config.SARVAM_API_KEY
        if not api_key:
            return None

        lang_code = LANG_TO_SARVAM.get(language, "en-IN")
        speaker = voice or SARVAM_VOICES.get(gender, "priya")

        print(f"Sarvam TTS: lang={lang_code}, speaker={speaker}, text_len={len(text)}")

        client = SarvamAI(api_subscription_key=api_key)
        resp = client.text_to_speech.convert(
            text=text,
            target_language_code=lang_code,
            model="bulbul:v3",
            speaker=speaker,
            speech_sample_rate=sample_rate,
        )

        if not hasattr(resp, "audios") or not resp.audios:
            print("Sarvam: no audio in response")
            return None

        audio_data = base64.b64decode(resp.audios[0])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(output_path), "wb") as f:
            f.write(audio_data)

        if output_path.exists():
            print(f"Sarvam TTS: saved {len(audio_data)} bytes to {output_path.name}")
            return output_path
        return None

    except Exception as e:
        print(f"Sarvam TTS error: {e}")
        return None
