import json

LONGCAT_BASE_URL = "https://api.longcat.chat/openai"
LONGCAT_MODEL = "LongCat-2.0-Preview"

SYSTEM_PROMPT_EN = """You are a professional English TV news anchor script writer.
Write engaging, clear news scripts suitable for spoken delivery with a natural news anchor accent.

CRITICAL RULES:
- Return ONLY valid JSON, no other text
- Structure: {"title": "...", "segments": [{"speaker_style": "...", "text": "..."}]}
- speaker_style: "neutral", "serious", "engaging", "breaking"
- Each segment text should be 1-3 sentences, easy to read aloud
- Use conversational news anchor tone, avoid complex sentences
- Include natural pauses and emphasis points
- Start with a proper news anchor greeting like "Good evening, and welcome to our news broadcast." or similar
- The script length is specified by the user; generate exactly that many segments to fill the time"""

SYSTEM_PROMPT_MR = """You are a professional Marathi TV news anchor script writer (व्यावसायिक मराठी बातमीदार).
Write engaging, clear news scripts in Marathi language suitable for spoken delivery with a natural news anchor accent.

CRITICAL RULES:
- Return ONLY valid JSON, no other text
- All text MUST be in Marathi (मराठी) language ONLY
- Structure: {"title": "...", "segments": [{"speaker_style": "...", "text": "..."}]}
- speaker_style: "neutral", "serious", "engaging", "breaking"
- Each segment text should be 1-3 sentences, easy to read aloud
- Use conversational news anchor tone in Marathi, avoid complex sentences
- Include natural pauses and emphasis points
- Start with a proper Marathi news anchor greeting like "नमस्कार, आपले स्वागत आहे आजच्या बातमीपत्रात." or similar
- IMPORTANT: Title must also be in Marathi
- The script length is specified by the user; generate exactly that many segments to fill the time"""

STYLE_PROMPTS = {
    "neutral": "Write in a calm, balanced reporting style.",
    "breaking": "Write as breaking news with urgency.",
    "entertainment": "Write in a light, engaging entertainment style.",
    "sports": "Write with sports commentary energy.",
}


def _safe_json_parse(text: str) -> dict:
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(text)


def _tokens_for_duration(duration_minutes: int) -> int:
    return {1: 1024, 2: 2048, 3: 3072, 4: 4096, 5: 5120}.get(duration_minutes, 1024)


def _segments_for_duration(duration_minutes: int) -> int:
    return duration_minutes * 8


def generate_script(api_key: str, topic: str, style: str = "neutral", language: str = "english", duration_minutes: int = 1) -> dict | None:
    if not api_key:
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return None

    client = OpenAI(api_key=api_key, base_url=LONGCAT_BASE_URL)
    num_segments = _segments_for_duration(duration_minutes)

    if language == "marathi":
        system_prompt = SYSTEM_PROMPT_MR
        lang_instruction = "Write the COMPLETE script in Marathi (मराठी) language only."
        duration_note = f"ही बातमी अंदाजे {duration_minutes} मिनिटांची आहे. कृपया वेगवेगळ्या विषयांवर {num_segments} पेक्षा कमी भाग लिहा."
    else:
        system_prompt = SYSTEM_PROMPT_EN
        lang_instruction = "Write the script in English."
        duration_note = f"This script should be approximately {duration_minutes} minute(s) long when read aloud. Write {num_segments} segments covering different angles of the story."

    style_guide = STYLE_PROMPTS.get(style, STYLE_PROMPTS["neutral"])

    try:
        completion = client.chat.completions.create(
            model=LONGCAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Topic: {topic}\n{style_guide}\n{lang_instruction}\n{duration_note}\n\nWrite a news script covering multiple angles with a proper anchor introduction and sign-off."},
            ],
            temperature=0.7,
            max_tokens=_tokens_for_duration(duration_minutes),
            response_format={"type": "json_object"},
        )
        raw = completion.choices[0].message.content
        return _safe_json_parse(raw)
    except Exception as e:
        print(f"LongCat API error: code={type(e).__name__}")
        return None
