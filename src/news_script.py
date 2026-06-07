import json
import config

try:
    import src.longcat_chat as longcat_chat
    _LONGCAT_AVAILABLE = True
except Exception:
    longcat_chat = None
    _LONGCAT_AVAILABLE = False


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


def _safe_json_parse(text: str) -> dict:
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(text)


def _tokens_for_duration(duration_minutes: int) -> int:
    return {1: 1024, 2: 2048, 3: 3072, 4: 4096, 5: 5120}.get(duration_minutes, 1024)


def _segments_for_duration(duration_minutes: int) -> int:
    return duration_minutes * 8


def _generate_via_groq(topic: str, style: str, language: str, duration_minutes: int, num_segments: int) -> dict | None:
    if not config.GROQ_API_KEY or config.GROQ_API_KEY == "gsk_your_groq_api_key_here":
        return None

    try:
        from groq import Groq
    except ImportError:
        return None

    client = Groq(api_key=config.GROQ_API_KEY)

    if language == "marathi":
        system_prompt = SYSTEM_PROMPT_MR
        lang_instruction = "Write the COMPLETE script in Marathi (मराठी) language only."
        duration_note = f"ही बातमी अंदाजे {duration_minutes} मिनिटांची आहे. कृपया वेगवेगळ्या विषयांवर {num_segments} पेक्षा कमी भाग लिहा."
    else:
        system_prompt = SYSTEM_PROMPT_EN
        lang_instruction = "Write the script in English."
        duration_note = f"This script should be approximately {duration_minutes} minute(s) long when read aloud. Write {num_segments} segments covering different angles of the story."

    STYLE_PROMPTS = {
        "neutral": "Write in a calm, balanced reporting style.",
        "breaking": "Write as breaking news with urgency.",
        "entertainment": "Write in a light, engaging entertainment style.",
        "sports": "Write with sports commentary energy.",
    }
    style_guide = STYLE_PROMPTS.get(style, STYLE_PROMPTS["neutral"])

    try:
        completion = client.chat.completions.create(
            model=config.GROQ_MODEL,
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
        print(f"Groq API error: {e}")
        return None


def generate_news_script(topic: str, style: str = "neutral", language: str = "english", duration_minutes: int = 1) -> dict:
    result = _generate_via_groq(topic, style, language, duration_minutes, _segments_for_duration(duration_minutes))
    if result is not None:
        return result

    if _LONGCAT_AVAILABLE and longcat_chat:
        try:
            result = longcat_chat.generate_script(
                api_key=config.LONGCAT_CHAT_API_KEY,
                topic=topic,
                style=style,
                language=language,
                duration_minutes=duration_minutes,
            )
            if result is not None:
                return result
        except Exception:
            pass

    return _fallback_script(topic, language, duration_minutes)


def _repeat_segments(base_segments: list[dict], target: int) -> list[dict]:
    result = []
    while len(result) < target:
        for seg in base_segments:
            if len(result) >= target:
                break
            seg_copy = dict(seg)
            result.append(seg_copy)
    return result


def _fallback_script(topic: str, language: str = "english", duration_minutes: int = 1) -> dict:
    num_segments = _segments_for_duration(duration_minutes)

    if language == "marathi":
        base = [
            {"speaker_style": "neutral", "text": f"नमस्कार, आपले स्वागत आहे आजच्या बातमीपत्रात. आज आपण एका महत्त्वाच्या बातमीवर चर्चा करणार आहोत - {topic}."},
            {"speaker_style": "engaging", "text": f"सूत्रांनी दिलेल्या माहितीनुसार, {topic} विषयी अधिक माहिती समोर येत आहे. अनेक स्रोतांकडून ही माहिती पुष्टी करण्यात आली आहे."},
            {"speaker_style": "breaking", "text": f"या प्रकरणी तज्ज्ञांचे म्हणणे आहे की {topic} चा परिणाम सर्वसामान्य नागरिकांवर दिसून येईल. यासंदर्भात अधिक तपास सुरू आहे."},
            {"speaker_style": "serious", "text": f"आमच्या प्रतिनिधीने दिलेल्या माहितीनुसार, {topic} बाबत सरकारी स्तरावर चर्चा सुरू आहे. लवकरच याबाबत निर्णय घेतला जाईल."},
            {"speaker_style": "neutral", "text": "आम्ही या बातमीवर लक्ष ठेवून आहोत आणि नवीन माहिती मिळताच तुम्हाला अपडेट देऊ. बातमीपत्रात सहभागी झाल्याबद्दल धन्यवाद."},
        ]
        segments = _repeat_segments(base, num_segments)
        return {"title": f"बातमी अपडेट: {topic}", "segments": segments}

    if language == "hindi":
        base = [
            {"speaker_style": "neutral", "text": f"नमस्कार, आज की इस खास खबर में हम बात करेंगे {topic} के बारे में।"},
            {"speaker_style": "engaging", "text": f"हमारे सूत्रों के अनुसार, {topic} से जुड़ी नई जानकारी सामने आ रही है। कई स्रोतों से इसकी पुष्टि हुई है।"},
            {"speaker_style": "breaking", "text": f"विशेषज्ञों का कहना है कि {topic} का असर आम नागरिकों पर भी देखने को मिलेगा। इस मामले में आगे की जांच जारी है।"},
            {"speaker_style": "serious", "text": f"हमारे संवाददाता के अनुसार, {topic} को लेकर सरकारी स्तर पर चर्चा जारी है। जल्द ही इस पर फैसला लिया जा सकता है।"},
            {"speaker_style": "neutral", "text": "हम इस खबर पर नज़र बनाए हुए हैं और नई जानकारी मिलते ही आपको अपडेट करेंगे। हमारे साथ जुड़ने के लिए धन्यवाद।"},
        ]
        segments = _repeat_segments(base, num_segments)
        return {"title": f"खबर अपडेट: {topic}", "segments": segments}

    base = [
        {"speaker_style": "neutral", "text": f"Good evening, and welcome to our news broadcast. We're covering an important story about {topic}."},
        {"speaker_style": "engaging", "text": f"Reports indicate that {topic} continues to develop, with new information emerging from multiple sources."},
        {"speaker_style": "breaking", "text": f"Experts suggest that {topic} could have significant implications in the coming days. Authorities are closely monitoring the situation."},
        {"speaker_style": "serious", "text": f"Our correspondents on the ground report that the situation regarding {topic} remains dynamic. We are expecting further updates shortly."},
        {"speaker_style": "neutral", "text": f"We'll continue to monitor this story and bring you updates as they become available. Thank you for watching."},
    ]
    segments = _repeat_segments(base, num_segments)
    return {"title": f"News Update: {topic}", "segments": segments}


def generate_manual_script(title: str, text: str) -> dict:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    segments = []
    for i, p in enumerate(paragraphs):
        style = "neutral" if i == 0 else "engaging" if i < len(paragraphs) - 1 else "serious"
        segments.append({"speaker_style": style, "text": p})
    return {"title": title, "segments": segments}


def fetch_live_news(category: str = "general") -> list[dict]:
    if not config.NEWSAPI_KEY:
        return []
    import requests
    try:
        resp = requests.get(
            "https://newsapi.org/v2/top-headlines",
            params={"country": "us", "category": category, "pageSize": 5, "apiKey": config.NEWSAPI_KEY},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return [{"title": a["title"], "description": a.get("description", "")} for a in data.get("articles", [])]
    except Exception as e:
        print(f"NewsAPI error: {e}")
        return []
