"""
NLP Amplifier — Natural Language Prompt Enhancer

Takes a brief user prompt and expands it into a rich, production-quality
prompt optimized for AI image/video generation. Uses Groq (Llama 3.3 70B)
to add cinematic detail, lighting, mood, and visual style — resulting in
better first-generation outputs and fewer retries (lower API spend).

Usage:
    from src.nlp_amplifier import amplify_prompt
    enhanced = amplify_prompt("a man speaking about technology")
"""
import config

SYSTEM_PROMPT = """You are a professional cinematic prompt engineer. Your job is to take a brief user description and expand it into a rich, detailed production-quality prompt optimized for AI image/video generation.

RULES:
- Return ONLY the enhanced prompt text, no explanations, no markdown
- Add: cinematic lighting, camera angle, mood, atmosphere, color palette, visual style
- Keep it concise (40-80 words) — optimized for AI generation APIs
- Focus on visual elements that improve generation quality
- Do NOT add people/characters unless the original prompt mentions them
- Use professional cinematography terminology
- Output must be a single paragraph"""


def amplify_prompt(brief_prompt: str, style: str = "cinematic") -> str:
    if not brief_prompt or len(brief_prompt.strip()) < 3:
        return brief_prompt
    if not config.GROQ_API_KEY or config.GROQ_API_KEY == "gsk_your_groq_api_key_here":
        return brief_prompt
    try:
        from groq import Groq
        client = Groq(api_key=config.GROQ_API_KEY)
        style_guide = {
            "cinematic": "cinematic lighting, shallow depth of field, professional color grading",
            "broadcast": "broadcast studio quality, professional video production, television-grade lighting",
            "natural": "natural soft lighting, realistic textures, true-to-life colors",
            "dramatic": "dramatic chiaroscuro lighting, high contrast, moody atmosphere",
            "vibrant": "vibrant colors, high saturation, dynamic visual energy",
        }.get(style, "cinematic lighting, professional grade output")
        completion = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Enhance this prompt for AI video generation. Style: {style_guide}. Prompt: {brief_prompt}"},
            ],
            temperature=0.4,
            max_tokens=200,
        )
        enhanced = completion.choices[0].message.content.strip().strip('"\'')
        if enhanced:
            return enhanced
    except Exception:
        pass
    return brief_prompt


def batch_amplify(prompts: list[str], style: str = "cinematic") -> list[str]:
    return [amplify_prompt(p, style) for p in prompts]
