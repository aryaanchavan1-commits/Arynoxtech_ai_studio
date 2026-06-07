from typing import List, Dict, Optional


ANCHOR_VISUAL_PROMPTS = {
    "neutral": (
        "professional presenter, calm authoritative presence, well-lit studio, "
        "formal business attire, looking directly at camera, symmetrical composition, "
        "4K cinematic, soft key lighting, shallow depth of field, professional broadcast ambiance"
    ),
    "professional": (
        "professional presenter in high-end studio, confident poised demeanor, "
        "clean modern aesthetic, broadcast quality lighting, 4K cinematic, "
        "professional atmosphere, sharp focus, elegant composition"
    ),
    "casual": (
        "relaxed friendly presenter, warm natural smile, casual comfortable setting, "
        "conversational tone, soft natural lighting, 4K cinematic, "
        "inviting atmosphere, genuine expression, lifestyle style composition"
    ),
    "dramatic": (
        "intense dramatic presenter, serious focused expression, dramatic studio lighting, "
        "dynamic shadows, powerful presence, 4K cinematic, "
        "dramatic atmosphere, high contrast lighting, compelling composition"
    ),
}

TOPIC_B_ROLL_PROMPTS = {
    "technology": (
        "futuristic technology visualization, advanced AI neural network animation, "
        "glowing digital circuits and data streams, blue and purple neon lighting, "
        "holographic displays, cinematic 4K, motion design, tech documentary style, "
        "smooth camera movement, professional visual effects"
    ),
    "business": (
        "professional corporate environment, modern skyscraper cityscape, "
        "stock market data visualization, glass and steel architecture, "
        "clean professional aesthetic, cinematic 4K, corporate documentary style, "
        "golden hour lighting, smooth gimbal shot"
    ),
    "health": (
        "advanced medical research laboratory, DNA double helix visualization, "
        "clean sterile environment, scientific equipment in motion, "
        "blue and white color palette, cinematic 4K, medical documentary style, "
        "precise lighting, professional scientific atmosphere"
    ),
    "science": (
        "cosmic space exploration visualization, distant galaxies and nebulae, "
        "scientific discovery atmosphere, laboratory research setting, "
        "mysterious deep space background, cinematic 4K, BBC documentary style, "
        "dramatic lighting, awe-inspiring composition"
    ),
    "sports": (
        "professional sports arena, dramatic stadium lighting, "
        "championship trophy visualization, dynamic action freeze frames, "
        "crowd atmosphere, cinematic 4K, sports documentary style, "
        "golden hour exterior shots, professional broadcast quality"
    ),
    "breaking": (
        "emergency news coverage atmosphere, red alert visualization, "
        "newsroom chaos organized, urgent breaking news graphics, "
        "dramatic lighting with red accents, cinematic 4K, "
        "intense documentary style, urgent professional broadcast"
    ),
    "politics": (
        "government building exterior, parliament or capitol architecture, "
        "formal debate chamber, national flags, official ceremony setting, "
        "cinematic 4K, political documentary style, authoritative composition, "
        "grand architecture, professional news coverage"
    ),
    "weather": (
        "dramatic weather visualization, atmospheric cloud formations, "
        "dynamic climate maps with data overlays, storm systems, "
        "meteorological graphics, cinematic 4K, weather documentary style, "
        "dramatic sky lighting, professional forecast studio"
    ),
    "default": (
        "professional broadcast news studio, state-of-the-art production facility, "
        "multiple camera setup, broadcast quality lighting, "
        "clean modern design, cinematic 4K, professional news production, "
        "smooth camera operation, high-end television production quality"
    ),
}

TRANSITION_PROMPTS = {
    "crossfade": "professional broadcast crossfade transition, smooth seamless dissolve",
    "wipe": "broadcast quality wipe transition, clean edge movement, professional video production",
    "slide": "professional slide transition, smooth camera movement, news broadcast standard",
}

SCENE_CINEMATIC_STYLES = {
    "morning": "warm golden morning light, soft shadows, fresh bright atmosphere",
    "evening": "warm evening studio lighting, slightly dramatic shadows, cozy atmosphere",
    "night": "dark studio with blue ambient light, dramatic spotlight on anchor, professional night broadcast",
    "breaking": "high contrast dramatic lighting, red and blue emergency colors, urgent intensity",
}


class Scene:
    def __init__(self, scene_type: str, prompt: str, duration_sec: float,
                 text: str = "", speaker_style: str = "neutral",
                 image_condition: str = None, scene_id: int = 0,
                 cinematic_style: str = "evening"):
        self.scene_type = scene_type
        self.prompt = prompt
        self.duration_sec = duration_sec
        self.text = text
        self.speaker_style = speaker_style
        self.image_condition = image_condition
        self.scene_id = scene_id
        self.cinematic_style = cinematic_style

    def to_dict(self) -> dict:
        return {
            "type": self.scene_type,
            "prompt": self.prompt,
            "duration_sec": self.duration_sec,
            "text": self.text,
            "speaker_style": self.speaker_style,
            "image_condition": self.image_condition,
            "scene_id": self.scene_id,
            "cinematic_style": self.cinematic_style,
        }


def _classify_topic(topic: str) -> str:
    topic_lower = topic.lower()
    keywords = {
        "technology": ["tech", "ai", "software", "computer", "digital", "robot", "cyber", "internet", "code", "data", "algorithm", "startup", "innovation"],
        "business": ["business", "market", "stock", "economy", "finance", "corporate", "trade", "bank", "investment", "revenue", "ceo", "company"],
        "health": ["health", "medical", "hospital", "disease", "doctor", "patient", "covid", "vaccine", "medicine", "surgery", "wellness"],
        "science": ["science", "research", "study", "discovery", "space", "nasa", "climate", "biology", "physics", "chemistry", "evolution"],
        "sports": ["sports", "game", "match", "tournament", "champion", "player", "cricket", "football", "olympic", "soccer", "tennis", "stadium"],
        "politics": ["politics", "government", "minister", "election", "policy", "parliament", "president", "senate", "congress", "democracy"],
        "weather": ["weather", "storm", "rain", "temperature", "climate", "flood", "hurricane", "tornado", "forecast", "heatwave"],
    }
    for category, words in keywords.items():
        if any(w in topic_lower for w in words):
            return category
    return "default"


def _get_cinematic_style(style: str) -> str:
    return SCENE_CINEMATIC_STYLES.get(style, SCENE_CINEMATIC_STYLES["evening"])


def plan_scenes(
    script: dict,
    character_image_path: str = None,
    duration_minutes: int = 1,
    use_generated_b_roll: bool = True,
    cinematic_style: str = "evening",
    user_visual_prompt: str = "",
) -> List[Scene]:
    title = script.get("title", "Video")
    segments = script.get("segments", [])
    if not segments:
        anchor_p = user_visual_prompt or ANCHOR_VISUAL_PROMPTS["neutral"]
        return [Scene("anchor", anchor_p,
                      duration_minutes * 60, scene_id=0,
                      cinematic_style=cinematic_style,
                      image_condition=character_image_path)]

    topic = title.replace("News Update: ", "").replace("बातमी अपडेट: ", "").strip()
    topic_category = _classify_topic(topic)
    visual_style = TOPIC_B_ROLL_PROMPTS.get(topic_category, TOPIC_B_ROLL_PROMPTS["default"])
    cinematic = _get_cinematic_style(cinematic_style)

    total_seconds = duration_minutes * 60
    scenes = []
    scene_id = 0
    num_segments = len(segments)

    for i, seg in enumerate(segments):
        seg_text = seg["text"]
        seg_style = seg.get("speaker_style", "neutral")

        if user_visual_prompt:
            bkg_prompt = user_visual_prompt
        else:
            bkg_prompt = visual_style

        if i == 0:
            anchor_prompt = (
                f"{ANCHOR_VISUAL_PROMPTS.get(seg_style, ANCHOR_VISUAL_PROMPTS['neutral'])}, "
                f"opening introduction about {topic}, {cinematic}, "
                f"{bkg_prompt[:100]}"
            )
            duration = max(8.0, min(20.0, total_seconds * 0.25))
            scene = Scene("anchor", anchor_prompt, duration,
                         seg_text, seg_style, character_image_path,
                         scene_id, cinematic_style)
            scenes.append(scene)

        elif i == num_segments - 1:
            anchor_prompt = (
                f"{ANCHOR_VISUAL_PROMPTS.get(seg_style, ANCHOR_VISUAL_PROMPTS['neutral'])}, "
                f"concluding remarks about {topic}, professional sign-off, {cinematic}, "
                f"{bkg_prompt[:100]}"
            )
            duration = max(10.0, min(25.0, total_seconds * 0.3))
            scene = Scene("anchor", anchor_prompt, duration,
                         seg_text, seg_style, character_image_path,
                         scene_id, cinematic_style)
            scenes.append(scene)

        else:
            if use_generated_b_roll:
                words = seg_text.split()[:12]
                broll_prompt = (
                    f"{' '.join(words)}, {bkg_prompt}, "
                    f"matching broadcast atmosphere, {cinematic}"
                )
                duration = max(5.0, min(12.0, total_seconds / num_segments))
                scene = Scene("broll", broll_prompt, duration,
                             seg_text, seg_style, None,
                             scene_id, cinematic_style)
                scenes.append(scene)
            else:
                anchor_prompt = (
                    f"{ANCHOR_VISUAL_PROMPTS.get(seg_style, ANCHOR_VISUAL_PROMPTS['neutral'])}, "
                    f"continuing broadcast about {topic}, {cinematic}"
                )
                duration = max(5.0, min(12.0, total_seconds / num_segments))
                scene = Scene("anchor", anchor_prompt, duration,
                             seg_text, seg_style, character_image_path,
                             scene_id, cinematic_style)
                scenes.append(scene)

        scene_id += 1

    return scenes
