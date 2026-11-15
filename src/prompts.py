"""Prompt templates for PromptCanvas.

Three prompt families, each with one job:

1. CAPTION_PROMPT      -> vision LLM returns {caption, tags} used for search.
2. Planner prompt      -> vision LLM turns a loose user request into one precise,
                          single-purpose edit instruction (+ what to preserve).
3. Edit prompt         -> the image-generation model receives the refined
                          instruction wrapped in guard-rails (preserve rules,
                          no text/watermarks, photo-realism).

Keeping these pure strings (no API imports) makes them easy to unit-test and to
iterate on in ``scripts/prompt_playground.py``.
"""
from __future__ import annotations

import json
import re

EDIT_TYPES = ["remove", "add", "replace_background", "recolor", "style", "enhance", "other"]

# ---------------------------------------------------------------------------
# 1. Captioning (drives natural-language search)
# ---------------------------------------------------------------------------
CAPTION_PROMPT = """You are cataloguing photos for a searchable image library.
Describe the image so a person could find it later with a casual text query.

Return ONLY a JSON object with exactly these keys:
{
  "caption": "1-2 sentences: main subjects, setting, notable objects, colours, mood, time of day",
  "tags": ["8-15 lowercase single or two-word keywords: objects, scene, colours, activity, style"]
}
Rules: be literal and specific (say "golden retriever on a sandy beach", not "a pet outdoors").
Do not mention that it is an image or photo. No markdown, no commentary."""

# ---------------------------------------------------------------------------
# 2. Planner: user request -> precise instruction
# ---------------------------------------------------------------------------
PLANNER_PROMPT = """You are the planning stage of an AI photo editor. Look at the image and the
user's request, then rewrite the request as ONE precise instruction for an image-editing model.

User request: "{request}"

Return ONLY a JSON object:
{{
  "edit_type": one of {types},
  "instruction": "imperative sentence naming exactly what to change and where, using visual anchors from THIS image (e.g. 'the man in the red jacket on the left')",
  "preserve": "short list of things that must stay identical (faces, framing, other objects, lighting)"
}}
Rules:
- Resolve vague references ("that thing", "the left one") into concrete descriptions of what you see.
- Keep the instruction to a single change; do not add edits the user did not ask for.
- No markdown, no commentary."""

# ---------------------------------------------------------------------------
# 3. Edit prompt wrapper (sent to the image model)
# ---------------------------------------------------------------------------
EDIT_TEMPLATE = """You are a precise professional photo editor. Edit the provided image.

TASK: {instruction}

{type_hint}
PRESERVE: Keep everything not mentioned in the task unchanged{preserve_clause}. Keep the same
framing, aspect ratio and resolution.

OUTPUT RULES:
- Return the edited image only.
- The result must look natural: match lighting, shadows, perspective, grain and colour grading.
- Do not add text, captions, borders, logos or watermarks."""

TYPE_HINTS = {
    "remove": "REMOVAL: Erase the named element completely and reconstruct what would be behind it "
              "(inpaint seamlessly). Leave no ghosting, outline or blur patch.",
    "add": "ADDITION: Insert the new element so it fits the scene's scale, perspective, lighting and "
           "shadows. Do not alter existing subjects to make room.",
    "replace_background": "BACKGROUND: Keep the foreground subject exactly as is with clean, accurate edges "
                          "(hair, fur, transparent parts). Only the background changes.",
    "recolor": "RECOLOUR: Change only the colour of the named element. Preserve texture, shading and "
               "highlights so it still looks like the same material.",
    "style": "STYLE: Apply the look to the whole image while keeping the composition and subjects "
             "recognisable.",
    "enhance": "ENHANCE: Make subtle, realistic improvements (exposure, contrast, colour, sharpness). "
               "Do not change content or add artefacts.",
    "other": "",
}

# ---------------------------------------------------------------------------
# Pre-set one-click edits: label -> (edit_type, instruction)
# ---------------------------------------------------------------------------
PRESETS: dict[str, tuple[str, str]] = {
    "Remove background": ("replace_background", "Remove the background and replace it with a plain, clean white background."),
    "Studio backdrop": ("replace_background", "Replace the background with a soft neutral grey studio backdrop with gentle lighting."),
    "Remove all clutter": ("remove", "Remove all small distracting objects and clutter from the scene, keeping the main subject and overall setting."),
    "Remove people": ("remove", "Remove every person from the image and fill in the background naturally."),
    "Auto enhance": ("enhance", "Enhance the photo: balance exposure, improve contrast and colour, and add a touch of sharpness."),
    "Vivid colours": ("recolor", "Make the colours richer and more vibrant without looking over-saturated or unnatural."),
    "Black & white": ("style", "Convert the image to a high-quality black and white photograph with rich tonal range."),
    "Golden hour": ("style", "Re-light the scene with warm golden-hour sunlight and a gentle warm colour grade."),
    "Blur background": ("replace_background", "Apply a realistic shallow depth-of-field blur to the background while keeping the subject sharp."),
    "Cartoon style": ("style", "Redraw the image as a clean, colourful cartoon illustration while keeping the composition."),
}


def build_planner_prompt(request: str) -> str:
    return PLANNER_PROMPT.format(request=request.strip().replace('"', "'"), types=json.dumps(EDIT_TYPES))


def build_edit_prompt(edit_type: str, instruction: str, preserve: str = "") -> str:
    edit_type = edit_type if edit_type in TYPE_HINTS else "other"
    preserve = (preserve or "").strip().rstrip(".")
    return EDIT_TEMPLATE.format(
        instruction=instruction.strip(),
        type_hint=TYPE_HINTS[edit_type],
        preserve_clause=f" (especially: {preserve})" if preserve else "",
    )


def parse_json_loose(text: str) -> dict:
    """Parse a JSON object from LLM output, tolerating ```json fences and chatter."""
    if not text:
        return {}
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        data = json.loads(cleaned[start : end + 1])
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}
