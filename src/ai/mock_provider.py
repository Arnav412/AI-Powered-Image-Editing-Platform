"""Offline demo provider.

No API key, no network. It "simulates" edits with simple Pillow filters chosen
by keywords, so the whole app (upload -> edit -> version history -> search) can
be demoed and tested for free. It is NOT real generative editing; the UI labels
it clearly as Demo mode.
"""
from __future__ import annotations

import io
from typing import Optional

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps, ImageStat

from .. import prompts
from .base import AIProvider, Caption, EditPlan

_KEYWORDS = [
    ("grayscale", ("black and white", "black & white", "grayscale", "greyscale", "monochrome", "b&w")),
    ("warm", ("golden", "sunset", "warm", "sepia", "vintage")),
    ("cool", ("cool", "cold", "blue tone", "winter")),
    ("vivid", ("vivid", "vibrant", "saturat", "colour", "color", "richer")),
    ("blur", ("blur", "soften", "bokeh")),
    ("enhance", ("enhance", "improve", "sharpen", "auto", "fix")),
    ("cartoon", ("cartoon", "illustration", "poster", "comic")),
    ("bright", ("brighter", "brighten", "lighter", "lighten")),
    ("dark", ("darker", "darken", "moody", "dim")),
    ("flip", ("flip", "mirror")),
]


def _classify(text: str) -> str:
    t = text.lower()
    for op, words in _KEYWORDS:
        if any(w in t for w in words):
            return op
    return "label"


def _color_name(r: float, g: float, b: float) -> str:
    mx, mn = max(r, g, b), min(r, g, b)
    if mx < 50:
        return "black"
    if mn > 200:
        return "white"
    if mx - mn < 25:
        return "grey"
    if r >= g and r >= b:
        return "orange" if g > b + 40 and g > 100 else "red"
    if g >= r and g >= b:
        return "green"
    return "blue"


class MockProvider(AIProvider):
    name = "demo"
    label = "Demo mode (offline)"
    supports_embeddings = False

    def vision_text(self, prompt: str, image_bytes: bytes) -> str:  # not used; caption/plan overridden
        return "{}"

    def caption(self, image_bytes: bytes) -> Caption:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        small = img.resize((32, 32))
        r, g, b = ImageStat.Stat(small).mean
        color = _color_name(r, g, b)
        brightness = (r + g + b) / 3
        light = "bright" if brightness > 170 else "dark" if brightness < 80 else "balanced"
        shape = "landscape" if img.width > img.height * 1.1 else "portrait" if img.height > img.width * 1.1 else "square"
        caption = f"A {shape} image dominated by {color} tones with {light} lighting (demo caption)."
        return Caption(caption=caption, tags=[color, shape, light, "demo"])

    def plan_edit(self, image_bytes: bytes, request: str) -> EditPlan:
        t = request.lower()
        if any(w in t for w in ("remove", "erase", "delete", "get rid")):
            etype = "remove"
        elif any(w in t for w in ("add ", "insert", "put ")):
            etype = "add"
        elif "background" in t:
            etype = "replace_background"
        elif any(w in t for w in ("colour", "color", "recolor", "make the")):
            etype = "recolor"
        elif any(w in t for w in ("style", "cartoon", "black and white", "golden", "vintage")):
            etype = "style"
        elif any(w in t for w in ("enhance", "improve", "sharpen")):
            etype = "enhance"
        else:
            etype = "other"
        return EditPlan(etype, request.strip(), prompts.build_edit_prompt(etype, request))

    def edit_image(self, image_bytes: bytes, prompt: str) -> bytes:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Use only the TASK line so template boilerplate doesn't trigger keywords.
        task = prompt.split("TASK:", 1)[-1].split("\n", 1)[0]
        op = _classify(task)
        if op == "grayscale":
            out = ImageOps.grayscale(img).convert("RGB")
        elif op == "warm":
            r, g, b = img.split()
            out = Image.merge("RGB", (r.point(lambda v: min(255, int(v * 1.15) + 8)), g, b.point(lambda v: int(v * 0.85))))
        elif op == "cool":
            r, g, b = img.split()
            out = Image.merge("RGB", (r.point(lambda v: int(v * 0.88)), g, b.point(lambda v: min(255, int(v * 1.15) + 8))))
        elif op == "vivid":
            out = ImageEnhance.Color(img).enhance(1.6)
        elif op == "blur":
            out = img.filter(ImageFilter.GaussianBlur(4))
        elif op == "enhance":
            out = ImageEnhance.Sharpness(ImageOps.autocontrast(img, cutoff=1)).enhance(1.6)
        elif op == "cartoon":
            out = ImageOps.posterize(img.filter(ImageFilter.SMOOTH_MORE), 3)
        elif op == "bright":
            out = ImageEnhance.Brightness(img).enhance(1.3)
        elif op == "dark":
            out = ImageEnhance.Brightness(img).enhance(0.7)
        elif op == "flip":
            out = ImageOps.mirror(img)
        else:
            out = img.copy()
            draw = ImageDraw.Draw(out)
            bar = max(18, out.height // 14)
            draw.rectangle([0, out.height - bar, out.width, out.height], fill=(20, 20, 20))
            draw.text((8, out.height - bar + bar // 4), ("DEMO EDIT: " + task.strip())[:90], fill=(255, 255, 255))
        buf = io.BytesIO()
        out.save(buf, "PNG")
        return buf.getvalue()

    def embed(self, text: str) -> Optional[list[float]]:
        return None
