"""Google Gemini provider (image editing via "Nano Banana" image models)."""
from __future__ import annotations

import os
from typing import Optional

from .base import AIProvider, ProviderError


class GeminiProvider(AIProvider):
    name = "gemini"
    label = "Google Gemini"
    supports_embeddings = True

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ProviderError("GEMINI_API_KEY is not set")
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:  # pragma: no cover
            raise ProviderError("Install the SDK: pip install google-genai") from e
        self._types = types
        self.client = genai.Client(api_key=key)
        # Models are env-configurable because Google renames/replaces them often.
        self.vision_model = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")
        self.image_model = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
        self.embed_model = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")

    def _image_part(self, image_bytes: bytes):
        return self._types.Part.from_bytes(data=image_bytes, mime_type="image/png")

    def vision_text(self, prompt: str, image_bytes: bytes) -> str:
        try:
            resp = self.client.models.generate_content(
                model=self.vision_model,
                contents=[self._image_part(image_bytes), prompt],
                config=self._types.GenerateContentConfig(response_mime_type="application/json"),
            )
            return resp.text or ""
        except Exception as e:
            raise ProviderError(f"Gemini vision call failed: {e}") from e

    def edit_image(self, image_bytes: bytes, prompt: str) -> bytes:
        try:
            resp = self.client.models.generate_content(
                model=self.image_model,
                contents=[prompt, self._image_part(image_bytes)],
                config=self._types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
            )
        except Exception as e:
            raise ProviderError(f"Gemini image edit failed: {e}") from e

        text_parts = []
        for cand in resp.candidates or []:
            for part in (cand.content.parts if cand.content and cand.content.parts else []):
                if getattr(part, "inline_data", None) and part.inline_data.data:
                    return part.inline_data.data
                if getattr(part, "text", None):
                    text_parts.append(part.text)
        reason = " ".join(text_parts).strip() or "no image returned (possibly blocked by safety filters)"
        raise ProviderError(f"Gemini did not return an image: {reason}")

    def embed(self, text: str) -> Optional[list[float]]:
        try:
            resp = self.client.models.embed_content(model=self.embed_model, contents=text)
            return list(resp.embeddings[0].values)
        except Exception as e:
            raise ProviderError(f"Gemini embedding failed: {e}") from e
