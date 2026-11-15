"""OpenAI provider: GPT vision for captions/planning, gpt-image for edits."""
from __future__ import annotations

import base64
import io
import os
from typing import Optional

from .base import AIProvider, ProviderError


class OpenAIProvider(AIProvider):
    name = "openai"
    label = "OpenAI"
    supports_embeddings = True

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ProviderError("OPENAI_API_KEY is not set")
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise ProviderError("Install the SDK: pip install openai") from e
        self.client = OpenAI(api_key=key)
        self.vision_model = os.getenv("OPENAI_VISION_MODEL", "gpt-5-mini")
        self.image_model = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
        self.embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    def vision_text(self, prompt: str, image_bytes: bytes) -> str:
        b64 = base64.b64encode(image_bytes).decode()
        try:
            resp = self.client.chat.completions.create(
                model=self.vision_model,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        ],
                    }
                ],
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            raise ProviderError(f"OpenAI vision call failed: {e}") from e

    def edit_image(self, image_bytes: bytes, prompt: str) -> bytes:
        try:
            result = self.client.images.edit(
                model=self.image_model,
                image=("image.png", io.BytesIO(image_bytes), "image/png"),
                prompt=prompt,
            )
            b64 = result.data[0].b64_json
            if not b64:
                raise ProviderError("OpenAI returned no image data")
            return base64.b64decode(b64)
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(f"OpenAI image edit failed: {e}") from e

    def embed(self, text: str) -> Optional[list[float]]:
        try:
            resp = self.client.embeddings.create(model=self.embed_model, input=text)
            return list(resp.data[0].embedding)
        except Exception as e:
            raise ProviderError(f"OpenAI embedding failed: {e}") from e
