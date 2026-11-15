"""OpenRouter provider: vision via chat completions, image editing via unified images API, and embeddings."""
from __future__ import annotations

import base64
import os
from typing import Optional
import requests

from .base import AIProvider, ProviderError


class OpenRouterProvider(AIProvider):
    name = "openrouter"
    label = "OpenRouter"
    supports_embeddings = True

    def __init__(
        self,
        api_key: Optional[str] = None,
        flavor: str = "openai",
        vision_model: Optional[str] = None,
        image_model: Optional[str] = None,
        embed_model: Optional[str] = None,
    ):
        key = (
            api_key
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not key:
            raise ProviderError("OPENROUTER_API_KEY is not set")
        self.api_key = key
        self.flavor = (flavor or "openai").lower()

        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise ProviderError("Install the SDK: pip install openai") from e

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )

        if self.flavor == "gemini":
            self.name = "gemini"
            self.label = "Google Gemini (via OpenRouter)"
            default_vision = os.getenv("GEMINI_VISION_MODEL", "google/gemini-2.5-flash")
            default_image = os.getenv("GEMINI_IMAGE_MODEL", "google/gemini-2.5-flash-image")
            default_embed = os.getenv("GEMINI_EMBED_MODEL", "openai/text-embedding-3-small")
        elif self.flavor == "openai":
            self.name = "openai"
            self.label = "OpenAI (via OpenRouter)"
            default_vision = os.getenv("OPENAI_VISION_MODEL", "openai/gpt-4o-mini")
            default_image = os.getenv("OPENAI_IMAGE_MODEL", "openai/gpt-image-1")
            default_embed = os.getenv("OPENAI_EMBED_MODEL", "openai/text-embedding-3-small")
        else:
            self.name = "openrouter"
            self.label = "OpenRouter"
            default_vision = os.getenv("OPENROUTER_VISION_MODEL", "google/gemini-2.5-flash")
            default_image = os.getenv("OPENROUTER_IMAGE_MODEL", "google/gemini-2.5-flash-image")
            default_embed = os.getenv("OPENROUTER_EMBED_MODEL", "openai/text-embedding-3-small")

        self.vision_model = self._normalize_model(vision_model or default_vision)
        self.image_model = self._normalize_model(image_model or default_image)
        self.embed_model = self._normalize_model(embed_model or default_embed)

    def _normalize_model(self, model: str) -> str:
        model = model.strip()
        if "/" in model:
            return model
        if model.startswith("gemini"):
            return f"google/{model}"
        if model.startswith("gpt") or model.startswith("text-embedding"):
            return f"openai/{model}"
        if self.flavor == "gemini":
            return f"google/{model}"
        if self.flavor == "openai":
            return f"openai/{model}"
        return model

    def vision_text(self, prompt: str, image_bytes: bytes) -> str:
        b64 = base64.b64encode(image_bytes).decode()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ]
        headers = {
            "HTTP-Referer": "https://github.com/promptcanvas",
            "X-Title": "PromptCanvas",
        }
        try:
            resp = self.client.chat.completions.create(
                model=self.vision_model,
                response_format={"type": "json_object"},
                messages=messages,
                extra_headers=headers,
            )
            return resp.choices[0].message.content or ""
        except Exception:
            # Retry without JSON schema constraint if model does not support response_format
            try:
                resp = self.client.chat.completions.create(
                    model=self.vision_model,
                    messages=messages,
                    extra_headers=headers,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                raise ProviderError(f"OpenRouter vision call failed ({self.vision_model}): {e}") from e

    def edit_image(self, image_bytes: bytes, prompt: str) -> bytes:
        b64 = base64.b64encode(image_bytes).decode()
        url = "https://openrouter.ai/api/v1/images"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/promptcanvas",
            "X-Title": "PromptCanvas",
        }
        payload = {
            "model": self.image_model,
            "prompt": prompt,
            "input_references": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            ],
            "n": 1,
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=90)
            if resp.status_code != 200:
                raise ProviderError(f"OpenRouter image API error ({resp.status_code}): {resp.text}")
            data = resp.json()
            if "error" in data:
                raise ProviderError(f"OpenRouter error: {data['error']}")
            items = data.get("data", [])
            if not items:
                raise ProviderError(f"OpenRouter returned no image data: {data}")
            img_item = items[0]
            if "b64_json" in img_item and img_item["b64_json"]:
                return base64.b64decode(img_item["b64_json"])
            if "url" in img_item and img_item["url"]:
                img_resp = requests.get(img_item["url"], timeout=45)
                img_resp.raise_for_status()
                return img_resp.content
            raise ProviderError("OpenRouter response had neither b64_json nor url")
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(f"OpenRouter image edit failed ({self.image_model}): {e}") from e

    def embed(self, text: str) -> Optional[list[float]]:
        try:
            resp = self.client.embeddings.create(
                model=self.embed_model,
                input=text,
                extra_headers={
                    "HTTP-Referer": "https://github.com/promptcanvas",
                    "X-Title": "PromptCanvas",
                },
            )
            return list(resp.data[0].embedding)
        except Exception as e:
            raise ProviderError(f"OpenRouter embedding failed ({self.embed_model}): {e}") from e
