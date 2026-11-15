from __future__ import annotations

import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from .base import AIProvider, ProviderError
from .mock_provider import MockProvider


def _get_openrouter_key() -> str | None:
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        return openrouter_key
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key.startswith("sk-or-"):
        return openai_key
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key.startswith("sk-or-"):
        return gemini_key
    return None


def available_providers() -> dict[str, str]:
    """Provider ids -> human labels for those that are usable right now."""
    out: dict[str, str] = {}
    openrouter_key = _get_openrouter_key()

    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    has_real_gemini = gemini_key and not gemini_key.startswith("AQ.Ab8RN6") and not gemini_key.startswith("sk-or-")

    openai_key = os.getenv("OPENAI_API_KEY", "")
    has_real_openai = openai_key and not openai_key.startswith("sk-or-")

    if has_real_gemini:
        out["gemini"] = "Google Gemini (Nano Banana)"
    elif openrouter_key:
        out["gemini"] = "Google Gemini (via OpenRouter)"

    if has_real_openai:
        out["openai"] = "OpenAI (gpt-image)"
    elif openrouter_key:
        out["openai"] = "OpenAI (via OpenRouter)"

    if openrouter_key:
        out["openrouter"] = "OpenRouter (Custom / Multi-model)"

    out["demo"] = "Demo mode (offline, no API)"
    return out


def get_provider(name: str = "auto") -> AIProvider:
    name = (name or "auto").lower()
    if name == "auto":
        name = next(iter(available_providers()))  # first real provider, else demo

    openrouter_key = _get_openrouter_key()
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    has_real_gemini = gemini_key and not gemini_key.startswith("AQ.Ab8RN6") and not gemini_key.startswith("sk-or-")

    openai_key = os.getenv("OPENAI_API_KEY", "")
    has_real_openai = openai_key and not openai_key.startswith("sk-or-")

    if name in ("gemini", "openrouter-gemini"):
        if has_real_gemini:
            from .gemini_provider import GeminiProvider
            return GeminiProvider()
        if openrouter_key:
            from .openrouter_provider import OpenRouterProvider
            return OpenRouterProvider(api_key=openrouter_key, flavor="gemini")
        from .gemini_provider import GeminiProvider
        return GeminiProvider()

    if name in ("openai", "openrouter-openai"):
        if has_real_openai:
            from .openai_provider import OpenAIProvider
            return OpenAIProvider()
        if openrouter_key:
            from .openrouter_provider import OpenRouterProvider
            return OpenRouterProvider(api_key=openrouter_key, flavor="openai")
        from .openai_provider import OpenAIProvider
        return OpenAIProvider()

    if name == "openrouter":
        if not openrouter_key:
            raise ProviderError("OPENROUTER_API_KEY is not set")
        from .openrouter_provider import OpenRouterProvider
        return OpenRouterProvider(api_key=openrouter_key, flavor="openrouter")

    if name == "demo":
        return MockProvider()

    raise ProviderError(f"Unknown provider: {name}")
