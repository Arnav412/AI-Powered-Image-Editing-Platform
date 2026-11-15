"""Provider interface.

A provider only has to implement three primitives:

* ``vision_text``  - image + prompt -> text   (captioning, planning)
* ``edit_image``   - image + prompt -> image  (the actual edit)
* ``embed``        - text -> vector           (optional, for semantic search)

Captioning and edit-planning are built on top of those in this base class, so
all providers share the same prompts and the same fallback behaviour.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from .. import prompts


class ProviderError(RuntimeError):
    """Raised for any API failure, refusal, or missing configuration."""


@dataclass
class Caption:
    caption: str
    tags: list[str] = field(default_factory=list)


@dataclass
class EditPlan:
    edit_type: str
    instruction: str  # refined, precise instruction
    final_prompt: str  # what is actually sent to the image model


class AIProvider(ABC):
    name = "base"
    label = "Base"
    supports_embeddings = False

    @abstractmethod
    def vision_text(self, prompt: str, image_bytes: bytes) -> str: ...

    @abstractmethod
    def edit_image(self, image_bytes: bytes, prompt: str) -> bytes: ...

    def embed(self, text: str) -> Optional[list[float]]:
        return None

    # -- shared logic -----------------------------------------------------
    def caption(self, image_bytes: bytes) -> Caption:
        data = prompts.parse_json_loose(self.vision_text(prompts.CAPTION_PROMPT, image_bytes))
        caption = str(data.get("caption", "")).strip()
        tags = [str(t).strip().lower() for t in data.get("tags", []) if str(t).strip()]
        if not caption:
            raise ProviderError("Captioning returned no caption")
        return Caption(caption=caption, tags=tags[:15])

    def plan_edit(self, image_bytes: bytes, request: str) -> EditPlan:
        """Interpret the user's request with the vision LLM.

        Falls back to the raw request if the planner call fails, so a flaky
        text model never blocks an edit.
        """
        data: dict = {}
        try:
            raw = self.vision_text(prompts.build_planner_prompt(request), image_bytes)
            data = prompts.parse_json_loose(raw)
        except ProviderError:
            data = {}
        edit_type = data.get("edit_type")
        edit_type = edit_type if edit_type in prompts.EDIT_TYPES else "other"
        instruction = str(data.get("instruction") or request).strip()
        preserve = str(data.get("preserve") or "")
        return EditPlan(
            edit_type=edit_type,
            instruction=instruction,
            final_prompt=prompts.build_edit_prompt(edit_type, instruction, preserve),
        )
