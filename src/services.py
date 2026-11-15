"""Application logic that ties storage, AI providers, and vector search together.

The Streamlit UI calls functions here, keeping the business logic testable
without a browser.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Optional

from . import prompts
from .ai.base import AIProvider, ProviderError
from .exceptions import ImageNotFoundError, VersionNotFoundError
from .storage import ImageRecord, Store, VersionRecord

log = logging.getLogger(__name__)


def analyze_image(store: Store, provider: AIProvider, image_id: str) -> ImageRecord:
    """(Re)generate caption, tags and embedding from the image's CURRENT version."""
    img = store.get_image(image_id)
    if img is None:
        raise ImageNotFoundError(f"Image not found: {image_id}")
    cur_ver = store.get_version(img.current_version_id)
    if cur_ver is None:
        raise VersionNotFoundError(f"Current version not found for image {image_id}")
    data = store.read_version_bytes(cur_ver)
    cap = provider.caption(data)
    embedding = None
    if provider.supports_embeddings:
        try:
            embedding = provider.embed(f"{cap.caption}. Tags: {', '.join(cap.tags)}")
        except ProviderError as e:
            log.warning("Embedding failed, search will be keyword-only: %s", e)
    store.update_metadata(image_id, cap.caption, cap.tags, embedding)
    updated = store.get_image(image_id)
    if updated is None:
        raise ImageNotFoundError(f"Image {image_id} missing after metadata update")
    return updated


def ingest_upload(store: Store, provider: AIProvider, data: bytes, name: str) -> ImageRecord:
    """Save an upload, then caption it. A captioning failure never loses the upload."""
    img = store.add_image(data, name)
    try:
        return analyze_image(store, provider, img.id)
    except ProviderError as e:
        log.warning("Captioning failed for %s: %s", name, e)
        return img


def apply_edit(
    store: Store,
    provider: AIProvider,
    image_id: str,
    instruction: str,
    refresh_caption: bool = True,
    use_planner: bool = True,
) -> VersionRecord:
    """Edit the image's current version and record the result as a new version.

    Steps: (1) planner LLM refines the request, (2) image model performs the
    edit, (3) result is stored as a child of the version it was edited from,
    (4) caption/embedding are refreshed so search reflects the edit.
    """
    instruction = instruction.strip()
    if not instruction:
        raise ValueError("Please enter an edit instruction.")
    img = store.get_image(image_id)
    if img is None:
        raise ImageNotFoundError(f"Image not found: {image_id}")
    base = store.get_version(img.current_version_id)
    if base is None:
        raise VersionNotFoundError(f"Base version {img.current_version_id} not found")
    base_bytes = store.read_version_bytes(base)

    t0 = time.perf_counter()
    if use_planner:
        plan = provider.plan_edit(base_bytes, instruction)
        final_prompt, refined = plan.final_prompt, plan.instruction
    else:
        final_prompt, refined = prompts.build_edit_prompt("other", instruction), instruction

    result = provider.edit_image(base_bytes, final_prompt)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    version = store.add_version(
        image_id,
        result,
        prompt=instruction,
        parent_id=base.id,
        provider=provider.name,
        refined_prompt=refined,
        duration_ms=round(elapsed_ms, 2),
    )
    if refresh_caption:
        try:
            analyze_image(store, provider, image_id)
        except ProviderError as e:
            log.warning("Caption refresh failed: %s", e)
    return version


def apply_preset(store: Store, provider: AIProvider, image_id: str, preset: str, **kw) -> VersionRecord:
    _etype, instruction = prompts.PRESETS[preset]
    return apply_edit(store, provider, image_id, instruction, **kw)


def restore_version(
    store: Store,
    provider: AIProvider,
    image_id: str,
    version_id: str,
    refresh_caption: bool = True,
) -> None:
    """Make an older version current (non-destructive: nothing is deleted)."""
    store.set_current_version(image_id, version_id)
    if refresh_caption:
        try:
            analyze_image(store, provider, image_id)
        except ProviderError as e:
            log.warning("Caption refresh failed after restore: %s", e)


def batch_reindex_embeddings(
    store: Store,
    provider: AIProvider,
    force: bool = False,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> int:
    """Generate search embeddings for all uploaded images in the library.

    Returns the count of newly embedded images.
    """
    images = store.list_images()
    total = len(images)
    reindexed = 0

    for idx, img in enumerate(images):
        if force or not img.embedding:
            try:
                # If caption is missing, run full analyze_image
                if not img.caption:
                    analyze_image(store, provider, img.id)
                elif provider.supports_embeddings:
                    emb = provider.embed(f"{img.caption}. Tags: {', '.join(img.tags)}")
                    if emb:
                        store.update_metadata(img.id, img.caption, img.tags, emb)
                reindexed += 1
            except Exception as e:
                log.warning("Failed to generate embedding for %s (%s): %s", img.id, img.name, e)

        if on_progress:
            on_progress(idx + 1, total)

    store.sync_vector_store()
    return reindexed
