"""Natural-language search over the library.

Hybrid ranking:
* Semantic  - Sub-millisecond vector similarity using FAISS / VectorStore
              over image caption embeddings (when provider supports embeddings).
* Lexical   - Stemmed keyword overlap on caption + tags + filename. Always
              available, so search still works in demo mode or if the
              embedding call fails.
"""
from __future__ import annotations

import logging
import math
import re
from typing import Any, Optional

from .ai.base import AIProvider
from .storage import ImageRecord, Store

log = logging.getLogger(__name__)

_STOP = {
    "a", "an", "the", "of", "in", "on", "at", "with", "and", "or", "to", "me", "show", "find", "all",
    "my", "images", "image", "photos", "photo", "pictures", "picture", "ones", "that", "have", "has",
    "where", "there", "is", "are", "some", "any", "containing", "contain", "featuring", "i", "of",
}


def _stem(tok: str) -> str:
    for suf in ("ing", "es", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            return tok[: -len(suf)]
    return tok


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


def lexical_score(query: str, record: ImageRecord) -> float:
    q = set(tokenize(query))
    if not q:
        return 0.0
    doc = set(tokenize(record.search_text))
    return len(q & doc) / len(q)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def search(
    records: list[ImageRecord],
    query: str,
    provider: Optional[AIProvider] = None,
    store: Optional[Store] = None,
    mode: str = "hybrid",
    tag_filter: Optional[list[str]] = None,
    min_score: float = 0.0,
) -> list[tuple[ImageRecord, float]]:
    """Return (record, score) pairs, best first. Empty query returns everything."""
    # Optional tag filtering upfront
    if tag_filter:
        active_tags = {t.lower().strip() for t in tag_filter if t.strip()}
        if active_tags:
            records = [
                r for r in records
                if any(t.lower() in active_tags for t in r.tags)
            ]

    query = query.strip()
    if not query:
        return [(r, 0.0) for r in records]

    # Generate query embedding if provider supports embeddings
    q_emb: Optional[list[float]] = None
    if mode in ("hybrid", "semantic") and provider and provider.supports_embeddings:
        has_any_embeddings = (
            (store and store.vector_store.count() > 0)
            or any(r.embedding for r in records)
        )
        if has_any_embeddings:
            try:
                q_emb = provider.embed(query)
            except Exception as e:
                log.warning("Embedding generation failed for query %r: %s", query, e)
                q_emb = None

    # Calculate semantic scores via VectorStore if store is provided
    sem_scores_by_id: dict[str, float] = {}
    if q_emb and store and store.vector_store.count() > 0:
        matches = store.vector_store.search(q_emb, top_k=len(records) + 10, min_score=0.0)
        for m_id, m_score, _ in matches:
            sem_scores_by_id[m_id] = m_score

    scored = []
    for r in records:
        lex = lexical_score(query, r) if mode in ("hybrid", "lexical") else 0.0

        if mode in ("hybrid", "semantic") and q_emb:
            if r.id in sem_scores_by_id:
                sem = sem_scores_by_id[r.id]
            elif r.embedding:
                sem = cosine(q_emb, r.embedding)
            else:
                sem = None
        else:
            sem = None

        scored.append((r, lex, sem))

    sems = [s for _, _, s in scored if s is not None]
    top_sem = max(sems) if sems else 0.0
    sem_cutoff = max(0.20, top_sem * 0.80) if top_sem > 0 else 0.20

    results = []
    for r, lex, sem in scored:
        semantic_hit = sem is not None and sem >= sem_cutoff
        if mode == "semantic":
            if semantic_hit and sem is not None and sem >= min_score:
                results.append((r, round(sem, 3)))
        elif mode == "lexical":
            if lex > 0 and lex >= min_score:
                results.append((r, round(lex, 3)))
        else:  # hybrid
            if lex > 0 or semantic_hit:
                score = 0.4 * lex + 0.6 * (sem or 0.0) if sem is not None else lex
                if score >= min_score:
                    results.append((r, round(score, 3)))

    results.sort(key=lambda x: x[1], reverse=True)
    return results
