"""Lightweight vector store for image search embeddings (FAISS with JSON fallback).

Provides sub-millisecond semantic search over image caption embeddings.
Uses FAISS (IndexFlatIP with L2 normalization for exact cosine similarity)
when available, with a resilient pure-Python/JSON fallback if FAISS is
not installed.
"""
from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

log = logging.getLogger(__name__)

try:
    import faiss  # type: ignore

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


def _normalize(vec: np.ndarray) -> np.ndarray:
    """Normalize 1D or 2D vector(s) to unit length for cosine similarity via inner product."""
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1e-12, norm)
    return vec / norm


class VectorStore:
    """Persistent vector database for image semantic search."""

    def __init__(self, data_dir: str | Path = "data", dim: int = 1536):
        self.root = Path(data_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self.faiss_path = self.root / "vector_index.faiss"
        self.meta_path = self.root / "vector_meta.json"
        self.json_path = self.root / "vector_store.json"

        # id -> index in matrix, and id -> metadata
        self.id_to_idx: dict[str, int] = {}
        self.idx_to_id: dict[int, str] = {}
        self.metadata: dict[str, dict[str, Any]] = {}
        self.vectors: list[list[float]] = []

        self.faiss_index: Optional[Any] = None
        self.backend = "faiss" if HAS_FAISS else "json"

        self._init_index()
        self.load()

    def _init_index(self) -> None:
        if HAS_FAISS:
            # IndexFlatIP with normalized vectors computes exact cosine similarity
            self.faiss_index = faiss.IndexFlatIP(self.dim)
            self.backend = "faiss"
        else:
            self.faiss_index = None
            self.backend = "json"

    # -- Persistence -----------------------------------------------------
    def save(self) -> None:
        """Persist index and metadata to disk."""
        meta_payload = {
            "dim": self.dim,
            "backend": self.backend,
            "id_to_idx": self.id_to_idx,
            "idx_to_id": {str(k): v for k, v in self.idx_to_id.items()},
            "metadata": self.metadata,
        }
        self.meta_path.write_text(json.dumps(meta_payload, indent=2))

        if HAS_FAISS and self.faiss_index is not None and self.faiss_index.ntotal > 0:
            faiss.write_index(self.faiss_index, str(self.faiss_path))
        else:
            # Fallback JSON dump
            json_payload = {
                "dim": self.dim,
                "items": [
                    {"id": id_, "vector": self.vectors[idx], "metadata": self.metadata.get(id_, {})}
                    for id_, idx in self.id_to_idx.items()
                ],
            }
            self.json_path.write_text(json.dumps(json_payload))

    def load(self) -> None:
        """Load persisted index and metadata if available."""
        if not self.meta_path.exists():
            if self.json_path.exists():
                self._load_json()
            return

        try:
            meta = json.loads(self.meta_path.read_text())
            self.dim = meta.get("dim", self.dim)
            self.id_to_idx = meta.get("id_to_idx", {})
            self.idx_to_id = {int(k): v for k, v in meta.get("idx_to_id", {}).items()}
            self.metadata = meta.get("metadata", {})

            if HAS_FAISS and self.faiss_path.exists() and len(self.id_to_idx) > 0:
                self.faiss_index = faiss.read_index(str(self.faiss_path))
                self.backend = "faiss"
            else:
                if self.json_path.exists():
                    self._load_json()
        except Exception as e:
            log.warning("Could not load vector store (%s), starting fresh", e)
            self.clear()

    def _load_json(self) -> None:
        try:
            data = json.loads(self.json_path.read_text())
            self.clear()
            self.dim = data.get("dim", self.dim)
            for item in data.get("items", []):
                self.add(item["id"], item["vector"], item.get("metadata"))
        except Exception as e:
            log.warning("JSON vector store load failed: %s", e)

    def clear(self) -> None:
        """Wipe the in-memory index."""
        self.id_to_idx.clear()
        self.idx_to_id.clear()
        self.metadata.clear()
        self.vectors.clear()
        self._init_index()

    # -- Mutations -------------------------------------------------------
    def add(
        self,
        id_: str,
        vector: list[float],
        metadata: Optional[dict[str, Any]] = None,
        auto_save: bool = False,
    ) -> None:
        """Add or update an item's embedding."""
        if not vector:
            return
        if len(vector) != self.dim:
            # Adapt dimension if needed on first item
            if len(self.id_to_idx) == 0:
                self.dim = len(vector)
                self._init_index()
            else:
                log.warning("Dimension mismatch: expected %d, got %d. Skipping %s", self.dim, len(vector), id_)
                return

        # If already exists, rebuild to update cleanly
        if id_ in self.id_to_idx:
            self.delete(id_, auto_save=False)

        idx = len(self.id_to_idx)
        self.id_to_idx[id_] = idx
        self.idx_to_id[idx] = id_
        self.metadata[id_] = metadata or {}
        self.vectors.append(vector)

        if HAS_FAISS and self.faiss_index is not None:
            arr = np.array([vector], dtype=np.float32)
            arr = _normalize(arr)
            self.faiss_index.add(arr)

        if auto_save:
            self.save()

    def delete(self, id_: str, auto_save: bool = False) -> bool:
        """Delete an item by id. Rebuilds the index."""
        if id_ not in self.id_to_idx:
            return False

        old_id_to_idx = dict(self.id_to_idx)
        old_vectors = list(self.vectors)
        old_meta = dict(self.metadata)

        self.clear()
        for old_id, old_idx in old_id_to_idx.items():
            if old_id != id_:
                self.add(old_id, old_vectors[old_idx], old_meta.get(old_id))

        if auto_save:
            self.save()
        return True

    # -- Search ----------------------------------------------------------
    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Return (id, similarity_score, metadata) ordered by similarity descending."""
        if not self.id_to_idx or not query_vector:
            return []

        if len(query_vector) != self.dim:
            log.warning("Query vector dimension mismatch: expected %d, got %d", self.dim, len(query_vector))
            return []

        if HAS_FAISS and self.faiss_index is not None and self.faiss_index.ntotal > 0:
            q_arr = np.array([query_vector], dtype=np.float32)
            q_arr = _normalize(q_arr)
            k = min(top_k, self.faiss_index.ntotal)
            scores, indices = self.faiss_index.search(q_arr, k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx in self.idx_to_id:
                    s_float = float(score)
                    if s_float >= min_score:
                        id_ = self.idx_to_id[idx]
                        results.append((id_, s_float, self.metadata.get(id_, {})))
            return results

        # Resilient Pure Python / NumPy fallback
        q_arr = np.array(query_vector, dtype=np.float32)
        q_arr = _normalize(q_arr)

        results = []
        for id_, idx in self.id_to_idx.items():
            vec = np.array(self.vectors[idx], dtype=np.float32)
            vec = _normalize(vec)
            score = float(np.dot(q_arr, vec))
            if score >= min_score:
                results.append((id_, score, self.metadata.get(id_, {})))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def count(self) -> int:
        return len(self.id_to_idx)

    def contains(self, id_: str) -> bool:
        return id_ in self.id_to_idx

    def get_metadata(self, id_: str) -> Optional[dict[str, Any]]:
        return self.metadata.get(id_)
