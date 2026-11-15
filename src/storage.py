"""Local image storage + version history, backed by SQLite, vector index, and the filesystem.

Layout on disk::

    data/
      library.db
      vector_index.faiss / vector_meta.json / vector_store.json
      images/<image_id>/v0.png, v0_thumb.png, v1.png, v1_thumb.png, ...

Every edit creates a NEW version row (never overwrites), so history is
non-destructive. ``images.current_version_id`` points at the active version.
"""
from __future__ import annotations

import io
import json
import logging
import shutil
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from PIL import Image, ImageOps

from .exceptions import (
    ImageNotFoundError,
    ImageValidationError,
    StorageError,
    VersionNotFoundError,
)

log = logging.getLogger(__name__)

MAX_SIDE = 2048  # cap stored resolution; keeps API payloads and disk small
THUMB_SIDE = 256


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_image(data: bytes) -> bytes:
    """Decode any supported image, fix EXIF rotation, cap size, re-encode as PNG."""
    if not data:
        raise ImageValidationError("Empty image data provided.")
    try:
        # Validate image integrity
        img = Image.open(io.BytesIO(data))
        img.verify()
        # Re-open after verify to perform operations
        img = Image.open(io.BytesIO(data))
    except Exception as e:
        raise ImageValidationError(f"Invalid or corrupted image: {e}") from e

    try:
        img = ImageOps.exif_transpose(img)
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return buf.getvalue()
    except Exception as e:
        raise ImageValidationError(f"Failed to normalize image: {e}") from e


def make_thumbnail(data: bytes) -> bytes:
    img = Image.open(io.BytesIO(data))
    img.thumbnail((THUMB_SIDE, THUMB_SIDE))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


@dataclass
class VersionRecord:
    id: str
    image_id: str
    version_no: int
    parent_id: Optional[str]
    kind: str  # "original" | "edit"
    prompt: str
    refined_prompt: str
    provider: str
    file_path: str
    thumb_path: str
    created_at: str
    width: int = 0
    height: int = 0
    file_size: int = 0
    duration_ms: float = 0.0


@dataclass
class ImageRecord:
    id: str
    name: str
    caption: str
    tags: list[str]
    embedding: Optional[list[float]]
    created_at: str
    original_version_id: str
    current_version_id: str
    version_count: int = 1
    width: int = 0
    height: int = 0

    @property
    def search_text(self) -> str:
        return f"{self.caption}. Tags: {', '.join(self.tags)}. File: {self.name}"


SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    caption TEXT NOT NULL DEFAULT '',
    tags TEXT NOT NULL DEFAULT '[]',
    embedding TEXT,
    created_at TEXT NOT NULL,
    original_version_id TEXT NOT NULL,
    current_version_id TEXT NOT NULL,
    width INTEGER NOT NULL DEFAULT 0,
    height INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS versions (
    id TEXT PRIMARY KEY,
    image_id TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    version_no INTEGER NOT NULL,
    parent_id TEXT,
    kind TEXT NOT NULL,
    prompt TEXT NOT NULL DEFAULT '',
    refined_prompt TEXT NOT NULL DEFAULT '',
    provider TEXT NOT NULL DEFAULT '',
    file_path TEXT NOT NULL,
    thumb_path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    width INTEGER NOT NULL DEFAULT 0,
    height INTEGER NOT NULL DEFAULT 0,
    file_size INTEGER NOT NULL DEFAULT 0,
    duration_ms REAL NOT NULL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_versions_image ON versions(image_id, version_no);
"""


class Store:
    def __init__(self, data_dir: str | Path = "data"):
        self.root = Path(data_dir)
        self.img_dir = self.root / "images"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "library.db"

        with self._tx() as c:
            c.executescript(SCHEMA)
            self._migrate_schema(c)

        from .vector_store import VectorStore
        self.vector_store = VectorStore(self.root)
        self.sync_vector_store()

    def _migrate_schema(self, c: sqlite3.Connection) -> None:
        """Safely apply backward-compatible schema updates to existing databases."""
        cols_versions = {row[1] for row in c.execute("PRAGMA table_info(versions)").fetchall()}
        if "width" not in cols_versions:
            c.execute("ALTER TABLE versions ADD COLUMN width INTEGER NOT NULL DEFAULT 0")
        if "height" not in cols_versions:
            c.execute("ALTER TABLE versions ADD COLUMN height INTEGER NOT NULL DEFAULT 0")
        if "file_size" not in cols_versions:
            c.execute("ALTER TABLE versions ADD COLUMN file_size INTEGER NOT NULL DEFAULT 0")
        if "duration_ms" not in cols_versions:
            c.execute("ALTER TABLE versions ADD COLUMN duration_ms REAL NOT NULL DEFAULT 0.0")

        cols_images = {row[1] for row in c.execute("PRAGMA table_info(images)").fetchall()}
        if "width" not in cols_images:
            c.execute("ALTER TABLE images ADD COLUMN width INTEGER NOT NULL DEFAULT 0")
        if "height" not in cols_images:
            c.execute("ALTER TABLE images ADD COLUMN height INTEGER NOT NULL DEFAULT 0")

    # -- plumbing ---------------------------------------------------------
    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise StorageError(f"Database transaction error: {e}") from e
        finally:
            conn.close()

    @staticmethod
    def _version(row: sqlite3.Row) -> VersionRecord:
        keys = row.keys()
        d = {k: row[k] for k in keys if k in VersionRecord.__dataclass_fields__}
        return VersionRecord(**d)

    @staticmethod
    def _image(row: sqlite3.Row, count: int = 1) -> ImageRecord:
        emb = json.loads(row["embedding"]) if row["embedding"] else None
        keys = row.keys()
        return ImageRecord(
            id=row["id"],
            name=row["name"],
            caption=row["caption"],
            tags=json.loads(row["tags"] or "[]"),
            embedding=emb,
            created_at=row["created_at"],
            original_version_id=row["original_version_id"],
            current_version_id=row["current_version_id"],
            version_count=count,
            width=row["width"] if "width" in keys and row["width"] is not None else 0,
            height=row["height"] if "height" in keys and row["height"] is not None else 0,
        )

    def _write_version_files(
        self, image_id: str, version_no: int, data: bytes
    ) -> tuple[str, str, int, int, int]:
        folder = self.img_dir / image_id
        folder.mkdir(parents=True, exist_ok=True)
        rel_img = f"images/{image_id}/v{version_no}.png"
        rel_thumb = f"images/{image_id}/v{version_no}_thumb.png"
        (self.root / rel_img).write_bytes(data)
        (self.root / rel_thumb).write_bytes(make_thumbnail(data))

        with Image.open(io.BytesIO(data)) as pil_img:
            w, h = pil_img.size
        size = len(data)
        return rel_img, rel_thumb, w, h, size

    # -- vector store sync ------------------------------------------------
    def sync_vector_store(self) -> int:
        """Sync embeddings from SQLite into the vector store."""
        count = 0
        with self._tx() as c:
            rows = c.execute(
                "SELECT id, name, caption, tags, embedding FROM images WHERE embedding IS NOT NULL"
            ).fetchall()
            for r in rows:
                if not self.vector_store.contains(r["id"]) and r["embedding"]:
                    emb = json.loads(r["embedding"])
                    self.vector_store.add(
                        r["id"],
                        emb,
                        metadata={
                            "name": r["name"],
                            "caption": r["caption"],
                            "tags": json.loads(r["tags"] or "[]"),
                        },
                        auto_save=False,
                    )
                    count += 1
        if count > 0:
            self.vector_store.save()
        return count

    # -- images -----------------------------------------------------------
    def add_image(self, data: bytes, name: str) -> ImageRecord:
        data = normalize_image(data)
        image_id = uuid.uuid4().hex[:12]
        version_id = uuid.uuid4().hex[:12]
        rel_img, rel_thumb, w, h, size = self._write_version_files(image_id, 0, data)
        now = _now()
        with self._tx() as c:
            c.execute(
                "INSERT INTO images (id,name,created_at,original_version_id,current_version_id,width,height)"
                " VALUES (?,?,?,?,?,?,?)",
                (image_id, name, now, version_id, version_id, w, h),
            )
            c.execute(
                "INSERT INTO versions (id,image_id,version_no,parent_id,kind,file_path,thumb_path,created_at,"
                "width,height,file_size) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (version_id, image_id, 0, None, "original", rel_img, rel_thumb, now, w, h, size),
            )
        img = self.get_image(image_id)
        if img is None:
            raise StorageError(f"Failed to create image record for {image_id}")
        return img

    def get_image(self, image_id: str) -> Optional[ImageRecord]:
        with self._tx() as c:
            row = c.execute("SELECT * FROM images WHERE id=?", (image_id,)).fetchone()
            if not row:
                return None
            n = c.execute("SELECT COUNT(*) FROM versions WHERE image_id=?", (image_id,)).fetchone()[0]
            return self._image(row, n)

    def list_images(self) -> list[ImageRecord]:
        with self._tx() as c:
            rows = c.execute(
                "SELECT i.*, (SELECT COUNT(*) FROM versions v WHERE v.image_id=i.id) AS n"
                " FROM images i ORDER BY i.created_at DESC, i.rowid DESC"
            ).fetchall()
            return [self._image(r, r["n"]) for r in rows]

    def update_metadata(
        self,
        image_id: str,
        caption: str,
        tags: list[str],
        embedding: Optional[list[float]] = None,
    ) -> None:
        with self._tx() as c:
            c.execute(
                "UPDATE images SET caption=?, tags=?, embedding=? WHERE id=?",
                (caption, json.dumps(tags), json.dumps(embedding) if embedding else None, image_id),
            )
        if embedding:
            img = self.get_image(image_id)
            self.vector_store.add(
                image_id,
                embedding,
                metadata={"name": img.name if img else "", "caption": caption, "tags": tags},
                auto_save=True,
            )

    def delete_image(self, image_id: str) -> None:
        with self._tx() as c:
            c.execute("DELETE FROM versions WHERE image_id=?", (image_id,))
            c.execute("DELETE FROM images WHERE id=?", (image_id,))
        shutil.rmtree(self.img_dir / image_id, ignore_errors=True)
        self.vector_store.delete(image_id, auto_save=True)

    # -- versions ---------------------------------------------------------
    def list_versions(self, image_id: str) -> list[VersionRecord]:
        with self._tx() as c:
            rows = c.execute(
                "SELECT * FROM versions WHERE image_id=? ORDER BY version_no", (image_id,)
            ).fetchall()
            return [self._version(r) for r in rows]

    def get_version(self, version_id: str) -> Optional[VersionRecord]:
        with self._tx() as c:
            row = c.execute("SELECT * FROM versions WHERE id=?", (version_id,)).fetchone()
            return self._version(row) if row else None

    def add_version(
        self,
        image_id: str,
        data: bytes,
        prompt: str,
        parent_id: Optional[str],
        provider: str,
        refined_prompt: str = "",
        make_current: bool = True,
        duration_ms: float = 0.0,
    ) -> VersionRecord:
        data = normalize_image(data)
        version_id = uuid.uuid4().hex[:12]
        with self._tx() as c:
            nxt = c.execute(
                "SELECT COALESCE(MAX(version_no), -1) + 1 FROM versions WHERE image_id=?",
                (image_id,),
            ).fetchone()[0]
        rel_img, rel_thumb, w, h, size = self._write_version_files(image_id, nxt, data)
        with self._tx() as c:
            c.execute(
                "INSERT INTO versions (id,image_id,version_no,parent_id,kind,prompt,refined_prompt,"
                "provider,file_path,thumb_path,created_at,width,height,file_size,duration_ms)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    version_id,
                    image_id,
                    nxt,
                    parent_id,
                    "edit",
                    prompt,
                    refined_prompt,
                    provider,
                    rel_img,
                    rel_thumb,
                    _now(),
                    w,
                    h,
                    size,
                    duration_ms,
                ),
            )
            if make_current:
                c.execute(
                    "UPDATE images SET current_version_id=?, width=?, height=? WHERE id=?",
                    (version_id, w, h, image_id),
                )
        v = self.get_version(version_id)
        if v is None:
            raise StorageError(f"Failed to create version {version_id}")
        return v

    def set_current_version(self, image_id: str, version_id: str) -> None:
        v = self.get_version(version_id)
        if not v or v.image_id != image_id:
            raise VersionNotFoundError(f"Version {version_id} does not belong to image {image_id}")
        with self._tx() as c:
            c.execute(
                "UPDATE images SET current_version_id=?, width=?, height=? WHERE id=?",
                (version_id, v.width, v.height, image_id),
            )

    def get_lineage(self, version_id: str) -> list[VersionRecord]:
        """Return the chain of versions leading from the original (v0) down to version_id."""
        chain: list[VersionRecord] = []
        curr = self.get_version(version_id)
        visited = set()
        while curr and curr.id not in visited:
            visited.add(curr.id)
            chain.append(curr)
            if not curr.parent_id:
                break
            curr = self.get_version(curr.parent_id)
        chain.reverse()
        return chain

    def get_version_tree(self, image_id: str) -> dict[str, Any]:
        """Return hierarchical tree of versions for image_id showing branching."""
        versions = self.list_versions(image_id)
        nodes: dict[str, dict[str, Any]] = {v.id: {"version": v, "children": []} for v in versions}
        roots: list[dict[str, Any]] = []
        for v in versions:
            if v.parent_id and v.parent_id in nodes:
                nodes[v.parent_id]["children"].append(nodes[v.id])
            else:
                roots.append(nodes[v.id])
        return {"roots": roots, "total": len(versions)}

    def read_version_bytes(self, version: VersionRecord) -> bytes:
        p = self.root / version.file_path
        if not p.exists():
            raise StorageError(f"Version file missing on disk: {version.file_path}")
        return p.read_bytes()

    def read_thumb_bytes(self, version: VersionRecord) -> bytes:
        p = self.root / version.thumb_path
        if not p.exists():
            raise StorageError(f"Thumbnail file missing on disk: {version.thumb_path}")
        return p.read_bytes()
