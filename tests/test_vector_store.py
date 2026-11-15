import io
import sys
from pathlib import Path
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.storage import Store, normalize_image
from src.vector_store import VectorStore, HAS_FAISS
from src import search, services
from src.ai.mock_provider import MockProvider


def make_png(color=(30, 90, 200), size=(320, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "data")


def test_vector_store_add_and_search(tmp_path):
    vs = VectorStore(tmp_path, dim=4)
    vs.add("img1", [1.0, 0.0, 0.0, 0.0], {"name": "first"})
    vs.add("img2", [0.0, 1.0, 0.0, 0.0], {"name": "second"})
    vs.add("img3", [0.9, 0.1, 0.0, 0.0], {"name": "close_to_first"})

    assert vs.count() == 3
    assert vs.contains("img1")

    # Search query aligned with first
    results = vs.search([1.0, 0.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    top_id, top_score, meta = results[0]
    assert top_id == "img1"
    assert top_score >= 0.99
    assert meta["name"] == "first"

    second_id, _, _ = results[1]
    assert second_id == "img3"


def test_vector_store_persistence(tmp_path):
    vs = VectorStore(tmp_path, dim=4)
    vs.add("a", [1.0, 0.0, 0.0, 0.0], {"tag": "alpha"})
    vs.add("b", [0.0, 1.0, 0.0, 0.0], {"tag": "beta"})
    vs.save()

    # Load in a fresh store instance
    vs2 = VectorStore(tmp_path, dim=4)
    assert vs2.count() == 2
    assert vs2.contains("a")
    assert vs2.contains("b")
    results = vs2.search([1.0, 0.0, 0.0, 0.0], top_k=1)
    assert results[0][0] == "a"


def test_vector_store_delete(tmp_path):
    vs = VectorStore(tmp_path, dim=4)
    vs.add("x", [1.0, 0.0, 0.0, 0.0])
    vs.add("y", [0.0, 1.0, 0.0, 0.0])
    assert vs.count() == 2

    assert vs.delete("x") is True
    assert vs.count() == 1
    assert not vs.contains("x")
    assert vs.contains("y")


def test_store_vector_sync(store):
    rec = store.add_image(make_png(), "sunset.png")
    store.update_metadata(rec.id, "A warm sunset over the ocean", ["sunset", "beach"], [0.1, 0.9, 0.2, 0.0])

    assert store.vector_store.contains(rec.id)
    matches = store.vector_store.search([0.1, 0.9, 0.2, 0.0], top_k=1)
    assert len(matches) == 1
    assert matches[0][0] == rec.id


def test_version_lineage_and_metadata(store):
    rec = store.add_image(make_png(), "base.png")
    v1 = store.add_version(rec.id, make_png((200, 50, 50)), "make red", rec.original_version_id, "mock")
    v2 = store.add_version(rec.id, make_png((50, 200, 50)), "make green", v1.id, "mock")

    assert v1.width > 0 and v1.height > 0
    assert v1.file_size > 0

    lineage = store.get_lineage(v2.id)
    assert len(lineage) == 3
    assert lineage[0].id == rec.original_version_id
    assert lineage[1].id == v1.id
    assert lineage[2].id == v2.id

    tree = store.get_version_tree(rec.id)
    assert tree["total"] == 3
    assert len(tree["roots"]) == 1


def test_batch_reindex_embeddings(store):
    rec1 = store.add_image(make_png(), "pic1.png")
    rec2 = store.add_image(make_png(), "pic2.png")
    store.update_metadata(rec1.id, "A cat on a mat", ["cat", "pet"], None)
    store.update_metadata(rec2.id, "A dog in a park", ["dog", "park"], None)

    class EmbeddingProvider(MockProvider):
        supports_embeddings = True
        def embed(self, text: str):
            return [0.5, 0.5, 0.0, 0.0]

    prov = EmbeddingProvider()
    count = services.batch_reindex_embeddings(store, prov)
    assert count == 2

    img1 = store.get_image(rec1.id)
    assert img1.embedding is not None
    assert store.vector_store.contains(rec1.id)
