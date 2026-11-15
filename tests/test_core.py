import io
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import prompts, search, services  # noqa: E402
from src.ai.base import AIProvider, ProviderError  # noqa: E402
from src.ai.mock_provider import MockProvider  # noqa: E402
from src.storage import Store  # noqa: E402


def make_png(color=(30, 90, 200), size=(320, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "data")


@pytest.fixture
def provider():
    return MockProvider()


# ---------------------------------------------------------------- storage
def test_add_image_creates_original_version(store):
    rec = store.add_image(make_png(), "a.png")
    versions = store.list_versions(rec.id)
    assert len(versions) == 1 and versions[0].kind == "original" and versions[0].version_no == 0
    assert rec.current_version_id == rec.original_version_id
    assert (store.root / versions[0].thumb_path).exists()


def test_invalid_image_rejected(store):
    with pytest.raises(Exception):
        store.add_image(b"not an image", "bad.png")
    assert store.list_images() == []


def test_versions_are_never_overwritten(store):
    rec = store.add_image(make_png(), "a.png")
    v1 = store.add_version(rec.id, make_png((255, 0, 0)), "make red", rec.original_version_id, "demo")
    v2 = store.add_version(rec.id, make_png((0, 255, 0)), "make green", v1.id, "demo")
    vs = store.list_versions(rec.id)
    assert [v.version_no for v in vs] == [0, 1, 2]
    assert v2.parent_id == v1.id
    assert store.get_image(rec.id).current_version_id == v2.id
    assert store.read_version_bytes(vs[0]) != store.read_version_bytes(vs[1])


def test_restore_version_is_non_destructive(store):
    rec = store.add_image(make_png(), "a.png")
    store.add_version(rec.id, make_png((255, 0, 0)), "x", rec.original_version_id, "demo")
    store.set_current_version(rec.id, rec.original_version_id)
    assert store.get_image(rec.id).current_version_id == rec.original_version_id
    assert len(store.list_versions(rec.id)) == 2


def test_cannot_set_foreign_version_current(store):
    a = store.add_image(make_png(), "a.png")
    b = store.add_image(make_png(), "b.png")
    with pytest.raises(ValueError):
        store.set_current_version(a.id, b.original_version_id)


def test_delete_image_removes_files_and_rows(store):
    rec = store.add_image(make_png(), "a.png")
    store.delete_image(rec.id)
    assert store.get_image(rec.id) is None
    assert not (store.img_dir / rec.id).exists()


def test_large_image_is_capped(store):
    rec = store.add_image(make_png(size=(4000, 3000)), "big.png")
    v = store.get_version(rec.current_version_id)
    w, h = Image.open(io.BytesIO(store.read_version_bytes(v))).size
    assert max(w, h) <= 2048


# --------------------------------------------------------------- services
def test_ingest_generates_caption(store, provider):
    rec = services.ingest_upload(store, provider, make_png(), "a.png")
    assert rec.caption and "blue" in rec.caption.lower()
    assert "blue" in rec.tags


class FailingCaptionProvider(MockProvider):
    def caption(self, image_bytes):
        raise ProviderError("boom")


def test_upload_survives_caption_failure(store):
    rec = services.ingest_upload(store, FailingCaptionProvider(), make_png(), "a.png")
    assert store.get_image(rec.id) is not None and rec.caption == ""


def test_apply_edit_creates_version_with_lineage(store, provider):
    rec = services.ingest_upload(store, provider, make_png(), "a.png")
    v = services.apply_edit(store, provider, rec.id, "make it black and white")
    assert v.version_no == 1 and v.prompt == "make it black and white"
    assert v.parent_id == rec.original_version_id and v.provider == "demo"
    assert store.get_image(rec.id).current_version_id == v.id
    # grayscale of a blue image should be neutral -> caption refreshed to grey
    assert "grey" in store.get_image(rec.id).caption.lower()


def test_edits_chain_from_current_version(store, provider):
    rec = services.ingest_upload(store, provider, make_png(), "a.png")
    v1 = services.apply_edit(store, provider, rec.id, "make it darker")
    v2 = services.apply_edit(store, provider, rec.id, "flip it")
    assert v2.parent_id == v1.id


def test_empty_instruction_rejected(store, provider):
    rec = services.ingest_upload(store, provider, make_png(), "a.png")
    with pytest.raises(ValueError):
        services.apply_edit(store, provider, rec.id, "   ")


def test_failed_edit_creates_no_version(store):
    class Bad(MockProvider):
        def edit_image(self, image_bytes, prompt):
            raise ProviderError("blocked")

    rec = services.ingest_upload(store, Bad(), make_png(), "a.png")
    with pytest.raises(ProviderError):
        services.apply_edit(store, Bad(), rec.id, "do something")
    assert len(store.list_versions(rec.id)) == 1


def test_every_preset_runs(store, provider):
    rec = services.ingest_upload(store, provider, make_png(), "a.png")
    for name in prompts.PRESETS:
        services.apply_preset(store, provider, rec.id, name, refresh_caption=False)
    assert len(store.list_versions(rec.id)) == 1 + len(prompts.PRESETS)


def test_restore_version(store, provider):
    rec = services.ingest_upload(store, provider, make_png(), "a.png")
    services.apply_edit(store, provider, rec.id, "make it black and white")
    services.restore_version(store, provider, rec.id, rec.original_version_id)
    assert "blue" in store.get_image(rec.id).caption.lower()


# ---------------------------------------------------------------- prompts
def test_parse_json_loose_handles_fences_and_chatter():
    assert prompts.parse_json_loose('```json\n{"a": 1}\n```') == {"a": 1}
    assert prompts.parse_json_loose('Sure! {"a": 1} hope it helps') == {"a": 1}
    assert prompts.parse_json_loose("nothing here") == {}
    assert prompts.parse_json_loose("") == {}


def test_edit_prompt_includes_guardrails():
    p = prompts.build_edit_prompt("remove", "Remove the dog.", "the sofa")
    assert "Remove the dog." in p and "REMOVAL" in p and "the sofa" in p and "watermarks" in p


def test_unknown_edit_type_falls_back():
    assert "TASK: x" in prompts.build_edit_prompt("nonsense", "x")


class JsonProvider(AIProvider):
    """Provider whose vision model returns canned text, to test base-class logic."""
    name = "fake"

    def __init__(self, text):
        self.text = text

    def vision_text(self, prompt, image_bytes):
        if isinstance(self.text, Exception):
            raise self.text
        return self.text

    def edit_image(self, image_bytes, prompt):
        return image_bytes


def test_planner_parses_valid_json():
    p = JsonProvider('{"edit_type":"remove","instruction":"Remove the man in the red jacket","preserve":"the car"}')
    plan = p.plan_edit(b"", "remove that guy")
    assert plan.edit_type == "remove" and "red jacket" in plan.final_prompt and "the car" in plan.final_prompt


def test_planner_falls_back_when_llm_fails_or_is_garbage():
    for text in (ProviderError("down"), "total garbage", '{"edit_type":"bogus"}'):
        plan = JsonProvider(text).plan_edit(b"", "add a tree")
        assert plan.edit_type == "other" and "add a tree" in plan.final_prompt


def test_caption_parsing_and_error():
    ok = JsonProvider('{"caption":"A dog on a beach","tags":["Dog"," beach ",""]}').caption(b"")
    assert ok.caption == "A dog on a beach" and ok.tags == ["dog", "beach"]
    with pytest.raises(ProviderError):
        JsonProvider("{}").caption(b"")


# ----------------------------------------------------------------- search
def test_lexical_search_ranks_relevant_first(store, provider):
    a = store.add_image(make_png(), "a.png")
    b = store.add_image(make_png(), "b.png")
    store.update_metadata(a.id, "A dog running on a sandy beach at sunset", ["dog", "beach", "sunset"])
    store.update_metadata(b.id, "A city street at night with neon signs", ["city", "night", "neon"])
    res = search.search(store.list_images(), "show me the ones with a beach")
    assert [r.id for r, _ in res] == [a.id]
    res = search.search(store.list_images(), "dogs")  # stemming: dogs -> dog
    assert [r.id for r, _ in res] == [a.id]
    assert search.search(store.list_images(), "spaceship") == []
    assert len(search.search(store.list_images(), "")) == 2


class EmbedProvider(MockProvider):
    supports_embeddings = True

    def embed(self, text):
        t = text.lower()
        return [1.0 if "ocean" in t or "sea" in t or "beach" in t else 0.0,
                1.0 if "city" in t else 0.0, 0.1]


def test_semantic_search_finds_synonyms(store):
    a = store.add_image(make_png(), "a.png")
    b = store.add_image(make_png(), "b.png")
    p = EmbedProvider()
    store.update_metadata(a.id, "Waves rolling onto the sea shore", ["sea"], p.embed("sea"))
    store.update_metadata(b.id, "Skyscrapers downtown", ["city"], p.embed("city"))
    res = search.search(store.list_images(), "beach holiday", p)  # no shared words with caption
    assert [r.id for r, _ in res] == [a.id]
