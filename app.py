"""PromptCanvas - Professional Creative Image Studio & Vector Search.

Run:  streamlit run app.py
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import streamlit as st

# Streamlit Cloud: expose st.secrets as env vars so providers can read them.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:
    pass

from src import prompts, search, services  # noqa: E402
from src.ai import ProviderError, available_providers, get_provider  # noqa: E402
from src.exceptions import PromptCanvasError  # noqa: E402
from src.storage import ImageRecord, Store, VersionRecord  # noqa: E402

st.set_page_config(
    page_title="PromptCanvas Studio",
    page_icon="☷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------- styling
# Clean, professional studio design system (Charcoal / Slate / Titanium / Apple Pro aesthetic)
# ZERO purple/violet generic AI gradients.
st.markdown(
    """
    <style>
      /* Base & Typography */
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
      
      html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0f172a;
      }
      .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
      }

      /* Clean Navigation & Header */
      .pc-topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 1rem;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
      }
      .pc-title {
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        color: #0f172a;
        display: flex;
        align-items: center;
        gap: 0.5rem;
      }
      .pc-subtitle {
        font-size: 0.875rem;
        color: #64748b;
        margin-top: 2px;
      }

      /* Metric Cards (Stat Bar) */
      .pc-stat-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
      }
      .pc-stat-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
      }
      .pc-stat-val {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 2px;
      }

      /* Card Grid & Image Container */
      .pc-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.75rem;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
        margin-bottom: 1.25rem;
      }
      .pc-card:hover {
        border-color: #cbd5e1;
        box-shadow: 0 4px 12px -2px rgba(15, 23, 42, 0.08);
      }
      .pc-card-caption {
        font-size: 0.8125rem;
        color: #334155;
        line-height: 1.4;
        font-weight: 500;
        min-height: 2.8em;
        margin-top: 0.5rem;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
      }

      /* Pills & Badges - Neutral Charcoal & Precision Slate (NO AI PURPLE) */
      .pc-pill {
        display: inline-flex;
        align-items: center;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.725rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        margin-right: 4px;
        margin-top: 4px;
      }
      .pc-pill-score {
        background: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
      }
      .pc-pill-version {
        background: #f1f5f9;
        color: #334155;
        border: 1px solid #e2e8f0;
      }
      .pc-pill-tag {
        background: #f8fafc;
        color: #475569;
        border: 1px solid #e2e8f0;
        font-weight: 500;
      }
      .pc-pill-active {
        background: #0f172a;
        color: #f8fafc;
        border: 1px solid #0f172a;
      }

      /* Studio Inspector & Breadcrumb Strip */
      .pc-breadcrumbs {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.85rem;
        color: #64748b;
        margin-bottom: 1rem;
        padding: 0.5rem 0.75rem;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
      }
      .pc-crumb-current {
        font-weight: 600;
        color: #0f172a;
      }

      /* Version Timeline Strip */
      .pc-vnode {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.6rem;
        margin-bottom: 0.5rem;
      }
      .pc-vnode-active {
        border: 2px solid #0f172a;
        box-shadow: 0 0 0 1px #0f172a;
      }
      .pc-vnode-meta {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 2px;
      }

      /* Quick Modifiers / Suggestion Chips */
      .pc-prompt-chip {
        display: inline-block;
        padding: 4px 10px;
        margin: 2px 4px 6px 0;
        border-radius: 16px;
        background: #f1f5f9;
        color: #1e293b;
        font-size: 0.75rem;
        font-weight: 500;
        border: 1px solid #cbd5e1;
        cursor: pointer;
      }

      /* Streamlit Element Overrides */
      div.stButton > button[kind="primary"] {
        background-color: #0f172a !important;
        border-color: #0f172a !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em !important;
        border-radius: 6px !important;
      }
      div.stButton > button[kind="primary"]:hover {
        background-color: #1e293b !important;
        border-color: #1e293b !important;
      }
      div.stButton > button[kind="secondary"] {
        border-radius: 6px !important;
        font-weight: 500 !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- resources
@st.cache_resource
def get_store() -> Store:
    return Store(os.getenv("DATA_DIR", "data"))


@st.cache_resource
def cached_provider(name: str):
    return get_provider(name)


store = get_store()
ss = st.session_state
ss.setdefault("view", "library")
ss.setdefault("selected", None)
ss.setdefault("uploader_key", 0)
ss.setdefault("grid_cols", 4)
ss.setdefault("prompt_buffer", "")


def go(view: str, image_id: str | None = None) -> None:
    ss.view, ss.selected = view, image_id
    st.rerun()


def fmt_time(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%d %b %Y, %H:%M UTC")
    except ValueError:
        return iso


def fmt_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.1f} MB"


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### ☷ **PromptCanvas Studio**")
    st.caption("AI Creative Studio with Vector Search")

    providers = available_providers()
    choice = st.selectbox(
        "Active AI Engine",
        list(providers.keys()),
        format_func=lambda k: providers[k],
        help="Switch provider: OpenRouter, Gemini, OpenAI, or offline Demo.",
    )
    try:
        provider = cached_provider(choice)
    except ProviderError as e:
        st.error(str(e))
        st.stop()

    if provider.name == "demo":
        st.info("Demo Mode Active: Edits run instant local PIL filters. Add an API key in .env for generative models.")

    st.divider()

    # System status metrics
    v_count = store.vector_store.count()
    v_backend = store.vector_store.backend.upper()
    total_imgs = len(store.list_images())

    st.markdown(f"**Index Status**: `{v_backend}` ({v_count}/{total_imgs} embedded)")

    if st.button("⚡ Sync & Re-index Embeddings", use_container_width=True,
                 help="Scans library and generates vector search embeddings for any missing images."):
        with st.spinner("Generating vector embeddings..."):
            prog_bar = st.progress(0.0)
            def on_p(done: int, total: int):
                prog_bar.progress(done / total if total else 1.0)
            indexed = services.batch_reindex_embeddings(store, provider, on_progress=on_p)
        st.toast(f"Vector search indexed {indexed} image(s)", icon="⚡")
        st.rerun()

    if ss.view != "library":
        if st.button("☷ Back to Library", use_container_width=True):
            go("library")

    st.divider()
    st.subheader("Import Photos")
    files = st.file_uploader(
        "Upload images to canvas",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key=f"uploader_{ss.uploader_key}",
        label_visibility="collapsed",
    )
    if files and st.button(f"Import {len(files)} Asset(s)", type="primary", use_container_width=True):
        bar = st.progress(0.0, text="Analyzing and indexing...")
        last_id = None
        for i, f in enumerate(files):
            try:
                rec = services.ingest_upload(store, provider, f.getvalue(), f.name)
                last_id = rec.id
            except Exception as e:
                st.warning(f"{f.name}: {e}")
            bar.progress((i + 1) / len(files), text=f"Processed {i + 1}/{len(files)}")
        ss.uploader_key += 1
        if last_id and len(files) == 1:
            go("detail", last_id)
        st.rerun()


# ------------------------------------------------------------------ library
def render_library() -> None:
    records = store.list_images()

    # Stat strip
    c_s1, c_s2, c_s3, c_s4 = st.columns(4)
    with c_s1:
        st.markdown(
            f"<div class='pc-stat-card'><div class='pc-stat-label'>Total Assets</div><div class='pc-stat-val'>{len(records)}</div></div>",
            unsafe_allow_html=True,
        )
    with c_s2:
        st.markdown(
            f"<div class='pc-stat-card'><div class='pc-stat-label'>Indexed Vectors</div><div class='pc-stat-val'>{store.vector_store.count()}</div></div>",
            unsafe_allow_html=True,
        )
    with c_s3:
        all_tags = sorted({t for r in records for t in r.tags if t})
        st.markdown(
            f"<div class='pc-stat-card'><div class='pc-stat-label'>Unique Tags</div><div class='pc-stat-val'>{len(all_tags)}</div></div>",
            unsafe_allow_html=True,
        )
    with c_s4:
        total_versions = sum(r.version_count for r in records)
        st.markdown(
            f"<div class='pc-stat-card'><div class='pc-stat-label'>Total Edits</div><div class='pc-stat-val'>{total_versions - len(records)}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    if not records:
        st.info("Your library is currently empty. Upload photos from the sidebar to start creating.")
        return

    # Interactive Search Deck
    c_search, c_mode, c_view = st.columns([3, 1.2, 0.8])
    with c_search:
        query = st.text_input(
            "Natural Language Search",
            placeholder="Search by description or mood (e.g. 'golden hour portrait', 'beach sunset', 'vintage car')",
            label_visibility="collapsed",
        )
    with c_mode:
        search_mode = st.selectbox(
            "Search Engine",
            ["Hybrid (FAISS + Keyword)", "Semantic Only (FAISS)", "Keyword Only (Lexical)"],
            index=0,
            label_visibility="collapsed",
        )
    with c_view:
        grid_cols_choice = st.selectbox(
            "Grid Layout",
            ["4 Columns", "3 Columns", "6 Columns"],
            index=0,
            label_visibility="collapsed",
        )
        col_map = {"4 Columns": 4, "3 Columns": 3, "6 Columns": 6}
        cols_per_row = col_map[grid_cols_choice]

    mode_map = {
        "Hybrid (FAISS + Keyword)": "hybrid",
        "Semantic Only (FAISS)": "semantic",
        "Keyword Only (Lexical)": "lexical",
    }
    active_mode = mode_map[search_mode]

    # Interactive Tag Chips (Filter Deck)
    selected_tags = []
    if all_tags:
        selected_tags = st.multiselect(
            "Filter by tag",
            all_tags,
            placeholder="Filter by tags (e.g. beach, person, sunset, car)...",
            label_visibility="collapsed",
        )

    # Search Execution
    scores_by_id: dict[str, float] = {}
    if query.strip() or selected_tags:
        results = search.search(
            records,
            query,
            provider,
            store=store,
            mode=active_mode,
            tag_filter=selected_tags if selected_tags else None,
        )
        scores_by_id = {r.id: score for r, score in results}
        shown = [r for r, _ in results]
        st.caption(f"Showing **{len(shown)}** matched result(s) in library")
        if not shown:
            st.warning("No matches found for this query. Try broader keywords or clear the tag filters.")
            return
    else:
        shown = records
        st.caption(f"Showing **{len(records)}** asset(s) in library")

    # Interactive Grid View
    for start in range(0, len(shown), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, rec in zip(cols, shown[start : start + cols_per_row]):
            with col:
                cur = store.get_version(rec.current_version_id)
                thumb_bytes = store.read_thumb_bytes(cur)

                st.image(thumb_bytes, use_container_width=True)

                badges = []
                if rec.id in scores_by_id and scores_by_id[rec.id] > 0:
                    score_pct = int(scores_by_id[rec.id] * 100)
                    badges.append(f"<span class='pc-pill pc-pill-score'>{score_pct}% Match</span>")
                if rec.version_count > 1:
                    badges.append(f"<span class='pc-pill pc-pill-version'>{rec.version_count - 1} edit(s)</span>")
                if rec.width and rec.height:
                    badges.append(f"<span class='pc-pill pc-pill-tag'>{rec.width}×{rec.height}</span>")

                badge_html = "".join(badges)
                caption_text = rec.caption if rec.caption else rec.name

                st.markdown(
                    f"<div class='pc-card-caption' title='{caption_text}'>{caption_text}</div>{badge_html}",
                    unsafe_allow_html=True,
                )

                st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
                if st.button("Open Studio →", key=f"open_{rec.id}", use_container_width=True):
                    go("detail", rec.id)


# ------------------------------------------------------------------- detail / studio
def run_edit(fn, *args, **kwargs) -> None:
    refresh = ss.get("refresh_caption", True)
    with st.spinner("Processing edit with AI model..."):
        try:
            fn(store, provider, *args, refresh_caption=refresh, **kwargs)
        except (PromptCanvasError, ProviderError, ValueError) as e:
            st.error(f"Edit failed: {e}")
            return
    st.toast("Edit applied successfully! New version saved.", icon="✓")
    st.rerun()


def render_detail(image_id: str) -> None:
    img = store.get_image(image_id)
    if img is None:
        st.error("Image not found.")
        if st.button("← Return to Library"):
            go("library")
        return

    versions = store.list_versions(image_id)
    current = store.get_version(img.current_version_id)
    original = versions[0]
    current_bytes = store.read_version_bytes(current)

    # Lineage breadcrumbs
    lineage = store.get_lineage(current.id)
    lineage_parts = []
    for v in lineage:
        name = "Original" if v.kind == "original" else f"v{v.version_no}"
        if v.id == current.id:
            lineage_parts.append(f"<span class='pc-crumb-current'>{name} (Active)</span>")
        else:
            lineage_parts.append(name)
    lineage_html = " &nbsp;›&nbsp; ".join(lineage_parts)

    # Top Bar Header
    c_top_l, c_top_r = st.columns([5, 1])
    with c_top_l:
        st.markdown(
            f"<div class='pc-breadcrumbs'><strong>Asset:</strong> {img.name} &nbsp;|&nbsp; <strong>Lineage:</strong> {lineage_html}</div>",
            unsafe_allow_html=True,
        )
    with c_top_r:
        if st.button("☷ Library", use_container_width=True):
            go("library")

    # Main Studio Layout (Left Canvas, Right Inspector & Control Deck)
    left_canvas, right_deck = st.columns([3, 2], gap="large")

    # ========================== Left Canvas ==========================
    with left_canvas:
        tab_preview, tab_compare, tab_info = st.tabs(["Canvas Preview", "Interactive Compare", "Metadata & Tags"])

        with tab_preview:
            st.image(current_bytes, use_container_width=True)
            dims = f"{current.width}×{current.height} px" if current.width else ""
            fsize = fmt_size(current.file_size) if current.file_size else ""
            dur_str = f" · Latency: {current.duration_ms / 1000:.1f}s" if current.duration_ms > 0 else ""
            st.caption(f"**Version {current.version_no}** · {dims} · {fsize} · {fmt_time(current.created_at)}{dur_str}")

        with tab_compare:
            st.markdown("**Before vs. After Inspection**")
            diff_mode = st.radio(
                "Compare Mode",
                ["Side-by-Side", "Toggle (Original / Current)"],
                horizontal=True,
                label_visibility="collapsed",
            )
            if diff_mode == "Side-by-Side":
                c1, c2 = st.columns(2)
                with c1:
                    st.caption("Original Asset (v0)")
                    st.image(store.read_version_bytes(original), use_container_width=True)
                with c2:
                    st.caption(f"Current Iteration (v{current.version_no})")
                    st.image(current_bytes, use_container_width=True)
            else:
                view_toggle = st.segmented_control(
                    "View",
                    options=["Original (v0)", f"Current (v{current.version_no})"],
                    default=f"Current (v{current.version_no})",
                )
                if view_toggle == "Original (v0)":
                    st.image(store.read_version_bytes(original), use_container_width=True)
                else:
                    st.image(current_bytes, use_container_width=True)

        with tab_info:
            if img.caption:
                st.markdown(f"**AI Scene Description:**\n{img.caption}")
                tag_pills = "".join(f"<span class='pc-pill pc-pill-tag'>#{t}</span>" for t in img.tags)
                st.markdown(f"**Extracted Tags:**<br>{tag_pills}", unsafe_allow_html=True)
            else:
                st.caption("No caption generated yet.")

            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            if st.button("↻ Re-analyze Scene & Vector Embeddings", use_container_width=True):
                with st.spinner("Analyzing visual scene and vectorizing..."):
                    try:
                        services.analyze_image(store, provider, image_id)
                    except (PromptCanvasError, ProviderError) as e:
                        st.error(str(e))
                st.rerun()

        # Download button
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        st.download_button(
            "⬇ Export Current PNG",
            current_bytes,
            file_name=f"{img.name.rsplit('.', 1)[0]}_v{current.version_no}.png",
            mime="image/png",
            use_container_width=True,
        )

    # ========================== Right Deck ==========================
    with right_deck:
        st.markdown("### Creative Controls")

        ss.setdefault("refresh_caption", True)
        st.checkbox(
            "Update search index after edit",
            key="refresh_caption",
            help="Automatically updates the vector embedding and caption so natural language search reflects the edit.",
        )

        # Categorized Presets Deck
        st.markdown("**Studio Presets**")
        cat_tabs = st.tabs(["Background", "Retouch", "Lighting & Mood", "Artistic"])

        with cat_tabs[0]:
            st.caption("Isolate, swap, or blur backgrounds")
            c_bg1, c_bg2, c_bg3 = st.columns(3)
            if c_bg1.button("Clean White", use_container_width=True, help="Replaces background with clean white studio surface"):
                run_edit(services.apply_preset, image_id, "Remove background")
            if c_bg2.button("Grey Studio", use_container_width=True, help="Soft neutral grey studio backdrop"):
                run_edit(services.apply_preset, image_id, "Studio backdrop")
            if c_bg3.button("Blur Depth", use_container_width=True, help="Shallow depth-of-field realistic background blur"):
                run_edit(services.apply_preset, image_id, "Blur background")

        with cat_tabs[1]:
            st.caption("Remove clutter and unwanted subjects")
            c_rt1, c_rt2, c_rt3 = st.columns(3)
            if c_rt1.button("Remove People", use_container_width=True, help="Removes pedestrians or people from scene"):
                run_edit(services.apply_preset, image_id, "Remove people")
            if c_rt2.button("Remove Clutter", use_container_width=True, help="Removes distracting objects"):
                run_edit(services.apply_preset, image_id, "Remove all clutter")
            if c_rt3.button("Auto Enhance", use_container_width=True, help="Balances exposure, improves contrast"):
                run_edit(services.apply_preset, image_id, "Auto enhance")

        with cat_tabs[2]:
            st.caption("Adjust color grading and natural lighting")
            c_lt1, c_lt2, c_lt3 = st.columns(3)
            if c_lt1.button("Golden Hour", use_container_width=True, help="Warm sunset sunlight and golden grade"):
                run_edit(services.apply_preset, image_id, "Golden hour")
            if c_lt2.button("Vivid Tones", use_container_width=True, help="Rich, punchy colors without over-saturation"):
                run_edit(services.apply_preset, image_id, "Vivid colours")
            if c_lt3.button("B&W Film", use_container_width=True, help="High-contrast monochromatic photograph"):
                run_edit(services.apply_preset, image_id, "Black & white")

        with cat_tabs[3]:
            st.caption("Stylistic illustrations and transforms")
            c_ar1, _ = st.columns(2)
            if c_ar1.button("Cartoon Style", use_container_width=True, help="Redraws image as a clean cartoon illustration"):
                run_edit(services.apply_preset, image_id, "Cartoon style")

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.markdown("**Custom Conversational Prompt**")

        # Interactive Prompt Helper Chips
        st.caption("Click to insert visual modification modifier:")
        col_c1, col_c2, col_c3 = st.columns(3)
        if col_c1.button("+ Warm Lighting", key="chip_warm"):
            ss.prompt_buffer = (ss.prompt_buffer + " Add warm golden sunlight").strip()
        if col_c2.button("+ Clean Background", key="chip_clean"):
            ss.prompt_buffer = (ss.prompt_buffer + " Clean and declutter background").strip()
        if col_c3.button("+ Depth Blur", key="chip_blur"):
            ss.prompt_buffer = (ss.prompt_buffer + " Apply soft background blur").strip()

        with st.form("custom_edit_form", clear_on_submit=False):
            custom_prompt = st.text_area(
                "Describe exact change",
                value=ss.prompt_buffer,
                placeholder="Example: 'Remove the pedestrian on the left', 'Add a retro bicycle beside the wall'...",
                height=90,
                label_visibility="collapsed",
            )
            c_btn1, c_btn2 = st.columns([3, 1])
            with c_btn1:
                submitted = st.form_submit_button("Execute Edit →", type="primary", use_container_width=True)
            with c_btn2:
                cleared = st.form_submit_button("Clear", use_container_width=True)

        if cleared:
            ss.prompt_buffer = ""
            st.rerun()

        if submitted:
            if not custom_prompt.strip():
                st.warning("Please enter a description of the change first.")
            else:
                ss.prompt_buffer = ""
                run_edit(services.apply_edit, image_id, custom_prompt)

    # ========================== Version Timeline ==========================
    st.divider()
    st.markdown(f"### Version History ({len(versions)} Iterations)")
    st.caption("Non-destructive version tree: every edit creates a new immutable node. Restoring any earlier iteration branches forward.")

    timeline_cols = st.columns(min(len(versions), 5))
    for idx, v in enumerate(versions):
        col_idx = idx % min(len(versions), 5)
        with timeline_cols[col_idx]:
            is_active = (v.id == current.id)
            thumb_b = store.read_thumb_bytes(v)
            st.image(thumb_b, use_container_width=True)

            v_title = "Original (v0)" if v.kind == "original" else f"Version {v.version_no}"
            active_badge = " · <span class='pc-pill pc-pill-score'>Active</span>" if is_active else ""
            st.markdown(f"**{v_title}**{active_badge}", unsafe_allow_html=True)

            v_dur = f" · {v.duration_ms / 1000:.1f}s" if v.duration_ms > 0 else ""
            st.caption(f"{fmt_time(v.created_at)}{v_dur}")

            if v.prompt:
                st.caption(f"“{v.prompt[:70]}{'…' if len(v.prompt) > 70 else ''}”")
                if v.refined_prompt and v.refined_prompt != v.prompt:
                    with st.expander("Planner prompt"):
                        st.write(v.refined_prompt)
                        st.caption(f"Engine: {v.provider}")

            if not is_active:
                if st.button("Restore Version", key=f"rst_{v.id}", use_container_width=True):
                    services.restore_version(
                        store,
                        provider,
                        image_id,
                        v.id,
                        refresh_caption=ss.get("refresh_caption", True),
                    )
                    st.rerun()

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    with st.expander("Asset Maintenance & Danger Zone"):
        confirm = st.checkbox("Confirm permanent deletion of this asset and all versions", key=f"del_{image_id}")
        if st.button("Permanently Delete Asset", disabled=not confirm):
            store.delete_image(image_id)
            go("library")


# ------------------------------------------------------------------- router
if ss.view == "detail" and ss.selected:
    render_detail(ss.selected)
else:
    render_library()
