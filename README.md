# 🎨 PromptCanvas

**AI-First Natural Language Image Editing Platform with Semantic Vector Search**

Upload photos, describe edits in plain English, track an immutable version tree with thumbnails, and search your entire library using natural language (e.g. *"photos with a beach"*, *"dark moody portraits"*, *"sunset at the mountains"*).

**Status: Week 3 Complete** — Fast FAISS vector database integration, hybrid semantic + lexical search engine, batch embedding re-indexing, structured version tracking with lineage paths, and multi-provider AI gateway.

---

## 🌟 Key Features

### 1. Natural Language Image Search & Vector DB
* **FAISS Vector Store (`src/vector_store.py`)**: Sub-millisecond dense vector indexing using `IndexFlatIP` with L2 normalization (exact cosine similarity), with automatic JSON/NumPy fallback.
* **Hybrid Search Modes**:
  * **Hybrid (FAISS + Keyword)**: Combines dense vector similarity ($60\%$) with stemmed keyword matching ($40\%$).
  * **Semantic Only**: Pure vector embedding retrieval for conceptual and mood queries.
  * **Keyword Only**: Exact lexical matching across captions, AI-generated tags, and filenames.
* **Tag Filtering**: Dynamic tag multi-select pills derived from AI-generated image tags.
* **Batch Re-indexing**: One-click in-app or CLI tool to generate missing vector embeddings across the entire library.

### 2. Conversational Editing & Two-Stage Prompt Engine
* **Two-Stage Edit Pipeline**:
  * **Stage 1 (Vision Planner LLM)**: Grounded in visual anchors, isolates the target subject, and generates an invariant preservation whitelist.
  * **Stage 2 (Prompt Compiler)**: Injects lighting continuity, perspective, grain matching, and negative safety constraints.
* **10 Instant One-Click Presets**: *Remove background*, *Studio backdrop*, *Remove clutter*, *Remove people*, *Auto enhance*, *Vivid colours*, *Black & white*, *Golden hour*, *Blur background*, and *Cartoon style*.
* **Custom Natural Language Edits**: Free-form natural language instructions (*"add a vintage car parked on the gravel road"*).

### 3. Non-Destructive Version Tree & Audit Trail
* **Immutable Branching**: Edits never overwrite past iterations. Restoring an earlier version leaves history intact and branches forward.
* **Lineage Breadcrumbs**: Visual breadcrumbs showing the exact ancestry: `Original → v1 (Remove Background) → v2 (Golden Hour)`.
* **Before / After Diff**: Interactive comparison tab to verify subject fidelity and detect generative drift.
* **Telemetry**: Displays pixel dimensions ($W \times H$), file size (KB/MB), and generative model latency ($s$).

### 4. Multi-Provider AI Architecture
* **OpenRouter Gateway**: Seamlessly route both Google Gemini (`google/gemini-2.5-flash`) and OpenAI (`openai/gpt-4o-mini`) using a single OpenRouter key.
* **Google Gemini**: Direct integration via `google-genai` (Nano Banana image models).
* **OpenAI**: Direct integration via `openai` SDK (`gpt-4o-mini`, `gpt-image-1`).
* **Offline Demo Mode**: Zero-credit fallback running local PIL filters for local test environments.

---

## 🚀 Quickstart & Local Setup

### 1. Clone & Set Up Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and provide your API key:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# OpenRouter API Key (supports Google Gemini, OpenAI, and other models)
OPENROUTER_API_KEY=

# Or direct vendor keys (optional)
# GEMINI_API_KEY=...
# OPENAI_API_KEY=...
```

*Note: If no API key is set, the platform starts in offline **Demo Mode**.*

### 3. Launch the Web Application

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`.

---

## 🧪 Testing & Developer Tooling

### Run Automated Unit Tests

PromptCanvas includes 33 unit tests covering vector storage, SQLite branching, OpenRouter routing, search ranking, and prompt planning:

```bash
pytest -v
```

### Prompt Engineering Playground CLI

Inspect the prompt transformation pipeline without calling paid image generation models:

```bash
python scripts/prompt_playground.py "remove the background and light with golden hour"
```

### Batch Embedding Re-indexer CLI

Generate vector embeddings for all uploaded images in batch:

```bash
python scripts/reindex_embeddings.py
```

To force-refresh existing embeddings:

```bash
python scripts/reindex_embeddings.py --force
```

---

## 🏗️ Architecture & Project Layout

```
promptcanvas/
├── app.py                      # Streamlit UI (Library, Search, Detail/Edit, History)
├── requirements.txt            # Core dependencies (streamlit, faiss, pillow, openai, etc.)
├── .env.example                # Template configuration
├── src/
│   ├── exceptions.py           # Structured typed exceptions (Validation, Storage, NotFound)
│   ├── vector_store.py         # FAISS vector store with JSON/NumPy fallback
│   ├── storage.py              # SQLite (library.db) + atomic file storage + version tree
│   ├── search.py               # Hybrid semantic + lexical ranking engine
│   ├── services.py             # Business logic (ingest, edit, reindex, version restore)
│   ├── prompts.py              # Prompt templates, presets, and loose JSON parser
│   └── ai/
│       ├── base.py             # Abstract AIProvider interface (vision_text, edit_image, embed)
│       ├── factory.py          # Provider factory (OpenRouter, Gemini, OpenAI, Demo)
│       ├── openrouter_provider.py # OpenRouter unified API gateway
│       ├── gemini_provider.py  # Google Gemini SDK provider
│       ├── openai_provider.py  # OpenAI SDK provider
│       └── mock_provider.py    # Offline filter-based demo provider
├── scripts/
│   ├── prompt_playground.py    # Zero-cost CLI prompt inspection
│   └── reindex_embeddings.py   # Batch vector embedding generator
├── tests/
│   ├── test_core.py            # Core engine, storage, and prompt tests
│   ├── test_openrouter.py      # OpenRouter API client integration tests
│   └── test_vector_store.py    # FAISS vector store & version lineage tests
└── docs/
    ├── ARCHITECTURE.md         # Detailed system design, data flow & Mermaid diagrams
    ├── API_AND_PROMPTS.md      # API selection rationale & prompt engineering guide
    └── PRESENTATION_DECK.md    # 12-slide executive presentation outline
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for full architectural diagrams and data flow details.

---

## 🗺️ Roadmap

- [x] **Week 1**: Foundational storage, SQLite schema, image normalization, basic Streamlit UI.
- [x] **Week 2**: AI photo editing pipeline, Two-Stage Vision Planner, 10 presets, immutable version history.
- [x] **Week 3**: FAISS vector database, hybrid natural language search, batch re-indexing CLI, lineage tracking, and error handling.
- [ ] **Week 4**: Cloud storage migration (AWS S3 / Cloudflare R2), multi-tenant user authentication, and Streamlit Cloud production deployment.
