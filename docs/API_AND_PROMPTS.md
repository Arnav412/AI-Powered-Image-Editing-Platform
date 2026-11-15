# How API selection and prompts work

## Pipeline for one edit

```
user text ──► [1] PLANNER (vision LLM sees image + request)
                 returns {edit_type, instruction, preserve}
          ──► [2] PROMPT BUILDER (template + type-specific hint + guard-rails)
          ──► [3] IMAGE MODEL (image + final prompt) ──► edited image
          ──► [4] STORE as new version (parent = version edited from)
          ──► [5] RE-CAPTION + re-embed so search reflects the edit
```

Why two model calls? Users say things like "remove the guy on the left" or "make it pop".
Image models do much better with a concrete instruction that names what is visible in
*this* image and says what must not change. The planner (a cheap vision LLM) does that
translation; the image model only has to execute. If the planner fails, the raw request
is used, so an edit is never blocked by the planning step.

## Provider selection

`src/ai/factory.py` picks the provider: **Gemini** if `GEMINI_API_KEY` is set,
**OpenAI** if `OPENAI_API_KEY` is set, or **OpenRouter** if `OPENROUTER_API_KEY` is set
(which also routes Gemini and OpenAI models seamlessly through OpenRouter), else **Demo mode**.
The sidebar lets the user switch.
Each provider implements only three primitives (`vision_text`, `edit_image`, `embed`);
captioning and planning are shared in `AIProvider`, so prompts behave the same everywhere.

| Job | Gemini | OpenAI |
|---|---|---|
| Caption / plan (image + text -> JSON) | `gemini-2.5-flash` | `gpt-5-mini` (chat vision) |
| Image edit | `gemini-2.5-flash-image` ("Nano Banana") | `gpt-image-1` via `images.edit` |
| Embeddings (search) | `gemini-embedding-001` | `text-embedding-3-small` |

Model names are environment variables (see `.env.example`) because vendors rename models
frequently; change them without touching code. Check each vendor's current model list
before your demo.

Trade-offs: Gemini's image model tends to be fast and good at preserving the rest of the
scene in conversational edits; OpenAI's `gpt-image-1` is strong at instruction following
and text-in-image. Having both behind one interface lets you compare them on the same photo.

## Prompt templates (`src/prompts.py`)

* **CAPTION_PROMPT** - forces JSON `{caption, tags}`, literal and specific wording, which is
  what makes casual queries like "ones with a beach" match.
* **PLANNER_PROMPT** - resolves vague references using visual anchors, limits the edit to
  one change, and lists what to preserve.
* **EDIT_TEMPLATE + TYPE_HINTS** - wraps the instruction with: preserve everything else,
  keep framing/resolution, match lighting and shadows, no text or watermarks, plus a hint per
  `edit_type` (removal = inpaint seamlessly; background = keep subject edges clean; recolour
  = keep texture; etc.).
* **PRESETS** - ten one-click edits (remove background, remove clutter/people, enhance, vivid
  colours, B&W, golden hour, blur background, cartoon, studio backdrop).

Try them without spending API credits: `python scripts/prompt_playground.py "your request"`.

## Version history

SQLite (`data/library.db`) + files (`data/images/<id>/vN.png` and `vN_thumb.png`).
Every edit adds a row with prompt, AI-refined prompt, provider, parent version and UTC
timestamp. Restoring an old version only moves `current_version_id`; the next edit branches
from it. Nothing is deleted except by the explicit "Delete image" action.

## Search (preview of Week 3)

Hybrid: cosine similarity on caption embeddings (when the provider offers embeddings) plus
stemmed keyword overlap. Keyword search alone works in Demo mode or if embedding calls fail.

## Known limitations

* Generative edits can drift (faces, small objects). The Before/After tab and version
  restore are the safety net.
* Stored images are capped at 2048 px on the long side to control cost and latency.
* Streamlit Cloud has an ephemeral disk: uploads reset on redeploy. Week 4 can swap
  `Store` for Supabase/S3 + Postgres; all storage goes through that one class.
