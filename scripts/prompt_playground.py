"""Prompt-handling playground.

Shows exactly what each stage sends to the AI APIs, without calling them.

    python scripts/prompt_playground.py
    python scripts/prompt_playground.py "remove the person on the left"

With an API key set (GEMINI_API_KEY or OPENAI_API_KEY) and an image path, it
also runs the real planner stage:

    python scripts/prompt_playground.py "add a tree in the background" photo.jpg
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import prompts  # noqa: E402


def banner(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    request = sys.argv[1] if len(sys.argv) > 1 else "add a tree in the background"
    image_path = sys.argv[2] if len(sys.argv) > 2 else None

    banner("1) CAPTION PROMPT (vision LLM -> JSON for search)")
    print(prompts.CAPTION_PROMPT)

    banner(f"2) PLANNER PROMPT for request: {request!r}")
    print(prompts.build_planner_prompt(request))

    if image_path:
        from src.ai import get_provider

        provider = get_provider("auto")
        banner(f"   Live planner output via provider: {provider.name}")
        plan = provider.plan_edit(Path(image_path).read_bytes(), request)
        print(f"edit_type:   {plan.edit_type}\ninstruction: {plan.instruction}")
        final = plan.final_prompt
    else:
        final = prompts.build_edit_prompt("add", request, "all existing objects, lighting, framing")

    banner("3) FINAL EDIT PROMPT (sent to the image model)")
    print(final)

    banner("PRESETS (one-click edits)")
    for label, (etype, instruction) in prompts.PRESETS.items():
        print(f"- {label:<20} [{etype}] {instruction}")


if __name__ == "__main__":
    main()
