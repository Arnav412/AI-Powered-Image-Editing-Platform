"""CLI script to batch generate search embeddings for all images in the library.

Usage:
    python scripts/reindex_embeddings.py
    python scripts/reindex_embeddings.py --force
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ai import get_provider
from src.services import batch_reindex_embeddings
from src.storage import Store


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate search embeddings for image library.")
    parser.add_argument("--force", action="store_true", help="Re-generate embeddings even if they already exist.")
    parser.add_argument("--data-dir", default="data", help="Path to data directory (default: data).")
    args = parser.parse_args()

    store = Store(args.data_dir)
    provider = get_provider("auto")
    print(f"Using AI provider: {provider.name} ({provider.label})")
    print(f"Provider supports embeddings: {provider.supports_embeddings}")

    images = store.list_images()
    print(f"Found {len(images)} total image(s) in {args.data_dir}.")

    def progress(done: int, total: int):
        pct = (done / total) * 100 if total else 100
        print(f"\rProgress: {done}/{total} ({pct:.1f}%)", end="", flush=True)

    reindexed = batch_reindex_embeddings(store, provider, force=args.force, on_progress=progress)
    print(f"\nCompleted! Generated embeddings for {reindexed} image(s).")
    print(f"Vector store now contains {store.vector_store.count()} indexed embeddings.")


if __name__ == "__main__":
    main()
