"""Reset app-owned metadata, originals, and known DocuzRAG Qdrant collections."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from core.config import PROJECT_ROOT, get_config
from qdrant_client import QdrantClient


def _remove_contents(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for path in folder.iterdir():
        if path.name == ".gitkeep":
            continue
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm-delete",
        action="store_true",
        required=True,
        help="Required because this removes application-owned data.",
    )
    parser.add_argument(
        "--delete-local-qdrant",
        action="store_true",
        help="Remove the repository-local Qdrant storage directory too.",
    )
    args = parser.parse_args()
    config = get_config()

    client = QdrantClient(
        url=config.qdrant_url,
        api_key=config.qdrant_api_key,
        check_compatibility=False,
        timeout=30,
    )
    known_application_collections = {
        config.qdrant_collection,
        "docuzrag",
        "docurag",
        "docurag_text",
        "docurag_images",
    }
    existing = {item.name for item in client.get_collections().collections}
    for name in sorted(existing & known_application_collections):
        client.delete_collection(name)
        print(f"Removed Qdrant collection {name}.")

    data_root = (PROJECT_ROOT / "data").resolve()
    for folder in (
        config.document_store_path.resolve(),
        config.processed_store_path.resolve(),
        config.index_store_path.resolve(),
    ):
        folder.relative_to(data_root)
        _remove_contents(folder)

    database_path = config.metadata_db_path.resolve()
    database_path.relative_to(data_root)
    if database_path.exists():
        database_path.unlink()
        print("Removed SQLite metadata database.")

    if args.delete_local_qdrant:
        local_qdrant = (data_root / "qdrant").resolve()
        local_qdrant.relative_to(data_root)
        if local_qdrant.exists():
            shutil.rmtree(local_qdrant)
            print("Removed repository-local Qdrant storage.")

    print("DocuzRAG application data reset complete.")


if __name__ == "__main__":
    main()
