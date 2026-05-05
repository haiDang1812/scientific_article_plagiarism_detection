from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embed_server.server import (  # noqa: E402
    BATCH_SIZE,
    _get_collection,
    _get_model,
    _upsert_batch,
    build_records,
    parse_pipeline_json,
)


def _select_source_type() -> str:
    while True:
        print("\nSelect input source:")
        print("  1) Single JSON file")
        print("  2) Folder of JSON files")
        choice = input("Choose 1 or 2: ").strip()

        if choice in {"1", "file", "f"}:
            return "file"
        if choice in {"2", "folder", "dir", "d"}:
            return "folder"

        print("Invalid choice. Please choose 1 or 2.")


def _collect_json_files() -> Tuple[List[Path], int]:
    source_type = _select_source_type()

    if source_type == "file":
        file_input = input("Enter JSON file path: ").strip().strip('"')
        file_path = Path(file_input).expanduser().resolve()

        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")
        if file_path.suffix.lower() != ".json":
            raise ValueError("Selected file must end with .json")

        return [file_path], 0

    dir_input = input("Enter folder path: ").strip().strip('"')
    folder = Path(dir_input).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Directory not found: {folder}")

    pattern_input = input("Glob pattern [*.json]: ").strip()
    pattern = pattern_input or "*.json"

    files = sorted(p for p in folder.glob(pattern) if p.is_file())
    if not files:
        raise FileNotFoundError(f"No files found in {folder} with pattern '{pattern}'")

    return files, 0


def run_interactive_upsert() -> None:
    print("=" * 60)
    print("EMBED JSON INTO CHROMA")
    print("=" * 60)

    files, _ = _collect_json_files()
    print(f"\nFound {len(files)} file(s). Building records...")

    all_records = []
    skipped = 0

    for json_file in files:
        record = parse_pipeline_json(json_file)
        if not record:
            skipped += 1
            print(f"[SKIP] No valid sections: {json_file}")
            continue

        all_records.extend(build_records(record))

    if not all_records:
        print("\nNo valid records to upsert. Nothing was written.")
        return

    model = _get_model()
    collection = _get_collection()
    upserted = _upsert_batch(collection, all_records, model, BATCH_SIZE)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Upserted records : {upserted}")
    print(f"Skipped files    : {skipped}")
    print(f"Collection total : {collection.count()}")


if __name__ == "__main__":
    run_interactive_upsert()
