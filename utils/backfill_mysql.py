from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

from utils.db_store import BackupDbStore


def backfill_from_state_path(db: BackupDbStore, state_path: Path) -> tuple[int, int]:
    if not state_path.exists():
        return (0, 0)

    with state_path.open("r", encoding="utf-8") as f:
        state = json.load(f)

    stored = state.get("stored_files", {})
    versions = state.get("file_versions", {})

    upserted_files = 0
    inserted_versions = 0

    for file_hash, file_path in stored.items():
        path_obj = Path(file_path)
        base_name = path_obj.name
        inferred_name = base_name.split("_", 1)[1] if "_" in base_name else base_name

        record = versions.get(file_hash, {})
        file_name = record.get("file_name", inferred_name)

        file_id = db.upsert_file(file_name=file_name, file_hash=file_hash)
        if file_id is None:
            continue
        upserted_files += 1

        version_list = record.get("versions", [])
        if not version_list:
            file_size = path_obj.stat().st_size if path_obj.exists() else 0
            db.add_file_version(
                file_id=file_id,
                version_number=1,
                file_path=str(path_obj),
                file_size=file_size,
            )
            db.log_action(file_id=file_id, action="backfill_upload")
            inserted_versions += 1
            continue

        for item in version_list:
            version_number = int(item.get("version_number", 1))
            version_path = Path(str(item.get("file_path", file_path)))
            file_size = version_path.stat().st_size if version_path.exists() else 0
            db.add_file_version(
                file_id=file_id,
                version_number=version_number,
                file_path=str(version_path),
                file_size=file_size,
            )
            action = str(item.get("event", "upload"))
            db.log_action(file_id=file_id, action=f"backfill_{action}")
            inserted_versions += 1

    return (upserted_files, inserted_versions)


def main() -> None:
    load_dotenv()

    state_path = Path(".agent_data/app_state.json")
    if not state_path.exists():
        print("backfill_skipped reason=state_file_missing")
        return

    db = BackupDbStore.from_env()
    if not db.enabled:
        print("backfill_skipped reason=db_not_enabled")
        return

    db.ensure_schema()
    upserted_files, inserted_versions = backfill_from_state_path(db, state_path)

    print(f"backfill_done files={upserted_files} versions={inserted_versions}")


if __name__ == "__main__":
    main()
