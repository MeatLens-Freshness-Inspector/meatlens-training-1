"""
Organize public dataset 2 images into 3 folders by freshness label only.

Creates 3 folders under 'public dataset 2/organized/':
  fresh, half fresh, spoiled
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path


def main() -> None:
    dataset_root = Path(r"e:\Thesis Code\public dataset 2")
    output_root = dataset_root / "organized"

    # Clean up old organized folder
    if output_root.exists():
        shutil.rmtree(output_root)

    freshness_columns = {
        "FRESH": "fresh",
        "HALF-FRESH": "half fresh",
        "SPOILED": "spoiled",
    }

    for folder_name in freshness_columns.values():
        (output_root / folder_name).mkdir(parents=True, exist_ok=True)

    total_copied = 0
    total_skipped = 0
    total_errors = 0

    for split in ["train", "valid"]:
        split_dir = dataset_root / split
        csv_path = split_dir / "_classes.csv"

        if not csv_path.exists():
            print(f"[WARN] CSV not found: {csv_path}")
            continue

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row["filename"]
                image_path = split_dir / filename

                if not image_path.exists():
                    print(f"[WARN] Image not found: {image_path}")
                    total_errors += 1
                    continue

                active_freshness = [
                    col for col in freshness_columns if row.get(col, "0") == "1"
                ]

                if not active_freshness:
                    print(f"[SKIP] No freshness label for: {filename}")
                    total_skipped += 1
                    continue

                for freshness_col in active_freshness:
                    folder_name = freshness_columns[freshness_col]
                    dest = output_root / folder_name / filename
                    if not dest.exists():
                        shutil.copy2(image_path, dest)
                        total_copied += 1

    print(f"\nDone!")
    print(f"Total images copied: {total_copied}")
    print(f"Total skipped (no label): {total_skipped}")
    print(f"Total errors (missing file): {total_errors}")
    print(f"\nFolder contents:")
    for folder in sorted(output_root.iterdir()):
        if folder.is_dir():
            count = len(list(folder.glob("*.jpg")))
            print(f"  {folder.name}: {count} images")


if __name__ == "__main__":
    main()
