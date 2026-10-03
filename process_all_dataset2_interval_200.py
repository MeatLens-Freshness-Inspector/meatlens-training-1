"""
Apply HSV/LAB threshold segmentation (gray background) to all folders in
'public dataset 2'. For each subfolder (fresh, half fresh/not fresh, spoiled),
it selects exactly 200 images using interval sampling.
"""

from __future__ import annotations

import shutil
from pathlib import Path
import numpy as np
from PIL import Image

# Import the process_image function from the existing script
from apply_hsv_lab_threshold_roi_batch import process_image

def main() -> None:
    input_root = Path(r"e:\Thesis Code\public dataset 2")
    output_root = Path(r"e:\Thesis Code\public dataset 2 processed 200")

    # Folders we want to process
    target_datasets = ["Public_Dataset_1", "Public_Dataset_2", "sample 9", "sample 10"]

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    total_processed = 0

    for dataset_name in target_datasets:
        dataset_dir = input_root / dataset_name
        if not dataset_dir.exists():
            print(f"[WARN] Dataset not found: {dataset_dir}")
            continue

        output_dataset_dir = output_root / dataset_name

        # Iterate through subclasses (fresh, half fresh, spoiled, not fresh, etc.)
        for class_dir in dataset_dir.iterdir():
            if not class_dir.is_dir():
                continue

            output_class_dir = output_dataset_dir / class_dir.name
            output_class_dir.mkdir(parents=True, exist_ok=True)

            # Get all images and sort them alphabetically
            image_paths = sorted(
                p for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )

            total_images = len(image_paths)
            if total_images == 0:
                print(f"[WARN] No images in {class_dir}")
                continue

            print(f"\nProcessing '{dataset_name}/{class_dir.name}' ({total_images} total images)...")

            # Interval sampling to get exactly 200 images
            target_count = 200
            if total_images <= target_count:
                if total_images < target_count:
                    print(f"  [WARN] Only {total_images} images available, taking all.")
                sampled_indices = np.arange(total_images)
            else:
                sampled_indices = np.round(np.linspace(0, total_images - 1, target_count)).astype(int)

            sampled_paths = [image_paths[i] for i in sampled_indices]

            # Process each sampled image
            for i, img_path in enumerate(sampled_paths, start=1):
                try:
                    # Apply segmentation and background fill (gray)
                    output_uint8, metadata = process_image(img_path, background_mode="gray")
                    
                    # Save processed image
                    output_path = output_class_dir / img_path.name
                    Image.fromarray(output_uint8).save(output_path, quality=95)
                    total_processed += 1
                    
                    if i % 50 == 0 or i == len(sampled_paths):
                        print(f"  Processed {i}/{len(sampled_paths)} images")
                except Exception as e:
                    print(f"  [ERROR] Failed to process {img_path.name}: {e}")

    print(f"\nDone! Processed {total_processed} images in total.")
    print(f"Results saved to: {output_root}")

if __name__ == "__main__":
    main()
