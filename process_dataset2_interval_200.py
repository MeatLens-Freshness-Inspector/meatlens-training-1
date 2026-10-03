"""
Select 200 images per folder using interval sampling, apply HSV/LAB threshold
segmentation with gray background, and save them to a new output directory.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from PIL import Image

# Import the process_image function from the existing script
from apply_hsv_lab_threshold_roi_batch import process_image

def main() -> None:
    input_root = Path(r"e:\Thesis Code\public dataset 2\organized")
    output_root = Path(r"e:\Thesis Code\public dataset 2\processed_organized_200")

    # Clean up output directory if it exists
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    folders = ["fresh", "half fresh", "spoiled"]

    for folder_name in folders:
        input_folder = input_root / folder_name
        output_folder = output_root / folder_name
        output_folder.mkdir(parents=True, exist_ok=True)

        if not input_folder.exists():
            print(f"[WARN] Folder not found: {input_folder}")
            continue

        # Get all images and sort them alphabetically
        image_paths = sorted(
            p for p in input_folder.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        total_images = len(image_paths)
        if total_images == 0:
            print(f"[WARN] No images in {input_folder}")
            continue

        print(f"Processing folder '{folder_name}' ({total_images} total images)...")

        # Interval sampling to get exactly 200 images
        target_count = 200
        if total_images < target_count:
            print(f"[WARN] Only {total_images} images available, taking all.")
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
                output_path = output_folder / img_path.name
                Image.fromarray(output_uint8).save(output_path, quality=95)
                
                if i % 50 == 0 or i == len(sampled_paths):
                    print(f"  Processed {i}/{len(sampled_paths)} images")
            except Exception as e:
                print(f"  [ERROR] Failed to process {img_path.name}: {e}")

    print("\nDone! All folders processed.")
    print("Results saved to:", output_root)

if __name__ == "__main__":
    main()
