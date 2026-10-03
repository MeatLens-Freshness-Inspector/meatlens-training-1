from __future__ import annotations

import os
from pathlib import Path
from PIL import Image

def main():
    target_dir = Path(r"E:\Thesis Code\public dataset 2 processed 200\sample 12\spoiled")
    
    if not target_dir.exists():
        print(f"Directory does not exist: {target_dir}")
        return

    fixed_count = 0
    total_count = 0

    for file_path in target_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            total_count += 1
            try:
                img = Image.open(file_path).convert("RGB")
                if img.size != (224, 224):
                    # Create a 224x224 gray background (127, 127, 127)
                    new_img = Image.new("RGB", (224, 224), (127, 127, 127))
                    
                    # Calculate center coordinates for pasting
                    # Wait, if the cropped image is LARGER than 224x224, we need to handle that.
                    # Usually cropped images are smaller.
                    w, h = img.size
                    
                    if w > 224 or h > 224:
                        print(f"[WARN] {file_path.name} is larger than 224x224 ({w}x{h}). Resizing it first.")
                        # Optionally resize it, but let's just resize maintaining aspect ratio
                        img.thumbnail((224, 224), Image.Resampling.LANCZOS)
                        w, h = img.size

                    x = (224 - w) // 2
                    y = (224 - h) // 2
                    
                    new_img.paste(img, (x, y))
                    
                    # Overwrite the original image
                    new_img.save(file_path, quality=95)
                    print(f"Fixed {file_path.name} (original size: {w}x{h})")
                    fixed_count += 1
            except Exception as e:
                print(f"[ERROR] Failed to process {file_path.name}: {e}")

    print(f"\nDone! Checked {total_count} images, fixed {fixed_count} images.")

if __name__ == "__main__":
    main()
