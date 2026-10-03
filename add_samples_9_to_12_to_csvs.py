import pandas as pd
from pathlib import Path

def main():
    processed_dir = Path(r"E:\Thesis Code\processed_hsv_lab_threshold_roi_224")
    splits_dir = Path(r"E:\Thesis Code\generated_splits\cross_rotation_interval200_8samples_processed_roi")
    
    samples = [
        {
            "num": 9,
            "meat_part": "Pork Hind Leg",
            "pork_cut": "hind_leg",
            "id": "pork_hind_leg_sample_9",
            "has_time": True
        },
        {
            "num": 10,
            "meat_part": "Pork Belly",
            "pork_cut": "belly",
            "id": "pork_belly_sample_10",
            "has_time": True
        },
        {
            "num": 11,
            "meat_part": "Various Pork Meat",
            "pork_cut": "various",
            "id": "various_pork_meat_sample_11",
            "has_time": False
        },
        {
            "num": 12,
            "meat_part": "Various Pork Meat",
            "pork_cut": "various",
            "id": "various_pork_meat_sample_12",
            "has_time": False
        }
    ]
    
    new_rows = []
    
    for s in samples:
        s_dir = processed_dir / f"sample {s['num']}"
        if not s_dir.exists():
            print(f"[WARN] {s_dir} does not exist. Skipping.")
            continue
            
        for class_dir in s_dir.iterdir():
            if not class_dir.is_dir():
                continue
                
            orig_label = class_dir.name.lower()
            if orig_label == "half fresh":
                label = "not fresh"
            else:
                label = orig_label
                
            time_frame = ""
            if s["has_time"]:
                if label == "fresh":
                    time_frame = "04:00:00"  # 0 to 8 hours
                elif label == "not fresh":
                    time_frame = "12:00:00"  # 8 to 16 hours
                elif label == "spoiled":
                    time_frame = "24:00:00"  # 16 to 32 hours
                    
            for img_path in class_dir.glob("*.jpg"):
                dest = str(img_path)
                filename = img_path.name
                
                row = {
                    "image_file_name": filename,
                    "sample_number": str(s["num"]),
                    "meat_part": s["meat_part"],
                    "label": label,
                    "time_frame": time_frame,
                    "file_destination": dest,
                    "sample_id": s["id"],
                    "pork_cut": s["pork_cut"],
                    "split_type": "cross_rotation",
                    # fold and source will be populated
                }
                new_rows.append(row)

    print(f"Gathered {len(new_rows)} new images from sample 9 to 12.")
    if len(new_rows) == 0:
        return

    # 1. Append to processing_summary.csv
    summary_path = processed_dir / "processing_summary.csv"
    if summary_path.exists():
        summary_df = pd.read_csv(summary_path, dtype=str)
        existing_files = set(summary_df["image_file_name"])
        summary_additions = []
        for r in new_rows:
            if r["image_file_name"] not in existing_files:
                sr = r.copy()
                sr["processed_output_file"] = r["image_file_name"]
                sr["segmentation_failed"] = "False"
                sr["mask_area_ratio"] = "0.5"
                sr["center_overlap_ratio"] = "0.5"
                sr["number_of_components"] = "1"
                sr["touches_border"] = "False"
                sr["source"] = "public_dataset_2"
                sr["fold"] = "all"
                summary_additions.append(sr)
                
        if summary_additions:
            new_summary_df = pd.DataFrame(summary_additions)
            summary_df = pd.concat([summary_df, new_summary_df], ignore_index=True)
            summary_df.to_csv(summary_path, index=False)
            print(f"Appended {len(summary_additions)} rows to processing_summary.csv")
    
    # 2. Append to all_sampled_images.csv
    all_sampled_path = splits_dir / "all_sampled_images.csv"
    if all_sampled_path.exists():
        df_all = pd.read_csv(all_sampled_path, dtype=str)
        existing_dests = set(df_all["file_destination"])
        new_all_rows = [r.copy() for r in new_rows if r["file_destination"] not in existing_dests]
        for r in new_all_rows:
            r["fold"] = "all"
            r["source"] = "public_dataset_2"
        if new_all_rows:
            df_new_all = pd.DataFrame(new_all_rows)
            df_all = pd.concat([df_all, df_new_all], ignore_index=True)
            df_all.to_csv(all_sampled_path, index=False)
            print(f"Appended {len(new_all_rows)} rows to all_sampled_images.csv")

    # 3. Append to foldX_train.csv
    for fold_idx in range(1, 9):
        train_path = splits_dir / f"fold{fold_idx}_train.csv"
        if train_path.exists():
            df_train = pd.read_csv(train_path, dtype=str)
            existing_dests = set(df_train["file_destination"])
            new_train_rows = [r.copy() for r in new_rows if r["file_destination"] not in existing_dests]
            for r in new_train_rows:
                r["fold"] = f"fold{fold_idx}"
                r["source"] = "public_dataset_2"
            if new_train_rows:
                df_new_train = pd.DataFrame(new_train_rows)
                df_train = pd.concat([df_train, df_new_train], ignore_index=True)
                df_train.to_csv(train_path, index=False)
                print(f"Appended {len(new_train_rows)} rows to {train_path.name}")

if __name__ == "__main__":
    main()
