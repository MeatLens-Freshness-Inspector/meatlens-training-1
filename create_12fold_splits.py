import pandas as pd
from pathlib import Path
import json

def sort_split_df(df: pd.DataFrame) -> pd.DataFrame:
    LABEL_ORDER = ["fresh", "not fresh", "spoiled"]
    LABEL_SORT = {label: idx for idx, label in enumerate(LABEL_ORDER)}
    
    out = df.copy()
    out["label_order"] = out["label"].map(LABEL_SORT).fillna(999)
    out = out.sort_values(
        by=["label_order", "time_frame", "sample_number", "image_file_name"],
        kind="stable",
    ).drop(columns=["label_order"])
    return out.reset_index(drop=True)

def stratified_train_val_split(df: pd.DataFrame, val_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    val_parts = []

    for group_index, (_, group_df) in enumerate(df.groupby("label", sort=False)):
        shuffled = group_df.sample(frac=1.0, random_state=seed + group_index).reset_index(drop=True)
        if len(shuffled) <= 1:
            train_parts.append(shuffled)
            continue

        val_count = int(round(len(shuffled) * val_fraction))
        val_count = max(1, val_count)
        val_count = min(val_count, len(shuffled) - 1)

        val_parts.append(shuffled.iloc[:val_count].copy())
        train_parts.append(shuffled.iloc[val_count:].copy())

    train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, val_df

def label_count_dict(df: pd.DataFrame) -> dict[str, int]:
    LABEL_ORDER = ["fresh", "not fresh", "spoiled"]
    counts = df["label"].value_counts().to_dict()
    return {label: int(counts.get(label, 0)) for label in LABEL_ORDER}

def main():
    old_splits_dir = Path(r"E:\Thesis Code\generated_splits\cross_rotation_interval200_8samples_processed_roi")
    new_splits_dir = Path(r"E:\Thesis Code\generated_splits\cross_rotation_interval200_12samples_processed_roi")
    
    all_sampled_path = old_splits_dir / "all_sampled_images.csv"
    if not all_sampled_path.exists():
        print(f"[ERROR] Cannot find {all_sampled_path}")
        return
        
    df_all = pd.read_csv(all_sampled_path, dtype=str).fillna("")
    
    # Cast sample_number to int for sorting
    df_all["sample_number_int"] = df_all["sample_number"].astype(int)
    
    new_splits_dir.mkdir(parents=True, exist_ok=True)
    
    # Also save a copy of all_sampled_images to the new directory
    df_all.drop(columns=["sample_number_int"]).to_csv(new_splits_dir / "all_sampled_images.csv", index=False)
    
    ordered_sample_ids = (
        df_all[["sample_id", "sample_number_int", "pork_cut"]]
        .drop_duplicates()
        .sort_values("sample_number_int")
    )
    
    summary_rows = []
    leakage_rows = []
    
    for fold_index, row in enumerate(ordered_sample_ids.itertuples(index=False), start=1):
        held_out_sample = row.sample_id
        fold_name = f"fold{fold_index}"
        
        test_df = df_all[df_all["sample_id"] == held_out_sample].copy()
        train_pool = df_all[df_all["sample_id"] != held_out_sample].copy()
        
        train_df, val_df = stratified_train_val_split(train_pool, val_fraction=0.15, seed=42)
        
        # Add metadata
        train_df["split_type"] = "cross_rotation"
        train_df["fold"] = fold_name
        val_df["split_type"] = "cross_rotation"
        val_df["fold"] = fold_name
        test_df["split_type"] = "cross_rotation"
        test_df["fold"] = fold_name
        
        # Clean up temporary column
        train_df = train_df.drop(columns=["sample_number_int"])
        val_df = val_df.drop(columns=["sample_number_int"])
        test_df = test_df.drop(columns=["sample_number_int"])
        
        train_df = sort_split_df(train_df)
        val_df = sort_split_df(val_df)
        test_df = sort_split_df(test_df)
        
        train_df.to_csv(new_splits_dir / f"{fold_name}_train.csv", index=False)
        val_df.to_csv(new_splits_dir / f"{fold_name}_val.csv", index=False)
        test_df.to_csv(new_splits_dir / f"{fold_name}_test.csv", index=False)
        
        print(f"Generated {fold_name}: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test.")
        
        train_counts = label_count_dict(train_df)
        val_counts = label_count_dict(val_df)
        test_counts = label_count_dict(test_df)

        LABEL_ORDER = ["fresh", "not fresh", "spoiled"]
        summary_row = {
            "fold": fold_name,
            "held_out_sample": held_out_sample,
            "held_out_cut": row.pork_cut,
            "train_count": int(len(train_df)),
            "val_count": int(len(val_df)),
            "test_count": int(len(test_df)),
            "train_class_counts": json.dumps(train_counts),
            "val_class_counts": json.dumps(val_counts),
            "test_class_counts": json.dumps(test_counts),
        }
        for label in LABEL_ORDER:
            label_key = label.replace(" ", "_")
            summary_row[f"train_{label_key}_count"] = train_counts[label]
            summary_row[f"val_{label_key}_count"] = val_counts[label]
            summary_row[f"test_{label_key}_count"] = test_counts[label]
        summary_rows.append(summary_row)

        leaked_in_train = bool(train_df["sample_id"].eq(held_out_sample).any())
        leaked_in_val = bool(val_df["sample_id"].eq(held_out_sample).any())
        leakage_rows.append(
            {
                "fold": fold_name,
                "held_out_sample": held_out_sample,
                "present_in_train": leaked_in_train,
                "present_in_val": leaked_in_val,
                "status": "PASS" if (not leaked_in_train and not leaked_in_val) else "FAIL",
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    leakage_df = pd.DataFrame(leakage_rows)
    summary_df.to_csv(new_splits_dir / "cross_rotation_summary.csv", index=False)
    leakage_df.to_csv(new_splits_dir / "cross_rotation_leakage_check.csv", index=False)
    
    print("\nSummary and Leakage CSVs generated.")
    print(f"All 12-fold splits are saved in:\n{new_splits_dir}")

if __name__ == "__main__":
    main()
