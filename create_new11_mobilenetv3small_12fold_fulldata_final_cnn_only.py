#!/usr/bin/env python3
import json
import uuid
from pathlib import Path
from textwrap import dedent


def add_md(cells: list[dict], text: str) -> None:
    cells.append(
        {
            "cell_type": "markdown",
            "id": uuid.uuid4().hex[:8],
            "metadata": {},
            "source": (dedent(text).strip() + "\n").splitlines(keepends=True),
        }
    )


def add_code(cells: list[dict], text: str) -> None:
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "id": uuid.uuid4().hex[:8],
            "metadata": {},
            "outputs": [],
            "source": (dedent(text).strip() + "\n").splitlines(keepends=True),
        }
    )


def build_notebook() -> dict:
    cells: list[dict] = []

    add_md(
        cells,
        """
        # MeatLens MobileNetV3Small — Full 12-Sample Dataset Final Training

        This notebook trains the **final deployment model** using the complete 12-sample dataset
        (`all_sampled_images.csv`). There is **no held-out test fold** — all 12 samples participate
        in training. A 10% random stratified validation split is used **only** for early stopping
        during training.

        | Setting | Value |
        |---|---|
        | Architecture | MobileNetV3Small, CNN-only |
        | Dataset | All 12 samples (8 MeatLens + 4 Public Dataset 2) |
        | Splits | 90% train / 10% val (random stratified, per seed) |
        | Seeds | 3 (42, 123, 2026) |
        | Output root | `training_outputs/mobilenetv3small_12fold_fulldata_final_cnn_only/` |

        > **Note**: Val-set metrics are reported only as a training stability check. They are **not**
        > comparable to held-out test results from the 12-fold cross-validation notebook.
        """,
    )

    add_code(
        cells,
        """
        import importlib
        import inspect

        import pandas as pd
        from IPython.display import Image as DisplayImage, display

        import mobilenetv3small_12fold_fulldata_final_cnn_only_lib as libfd

        libfd = importlib.reload(libfd)

        print("Library source:", libfd.__file__)
        print("SPLIT_ROOT:", libfd.SPLIT_ROOT)
        print("EXTENSION_OUTPUT_ROOT:", libfd.EXTENSION_OUTPUT_ROOT)
        print("ALL_SAMPLED_CSV:", libfd.ALL_SAMPLED_CSV)
        print("EXTENSION_FOLDS:", libfd.EXTENSION_FOLDS)
        print("EXTENSION_RUN_SEEDS:", libfd.EXTENSION_RUN_SEEDS)
        print("VAL_SPLIT_FRACTION:", libfd.VAL_SPLIT_FRACTION)
        print("USE_TRAINING_AUGMENTATION:", libfd.USE_TRAINING_AUGMENTATION)

        libfd.ensure_output_dirs()
        libfd.print_library_status()
        """,
    )

    add_md(
        cells,
        """
        ## Step 1 — Load and Inspect the Full Dataset
        """,
    )

    add_code(
        cells,
        """
        all_df = libfd.load_all_fulldata_df()
        print(f"Loaded {len(all_df)} rows from all_sampled_images.csv.")
        print("Label counts:", all_df["label"].value_counts().to_dict())
        print("Unique samples:", all_df["sample_id"].nunique())
        print("Missing image paths:", int(all_df["missing_image_path"].sum()))
        all_df.head(5)
        """,
    )

    add_code(
        cells,
        """
        # Show per-sample row counts
        all_df.groupby(["sample_id", "pork_cut", "capture_source"])["label"].value_counts().unstack(fill_value=0)
        """,
    )

    add_md(
        cells,
        """
        ## Step 2 — Preview the Train/Val Split for Each Seed
        """,
    )

    add_code(
        cells,
        """
        for seed in libfd.EXTENSION_RUN_SEEDS:
            train_df, val_df = libfd.split_fulldata_train_val(all_df, seed=seed)
            print(f"seed={seed}: train={len(train_df)} | val={len(val_df)}")
            print(f"  train labels: {train_df['label'].value_counts().to_dict()}")
            print(f"  val   labels: {val_df['label'].value_counts().to_dict()}")
        """,
    )

    add_code(
        cells,
        """
        # Optional: audit that all images resolve to 224x224 (slow — skippable)
        # shape_audit = libfd.audit_image_shapes({"all_all": all_df})
        # shape_audit
        print("Shape audit skipped. Run manually if needed.")
        """,
    )

    add_md(
        cells,
        """
        ## Step 3 — Sanity Checks
        """,
    )

    add_code(
        cells,
        """
        source = inspect.getsource(libfd.train_mobilenetv3small_fulldata_final_cnn_only_model)

        checks = {
            "No fold CSV loading": "fold1_train" not in source,
            "No held-out test split": "test_df" not in source or "val_df" in source,
            "No handcrafted features": not any(kw in source for kw in ["glcm", "lab_", "hsv_", "StandardScaler"]),
            "Model name correct": "fulldata_final_cnn_only" in source,
            "ALL_SAMPLED_CSV points to correct folder": "cross_rotation_interval200_12samples_processed_roi" in str(libfd.ALL_SAMPLED_CSV),
            "VAL_SPLIT_FRACTION correct": libfd.VAL_SPLIT_FRACTION == 0.10,
        }
        for check_name, result in checks.items():
            print(f"  [{'OK' if result else 'FAIL'}] {check_name}")
        """,
    )

    add_md(
        cells,
        """
        ## Step 4 — Debug Single Run (Optional)
        """,
    )

    add_code(
        cells,
        """
        # Uncomment to run a single seed as a smoke test before full training:
        # debug_result = libfd.run_single_fulldata_final_cnn_only_experiment(seed=42, all_df=all_df)
        # print(debug_result)
        """,
    )

    add_md(
        cells,
        """
        ## Step 5 — Full Training (3 Seeds)

        Set `MANUAL_CONFIRM_RUN_FULL_TRAINING = True` to start training.
        """,
    )

    add_code(
        cells,
        """
        MANUAL_CONFIRM_RUN_FULL_TRAINING = False

        if MANUAL_CONFIRM_RUN_FULL_TRAINING:
            all_results = libfd.run_fulldata_final_cnn_only_training()
            print("Training complete.")
            print("Best model bundle:", all_results.get("best_bundle"))
        else:
            print("Full-dataset final training is ready but not started.")
            print("Set MANUAL_CONFIRM_RUN_FULL_TRAINING = True to train.")
        """,
    )

    add_md(
        cells,
        """
        ## Step 6 — Results Review
        """,
    )

    add_code(
        cells,
        """
        if libfd.SEGMENTED6_SEED_METRICS_PATH.exists():
            seed_metrics_df = pd.read_csv(libfd.SEGMENTED6_SEED_METRICS_PATH)
            print(f"Completed seeds: {len(seed_metrics_df)}")
            display(seed_metrics_df[["seed", "train_count", "val_count",
                                     "accuracy", "macro_f1", "adjacent_accuracy",
                                     "severe_error_rate", "h5_size_mb", "tflite_size_mb",
                                     "inference_mean_ms_per_image", "note"]].to_string(index=False))
        else:
            print("No results yet. Run Step 5 first.")
        """,
    )

    add_code(
        cells,
        """
        # Show the best model's metadata
        import json
        best_meta_path = libfd.EXTENSION_MODELS_ROOT / "meatlens_best_fulldata_final_cnn_only_mobilenetv3small_metadata.json"
        if best_meta_path.exists():
            meta = json.loads(best_meta_path.read_text(encoding="utf-8"))
            for k, v in meta.items():
                print(f"  {k}: {v}")
        else:
            print("Best model metadata not found yet.")
        """,
    )

    add_code(
        cells,
        """
        # Show saved confusion matrix figures
        import os
        figs_dir = libfd.EXTENSION_FIGURES_ROOT
        if figs_dir.exists():
            fig_files = sorted(figs_dir.glob("fulldata_final_cnn_only_all_seed*_normalized_confusion_matrix.png"))
            for fig_path in fig_files:
                print(fig_path.name)
                display(DisplayImage(filename=str(fig_path)))
        else:
            print("No figures directory yet.")
        """,
    )

    add_md(
        cells,
        """
        ## Step 7 — Comparison with Cross-Validation Models
        """,
    )

    add_code(
        cells,
        """
        comparison_bundle = libfd.create_fulldata_final_vs_previous_models_comparison()
        if comparison_bundle is not None:
            print("Comparison CSV:", comparison_bundle["comparison_csv_path"])
            comparison_bundle["comparison_df"]
        else:
            print("No comparison data available yet (train first).")
        """,
    )

    add_md(
        cells,
        """
        ## Step 8 — Summary Regeneration (After Training)
        """,
    )

    add_code(
        cells,
        """
        MANUAL_CONFIRM_REGENERATE_SUMMARIES = False

        if MANUAL_CONFIRM_REGENERATE_SUMMARIES:
            regen_bundle = libfd.regenerate_metrics_summaries_and_graphs()
        else:
            print("Summary regeneration ready but not started.")
        """,
    )

    add_md(
        cells,
        """
        ## Completion Check
        """,
    )

    add_code(
        cells,
        """
        expected_seeds = libfd.EXTENSION_RUN_SEEDS
        print("Expected seeds:", expected_seeds)

        if libfd.SEGMENTED6_SEED_METRICS_PATH.exists():
            seed_metrics_df = pd.read_csv(libfd.SEGMENTED6_SEED_METRICS_PATH)
            completed_seeds = set(seed_metrics_df["seed"].astype(int).tolist())
            print("Completed seeds:", sorted(completed_seeds))
            missing = [s for s in expected_seeds if s not in completed_seeds]
            print("Missing seeds:", missing)
        else:
            print("No training results yet.")

        if libfd.SEGMENTED6_FAILED_RUNS_PATH.exists():
            print("Failed runs:")
            display(pd.read_csv(libfd.SEGMENTED6_FAILED_RUNS_PATH))
        else:
            print("No failed runs logged.")
        """,
    )

    add_md(
        cells,
        """
        ---

        This notebook trains the **final deployment MobileNetV3Small model** on all 12 pork samples.
        Because no sample is held out, the val-set accuracy reflects training stability, not
        generalisation. Use the 12-fold cross-validation notebook (`new10_mobilenetv3small_12fold_processed_roi_cnn_only.ipynb`)
        for generalization estimates. The best model (by val macro F1) is automatically copied to
        `training_outputs/mobilenetv3small_12fold_fulldata_final_cnn_only/models/meatlens_best_fulldata_final_cnn_only_mobilenetv3small.h5`.
        """,
    )

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    output_path = Path("new11_mobilenetv3small_12fold_fulldata_final_cnn_only.ipynb")
    notebook = build_notebook()
    output_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(f"Wrote notebook to {output_path.resolve()}")


if __name__ == "__main__":
    main()
