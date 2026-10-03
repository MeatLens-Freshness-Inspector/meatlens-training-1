# MeatLens Training

Training and evaluation code for MeatLens, a pork-freshness image classifier. The project classifies images into three ordered classes:

1. `fresh`
2. `not fresh`
3. `spoiled`

The current deployment candidate is a CNN-only `MobileNetV3Small` model. Images are center-cropped, resized to `224x224`, segmented with HSV/LAB thresholds, placed on a neutral gray background, and then passed to the CNN. No handcrafted RGB, HSV, LAB, or GLCM feature branch is used by the final CNN-only model.

## Important evaluation distinction

This repository contains two related experiments:

| Experiment | Purpose | Evaluation protocol |
|---|---|---|
| `new10_mobilenetv3small_12fold_processed_roi_cnn_only.ipynb` | Official model evaluation | 12-fold cross-rotation; one physical sample is held out for testing in each fold |
| `new11_mobilenetv3small_12fold_fulldata_final_cnn_only.ipynb` | Final deployment model | All 12 samples are used; a random 10% validation split is used for early stopping only |

Metrics from the full-data model are validation-only and must not be reported as held-out test performance. Use the 12-fold experiment for the main generalization result and the full-data model for inference/demo use.

## Pipeline

```text
raw sample folders
        |
        v
interval sampling (up to 200 images per class)
        |
        v
HSV/LAB ROI segmentation + gray background + 224x224 RGB output
        |
        v
CSV metadata and sample-level split generation
        |
        +--> 12-fold cross-rotation training/evaluation
        |
        +--> full-data training with seeds 42, 123, and 2026
                         |
                         v
              H5/TFLite deployment model + folder inference
```

The 12-sample dataset currently contains eight MeatLens samples and four samples from Public Dataset 2. The generated `all_sampled_images.csv` contains 6,755 image records after the available-data and class sampling steps.

## Repository map

### Main preprocessing and split scripts

| File | Role |
|---|---|
| `organize_public_dataset2_by_cs.py` | Organizes Public Dataset 2 into source folders and writes a manifest. |
| `process_dataset2_interval_200.py` | Segments and interval-samples the organized Public Dataset 2 data. |
| `process_all_dataset2_interval_200.py` | Batch variant for the Public Dataset 2 folders used in the final dataset. |
| `apply_hsv_lab_threshold_roi_batch.py` | Shared HSV/LAB threshold segmentation and ROI preprocessing. |
| `add_samples_9_to_12_to_csvs.py` | Adds samples 9–12 to the existing sampled CSV metadata. |
| `create_12fold_splits.py` | Creates the 12 held-out-sample folds and leakage-check CSV. |

### Training, conversion, and inference

| File | Role |
|---|---|
| `mobilenetv3small_12fold_processed_roi_cnn_only_lib.py` | Training/evaluation library for the held-out 12-fold experiment. |
| `new10_mobilenetv3small_12fold_processed_roi_cnn_only.ipynb` | Runs the 12-fold processed-ROI CNN-only experiment. |
| `mobilenetv3small_12fold_fulldata_final_cnn_only_lib.py` | Training library for the final all-data deployment model. |
| `new11_mobilenetv3small_12fold_fulldata_final_cnn_only.ipynb` | Trains the final model using all 12 samples and three random seeds. |
| `new12_convert_final_12samples_fulldata_h5_to_onnx.ipynb` | Converts the final H5 model to ONNX when required. |
| `new13_folder_inference_final_12samples_fulldata_cnn_only.ipynb` | Runs inference on a local image folder or uploaded ZIP archive. |

### Analysis

| File or directory | Contents |
|---|---|
| `create_visual_feature_analysis.py` | Generates the visual-feature analysis notebook. |
| `visual_feature_analysis.ipynb` | Color, texture, separability, repeated-measures, time-trend, and variance analyses. |
| `results/visual_analysis/` | CSV summaries and figures produced by the visual-feature analysis. |
| `training_outputs/` | Training histories, predictions, summaries, model metadata, and inference results. |

## Environment

The notebooks were developed with Python 3.10 and TensorFlow 2.10.0. A GPU-enabled environment is recommended for training, but CPU execution is possible for small checks and inference.

Create an environment from the repository root:

```powershell
py -3.10 -m venv .venv-meatlens
.\.venv-meatlens\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install tensorflow==2.10.0 numpy pandas pillow scipy scikit-image scikit-learn `
    matplotlib seaborn opencv-python joblib tqdm nbformat ipykernel ipywidgets `
    pingouin statsmodels umap-learn
```

If TensorFlow installation differs on the target operating system or hardware, install the compatible TensorFlow build first and then install the remaining packages.

Start Jupyter or open the repository in VS Code:

```powershell
jupyter lab
```

Run notebooks with the working directory set to the repository root. The training libraries use `Path.cwd()` to locate `generated_splits/` and `training_outputs/`. Several older data-preparation scripts also contain the Windows path `E:\Thesis Code`; update those constants when using another checkout or operating system.

## Expected data layout

Raw image datasets are intentionally not included in Git. A local checkout should provide the relevant source data and generated preprocessing directories, for example:

```text
E:\Thesis Code\
├── interval sampled\                         # original MeatLens sample folders
├── public dataset 2\                         # Public Dataset 2 source data
├── processed_hsv_lab_threshold_roi_224\      # 224x224 processed ROI images
├── generated_splits\
│   └── cross_rotation_interval200_12samples_processed_roi\
│       ├── all_sampled_images.csv
│       ├── fold1_train.csv / fold1_val.csv / fold1_test.csv
│       ├── ...
│       └── fold12_train.csv / fold12_val.csv / fold12_test.csv
├── training_outputs\
└── results\
```

The split CSVs store image paths and metadata such as `sample_id`, `sample_number`, `pork_cut`, `capture_source`, `label`, and `split_type`. The training code resolves paths relative to the CSV when possible, but the processed images must exist locally.

## Rebuilding the dataset and splits

The exact preparation order depends on which source data is already available. For the current 12-sample pipeline, the intended sequence is:

1. Organize Public Dataset 2 with `organize_public_dataset2_by_cs.py`.
2. Create the processed, interval-sampled images with `process_dataset2_interval_200.py` or `process_all_dataset2_interval_200.py`.
3. Run `apply_hsv_lab_threshold_roi_batch.py` or the batch processing script to create the `224x224` neutral-background ROI images.
4. Run `add_samples_9_to_12_to_csvs.py` to append the new sample metadata to the existing sampled CSV.
5. Run `create_12fold_splits.py` to generate the 12-fold train/validation/test CSVs and the leakage audit.

The preprocessing scripts delete and recreate some output directories. Confirm the input and output paths before running them on a new dataset.

## Running the experiments

### 1. Held-out 12-fold evaluation

Open `new10_mobilenetv3small_12fold_processed_roi_cnn_only.ipynb` and run the setup, quality-check, and training cells. The experiment uses:

- `MobileNetV3Small`
- input shape `(224, 224, 3)`
- labels in the order `fresh`, `not fresh`, `spoiled`
- folds `fold1` through `fold12`
- seeds `42`, `123`, and `2026`
- training augmentation
- an eight-epoch classification-head stage followed by fine-tuning for up to 20 epochs

Results are written under:

```text
training_outputs/mobilenetv3small_12fold_processed_roi_cnn_only/
```

The notebook and library skip completed runs when matching metrics already exist, allowing interrupted experiments to continue.

### 2. Final full-data deployment model

Open `new11_mobilenetv3small_12fold_fulldata_final_cnn_only.ipynb`. It loads `all_sampled_images.csv`, makes a 90%/10% stratified train/validation split for each seed, and trains seeds `42`, `123`, and `2026`.

Set the notebook variable below to `True` only when the data and output paths have been checked:

```python
MANUAL_CONFIRM_RUN_FULL_TRAINING = True
```

Outputs are written under:

```text
training_outputs/mobilenetv3small_12fold_fulldata_final_cnn_only/
├── models/
├── predictions/
├── figures/
├── gradcam/
└── fulldata_final_seed_metrics.csv
```

The best-model bundle includes an H5 model, a TFLite model, and a metadata JSON file. Model binaries such as `.h5`, `.tflite`, and `.onnx` are ignored by Git and must be generated or copied into the local output directory.

### 3. Folder inference

Run `new13_folder_inference_final_12samples_fulldata_cnn_only.ipynb` after the final model bundle exists. The notebook accepts either:

- a local folder path; or
- a ZIP upload containing images.

For every image it performs center-square cropping, HSV/LAB segmentation, gray-background filling, resizing, and classification. It writes a timestamped inference directory containing `folder_predictions.csv`, `inference_summary.json`, optional processed images, and an error CSV when necessary.

The notebook loads the `meatlens_best_fulldata_final_cnn_only_mobilenetv3small` bundle. Treat the metadata JSON as authoritative for the selected seed and model path; the notebook’s descriptive text may refer to a specific seed while the best-model bundle can be produced from another seed.

## Main outputs

Useful tracked outputs include:

- `generated_splits/cross_rotation_interval200_12samples_processed_roi/cross_rotation_summary.csv`
- `generated_splits/cross_rotation_interval200_12samples_processed_roi/cross_rotation_leakage_check.csv`
- `training_outputs/mobilenetv3small_12fold_fulldata_final_cnn_only/fulldata_final_seed_metrics.csv`
- `training_outputs/mobilenetv3small_12fold_fulldata_final_cnn_only/fulldata_final_vs_previous_models_comparison.csv`
- `training_outputs/mobilenetv3small_12fold_fulldata_final_cnn_only/models/meatlens_best_fulldata_final_cnn_only_mobilenetv3small_metadata.json`
- `results/visual_analysis/feature_ranking.csv`
- `results/visual_analysis/feature_separability.csv`
- `results/visual_analysis/repeated_measures_results.csv`
- `results/visual_analysis/variance_partitioning.csv`
- `results/visual_analysis/time_trend_per_sample.csv`

The visual-feature analysis treats physical pork samples as repeated-measure subjects. Its inferential summaries aggregate image-level features to sample-level means to avoid treating repeated photographs of one physical sample as independent observations.

## Git and large files

The repository tracks Python scripts, notebooks, CSV summaries, split metadata, and selected result files. It ignores:

- raw and processed image files;
- virtual environments and notebook caches;
- model binaries and checkpoints (`.h5`, `.tflite`, `.onnx`, and related formats);
- logs and temporary files.

Before committing new work, check the staged file list so that generated archives, uploaded images, or machine-specific output are not added accidentally.

