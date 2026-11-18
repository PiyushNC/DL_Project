# LeafLens

**AI-Powered Plant Disease Classification**

LeafLens is a college Deep Learning laboratory project that uses TensorFlow/Keras and transfer learning with MobileNetV2 to classify PlantVillage leaf images. It includes dataset validation, reproducible splitting, training, held-out evaluation, reusable inference, and a polished Streamlit interface.

No trained model, accuracy, sample prediction, or evaluation metric is generated or claimed until real project data is downloaded and the training/evaluation scripts run.

## Problem Statement

Plant diseases can reduce crop quality and yield. Automated leaf-image classification can provide an early educational screening step that helps users prioritize closer inspection and seek appropriate expert guidance.

LeafLens is not a substitute for diagnosis by a qualified agricultural or plant-health professional.

## Objectives

- Discover plant and disease classes automatically from a PlantVillage folder structure.
- Resize leaf images to 224×224 and apply MobileNetV2 preprocessing.
- Separate training, validation, and test images without overlap.
- Use augmentation for training images only.
- Transfer ImageNet knowledge through MobileNetV2.
- Fine-tune selected upper layers safely when requested.
- Measure real test accuracy, precision, recall, and F1-score.
- Provide reusable inference and top-three predictions.
- Present results through a botanical-themed Streamlit application.

## Technology Stack

- Python 3.11 or 3.12
- TensorFlow/Keras
- MobileNetV2 pretrained on ImageNet
- NumPy and pandas
- Pillow
- scikit-learn
- Matplotlib and Seaborn
- Streamlit
- pytest

## System Architecture

```text
PlantVillage folders
        |
        v
Dataset discovery and validation
        |
        v
Duplicate-aware 70/15/15 split
        |
        v
224x224 preprocessing + training augmentation
        |
        v
Frozen MobileNetV2 base + custom classification head
        |
        v
Keras .keras model + class_names.json
        |
        v
Held-out evaluation + metrics + plots
        |
        v
LeafLens Streamlit prediction interface
```

## Dataset Setup

The supported structure is:

```text
data/raw/
├── Apple___Apple_scab/
├── Apple___healthy/
├── Tomato___Early_blight/
└── Tomato___healthy/
```

Configure Kaggle authentication or manually place/extract PlantVillage in `data/raw`. Supported Kaggle setup:

```powershell
# Place your Kaggle API credentials in ~/.kaggle/kaggle.json
python scripts/download_dataset.py
python scripts/prepare_dataset.py data/raw
```

To inspect the dataset and generate a sample image grid:

```powershell
python scripts/inspect_dataset.py data/raw --samples-per-class 3
# Optional file output:
python scripts/inspect_dataset.py data/raw --samples-per-class 3 --output outputs/plots/dataset_samples.png
```

The loader validates files, logs unsupported formats and corrupted images, groups exact and near-identical images, and writes `data/splits.json`.

## Windows Environment Setup

From the project root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell prevents activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Alternative Conda setup:

```powershell
conda env create -f environment.yml
conda activate plant-disease-dl
```

## Model Training

Initial feature-extraction training:

```powershell
python -m src.train --dataset data/raw --epochs 15
```

Optional fine-tuning of the upper MobileNetV2 layers:

```powershell
python -m src.train --dataset data/raw --epochs 15 --fine-tune
```

Fine-tuning uses a smaller learning rate and keeps BatchNormalization layers frozen.

Additional options:

```powershell
python -m src.train `
  --dataset data/raw `
  --epochs 15 `
  --batch-size 32 `
  --learning-rate 0.001 `
  --fine-tune `
  --fine-tune-epochs 5 `
  --fine-tune-learning-rate 0.00001 `
  --fine-tune-layers 20 `
  --seed 42
```

Training creates:

- `outputs/models/plant_disease_mobilenetv2.keras`
- `outputs/models/class_names.json`
- `outputs/plots/training_history.png`
- `outputs/metrics/training_summary.json`
- `data/splits.json`

## Model Evaluation

```powershell
python -m src.evaluate --dataset data/raw
```

Evaluation creates `outputs/metrics/test_metrics.json` and `outputs/plots/confusion_matrix.png`. It reports actual held-out test loss and accuracy, macro/weighted precision, recall and F1-score, every per-class result, model parameter count, and model file size.

## Held-Out Prediction Test

```powershell
python -m src.predict --dataset data/raw --count 3
```

This runs real inference on the first three images in the held-out test split. Results include actual and predicted classes, confidence, top-three probabilities, and a confidence disclaimer.

## Launch LeafLens

```powershell
streamlit run app/streamlit_app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

The interface remains usable before training: it clearly explains that model artifacts are missing instead of generating demonstration predictions or metrics.

## Execution Flow

1. `scripts/download_dataset.py` obtains PlantVillage, or the user extracts it manually.
2. `scripts/prepare_dataset.py` validates the folder layout.
3. `src/data_loader.py` discovers classes and image files.
4. Corrupted and unsupported files are logged and skipped explicitly.
5. Exact and perceptually similar images are grouped so near-identical material is not distributed across splits.
6. A fixed seed produces 70% training, 15% validation, and 15% test records.
7. `src/preprocessing.py` resizes images and the model applies MobileNetV2 preprocessing.
8. Training augmentation is applied only to training batches.
9. `src/model.py` creates the frozen MobileNetV2 base and custom classifier head.
10. `src/train.py` trains the head, saves the best checkpoint, and optionally fine-tunes upper layers.
11. `src/evaluate.py` evaluates only the held-out test split and saves measured metrics and graphs.
12. `src/predict.py` loads the same model and class mapping to preprocess and classify a new leaf image.
13. LeafLens displays the result and educational disease information.

## Expected Directory Structure

```text
LeafLens/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits.json
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── disease_info.py
├── scripts/
│   ├── download_dataset.py
│   ├── prepare_dataset.py
│   └── inspect_dataset.py
├── tests/
├── outputs/
│   ├── models/
│   ├── plots/
│   └── metrics/
├── requirements.txt
├── environment.yml
└── README.md
```

## Screenshots

Place project screenshots in `outputs/screenshots/` and reference them here:

- Home page: `outputs/screenshots/home.png`
- Disease detection: `outputs/screenshots/detection.png`
- Model performance: `outputs/screenshots/performance.png`
- About project: `outputs/screenshots/about.png`

## Testing

Run all project tests:

```powershell
python -m pytest -q
```

Run syntax compilation:

```powershell
python -m compileall -q src app scripts tests
```

## Limitations

- Image-based model output does not guarantee a scientifically confirmed diagnosis.
- Disease appearance varies across species, growth stages, weather, and imaging conditions.
- Training images can contain photography or dataset bias.
- Confidence is a softmax probability, not medical or agricultural certainty.
- This project is designed for laboratory and educational use.

## Future Scope

- Add Grad-CAM visual explanations.
- Add per-user disease history.
- Expand controlled experiments across hyperparameters.
- Add cross-validation with several seeds.
- Improve duplicate detection with stronger perceptual-image fingerprints.
- Add confidence-threshold and out-of-distribution warnings.
- Integrate optional local expert-review workflows.
