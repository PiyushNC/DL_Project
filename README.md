# LeafLens / LeafDoc-Net-Inspired Plant Disease Classification

LeafLens is a Python, TensorFlow/Keras, and Streamlit project for plant leaf
disease classification. It combines a modular dataset pipeline, a
LeafDoc-Net-inspired DenseNet121 + MobileNetV2 hybrid, a MobileNetV2-only
baseline, held-out evaluation, and Grad-CAM++ explainability.

## Project Status

This repository is not yet a completed match to every part of the LeafDoc-Net
paper. The following have been verified:

- The existing trained MobileNetV2 baseline is available and has saved metrics.
- Actual MobileNetV2 test accuracy is `93.5711%` in
  `outputs/metrics/test_metrics.json`.
- The dataset currently contains **15 verified classes** across bell pepper,
  potato, and tomato.
- The repository does not contain a trained LeafDoc-Net hybrid checkpoint.
- Therefore, no hybrid accuracy, precision, recall, F1, AUC, or latency result
  is claimed or displayed.
- The 18-class upgrade target is configured but **not verified**.

Do not use these files as evidence that the hybrid architecture has been trained:
the existing metrics and model belong to the MobileNetV2 baseline.

## Installation

### Windows

```powershell
cd C:\path\to\DL-Project
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For PowerShell execution-policy issues:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

TensorFlow support depends on the installed Python version. Python 3.12 is the
version used for project verification.

### Conda alternative

```powershell
conda env create -f environment.yml
conda activate plant-disease-dl
```

## Dataset

PlantVillage is organized as class-named folders. The resolved nested layout is:

```text
data/raw/
└── PlantVillage/
    └── PlantVillage/
        ├── Pepper__bell___Bacterial_spot/
        ├── Pepper__bell___healthy/
        ├── Potato___Early_blight/
        ├── Potato___Late_blight/
        ├── Potato___healthy/
        ├── Tomato_Bacterial_spot/
        ├── Tomato_Early_blight/
        ├── Tomato_Late_blight/
        ├── Tomato_Leaf_Mold/
        ├── Tomato_Septoria_leaf_spot/
        ├── Tomato_Spider_mites_Two_spotted_spider_mite/
        ├── Tomato__Target_Spot/
        ├── Tomato__Tomato_YellowLeaf__Curl_Virus/
        ├── Tomato__Tomato_mosaic_virus/
        └── Tomato_healthy/
```

The actual dataset verification found 20,638 usable images across 15 classes.
The repository explicitly compares this result with an 18-class target and does
not report 18 classes as verified.

Configure Kaggle authentication or place/extract PlantVillage manually:

```powershell
python scripts/download_dataset.py
python scripts/prepare_dataset.py data/raw
python scripts/inspect_dataset.py data/raw --samples-per-class 3
```

## Dataset Split Method

The loader creates a deterministic 70/15/15 split using seed `42`:

- 14,416 training samples
- 3,099 validation samples
- 3,109 test samples

The test split is excluded from training, validation callbacks, checkpoint
selection, and hyperparameter tuning. Exact duplicate hashes and a
difference-hash grouping step keep duplicate/near-duplicate images from
crossing splits as reliably as practical. This is not a guarantee against every
duplicate or transformation variant.

The split manifest is generated at `data/splits.json`.

## Repository Structure

```text
app/
  streamlit_app.py
data/
  raw/
  processed/
  test/
  splits.json
src/
  config.py
  data_loader.py
  preprocessing.py
  model.py
  train.py
  evaluate.py
  predict.py
  gradcam_pp.py
  disease_info.py
scripts/
  download_dataset.py
  prepare_dataset.py
  inspect_dataset.py
tests/
outputs/
  models/
  plots/
  metrics/
```

## Hybrid Architecture

The implemented architecture contains:

- DenseNet121 without the original classification head.
- MobileNetV2 without the original classification head.
- DenseNet channel-attention transition:
  GAP -> Dense(1024) -> sigmoid -> reshape/multiply -> Conv2D(1024, 1×1)
  -> 2×2 max pooling.
- MobileNetV2 sigmoid attention:
  Conv2D(1280, 1×1) -> sigmoid -> element-wise multiplication.
- DenseNet and MobileNet global-average-pooled vectors concatenated.
- Dense(1024) -> Swish -> BatchNorm.
- Dense(512) -> Swish -> BatchNorm.
- Dense(128) -> Swish -> BatchNorm.
- Dense(num_classes) -> softmax.
- Configurable backbone freezing and upper-layer fine-tuning.

The paper's 128-layer activation/normalization details are ambiguous. This
repository assumes Swish and BatchNormalization for that layer, consistently
with the stated 1024/512 blocks.

## Training

The paper reference is Adam, learning rate `1e-5`, batch size `8`, 50 epochs,
and validation-loss early stopping with patience `3`. These are configurable.

For the hybrid model:

```powershell
python -m src.train --dataset data/raw --epochs 50 --batch-size 8 --learning-rate 1e-5 --seed 42 --freeze-backbones
```

Optional fine-tuning:

```powershell
python -m src.train --dataset data/raw --epochs 50 --batch-size 8 --learning-rate 1e-5 --fine-tune --fine-tune-epochs 5 --fine-tune-learning-rate 1e-6
```

Resume from the best checkpoint:

```powershell
python -m src.train --dataset data/raw --epochs 50 --batch-size 8 --resume
```

Training artifacts:

- `outputs/models/leafdoc_net.keras`
- `outputs/models/class_names.json`
- `outputs/metrics/leafdoc_training_history.json`
- `outputs/metrics/leafdoc_training_history.csv`
- `outputs/metrics/leafdoc_training_config.json`
- `outputs/plots/leafdoc_training_history.png`

## Evaluation

Evaluate a saved model exclusively on the held-out test set:

```powershell
python -m src.evaluate --dataset data/raw --model outputs/models/leafdoc_net.keras --class-names outputs/models/class_names.json --model-name leafdoc_net
```

A comparison JSON can contain explicitly trained baseline and hybrid models:

```json
[
  {
    "name": "mobilenetv2_baseline",
    "model_path": "outputs/models/plant_disease_mobilenetv2.keras",
    "class_names_path": "outputs/models/class_names.json"
  },
  {
    "name": "densenet121_baseline",
    "model_path": "outputs/models/densenet121_baseline.keras",
    "class_names_path": "outputs/models/class_names.json"
  },
  {
    "name": "leafdoc_net",
    "model_path": "outputs/models/leafdoc_net.keras",
    "class_names_path": "outputs/models/class_names.json"
  }
]
```

Run:

```powershell
python -m src.evaluate --dataset data/raw --comparison model_comparison.json
```

Evaluation writes JSON/CSV metrics, confusion matrices, and one-vs-rest ROC
plots. Undefined precision/recall/AUC values are represented by zero or null
with appropriate warnings, not fabricated values.

## Grad-CAM++

Generate overlays for uploaded images or held-out test samples:

```powershell
python -m src.gradcam_pp --model outputs/models/leafdoc_net.keras --class-names outputs/models/class_names.json --image leaf.jpg --branch both
python -m src.gradcam_pp --dataset data/raw --model outputs/models/leafdoc_net.keras --class-names outputs/models/class_names.json --count 5 --branch both --output outputs/plots/gradcampp
```

The DenseNet branch visualizes `conv5_block16_2_conv`; the MobileNet branch
visualizes `Conv_1`. The source heatmaps are coarse and interpolated to 224×224.
Grad-CAM++ identifies influential regions but is not medical or biological
diagnosis.

Grad-CAM++ is unavailable for the current app because no LeafDoc-Net checkpoint
is present. The app transparently falls back to the real trained MobileNetV2
model.

## Streamlit Application

Launch:

```powershell
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501`.

The app supports upload, preview, validation, prediction, top-three
probabilities, disease information, healthy/diseased status, performance
artifacts, and Grad-CAM++ when the hybrid checkpoint is available.

## Testing

```powershell
python -m pytest -q
python -m compileall -q src app scripts tests
```

At this audit, the unit suite passed `26` tests. The suite verifies the hybrid
shape/probabilities/save-load behavior and image preprocessing, but it does not
train the hybrid or prove paper-level accuracy.

## Hardware Requirements

- CPU: supported, but hybrid training is slow.
- RAM: at least 16 GB recommended.
- GPU: NVIDIA GPU with 4 GB VRAM is the minimum practical starting point.
- Hybrid batch size `8` follows the paper; `2` or `4` is appropriate for limited
  memory. This changes batch size only, not the architecture.
- Current verification detected CPU-only TensorFlow.

## Actual Measured Baseline Result

The saved MobileNetV2-only baseline reported a held-out test accuracy of
`93.57119798660278%`, with test loss `0.20681220293045044`, in the existing
saved metrics. These values apply to the actual MobileNetV2 baseline and must
not be presented as hybrid or LeafDoc-Net values.

No hybrid model is trained or saved in this repository, so hybrid experimental
results do not exist.

## Limitations

- 15, not 18, PlantVillage classes are currently verified.
- No measured hybrid experiment exists.
- The existing MobileNet-only metrics are not a LeafDoc-Net result.
- DenseNet and MobileNet preprocessing is mixed into one shared input by
  separate internal preprocessing operations.
- Grad-CAM++ heatmaps are branch-level localization, not causal explanations.
- Near-duplicate prevention is practical but imperfect.
- The project is an educational agricultural screening tool, not a clinical or
  professional diagnostic system.

## Research Paper Implementation Mapping

| Paper component | What the paper specifies | Repository implementation | Relevant source file | Status | Deviation or limitation |
|---|---|---|---|---|---|
| DenseNet121 backbone | Pretrained DenseNet121, include_top removed | `DenseNet121(include_top=False, weights="imagenet")` | `src/model.py` | Complete | Backbone frozen by default; paper reports broad fine-tuning, but configurable upper-layer tuning is included |
| MobileNetV2 backbone | Pretrained MobileNetV2, include_top removed | `MobileNetV2(include_top=False, weights="imagenet")` | `src/model.py` | Complete | Backbone frozen by default; configurable fine-tuning |
| DenseNet attentive transition | GAP -> Dense(1024) -> sigmoid -> reshape -> element-wise multiply -> Conv2D(1024,1×1) -> 2×2 max-pool | Custom `AttentiveTransition` implements these operations | `src/model.py` | Complete | Exactly mapped; assumes 1024 filters/channels based on paper text |
| MobileNet attention | 1×1 convolution -> sigmoid -> element-wise multiply | Custom `MobileAttention` uses Conv2D(1280,1×1) -> sigmoid -> multiply | `src/model.py` | Complete | 1280 matches MobileNetV2 final width at 224×224 |
| Dual GAP | GAP after each modified branch | Two `GlobalAveragePooling2D` layers | `src/model.py` | Complete | None |
| Feature concatenation | Concatenate DenseNet and MobileNet feature vectors | Concatenates 1024 and 1280 vectors | `src/model.py` | Complete | None |
| Dense head 1024 | Dense + Swish + BatchNorm | Implemented with 1024 units | `src/model.py` | Complete | Paper text is consistent |
| Dense head 512 | Dense + Swish + BatchNorm | Implemented with 512 units | `src/model.py` | Complete | Paper text is consistent |
| Dense head 128 | Additional dense layer | Dense(128) + Swish + BatchNorm | `src/model.py` | Complete with assumption | Paper gives insufficient activation/normalization detail; repository uses Swish/BatchNorm consistently |
| Softmax output | One softmax neuron per class | `Dense(num_classes, activation="softmax")` | `src/model.py` | Complete | Actual 15-class dataset instead of the paper's cassava/wheat datasets |
| Adam | Adam optimizer | Adam with configurable learning rate | `src/train.py` | Complete | Paper reference `1e-5`; current repo default is `1e-5` |
| Categorical crossentropy | Multiclass categorical loss | `categorical_crossentropy` | `src/train.py` | Complete | Saved/verified implementation |
| Batch size | Paper uses 8 | Default batch size 8 | `src/train.py`, `src/config.py` | Complete | Configurable for memory |
| Epochs | Paper uses 50 | Default 50 | `src/train.py`, `src/config.py` | Complete | Early stopping may stop sooner |
| Early stopping | Validation-loss monitor, patience 3 | `EarlyStopping(monitor="val_loss", patience=3)` | `src/train.py` | Complete | Configurable patience |
| Learning-rate reduction | Not central in this paper excerpt | `ReduceLROnPlateau` monitor val_loss | `src/train.py` | Complete | Extra robustness feature, not a paper contradiction |
| Model checkpoint | Not fully specified | Saves lowest validation-loss model | `src/train.py` | Complete | Implemented with `.keras` |
| 224×224 preprocessing | Fixed pixel size; normalized 0–1 | Resize RGB to 224×224; DenseNet/MobileNet-specific ImageNet preprocessors | `src/preprocessing.py`, `src/data_loader.py` | Complete | Split from the paper's generic 1/255 preprocessing and uses correct framework-specific transforms |
| Augmentation | Rotation, flips, width/height shifts | Horizontal flip, 8% rotation, 8% zoom | `src/preprocessing.py` | Complete with reduction | Paper does not require exact augmentation values; vertical flip/width-height shift are not copied exactly |
| Dataset split | 80/20 in paper | 70/15/15 project split | `src/data_loader.py` | Partial | No claim of exact paper split; project needs validation/test separation |
| Duplicate protection | Not specified | SHA-256 exact duplicate and difference-hash grouping | `src/data_loader.py` | Complete for practical practical scope | Cannot guarantee detection across strong transformations |
| Test isolation | Not explicitly in excerpt | Test dataset generated but never passed to `fit` | `src/data_loader.py`, `src/train.py`, `src/evaluate.py` | Complete | Verified in code |
| Training history | Not specified | JSON and CSV every epoch plus plot | `src/train.py` | Complete | Extra reporting |
| Resume training | Not specified | Reload `.keras`, recompile, infer initial epoch | `src/train.py` | Partial | Optimizer and callback resume are practical, but epoch/count history merge should be tested in a real resumed run |
| Grad-CAM++ | XAI interpretation | Mathematically weighted positive-gradient maps for both final backbone feature maps | `src/gradcam_pp.py` | Complete for branch localization | Not Grad-CAM++ on post-attention fused features; visualizes selected final backbone convolution maps |
| Confusion matrix | Paper evaluation | JSON/NumPy/CSV/PNG outputs | `src/evaluate.py` | Complete | None |
| Accuracy/precision/recall/F1/AUC | Paper evaluation | Real test metrics and one-vs-rest ROC-AUC | `src/evaluate.py` | Complete functionally | Actual hybrid metrics absent; existing baseline metrics apply only to MobileNet baseline |
| Baseline comparison | Not the LeafDoc-Net core | MobileNet-only and DenseNet-only constructors plus fair comparison runner | `src/model.py`, `src/evaluate.py` | Partial | Constructors exist; actual baseline training/evaluation artifacts are not fully supplied |
| Grad-CAM++ Streamlit integration | Paper uses Grad-CAM++ | Checkbox and original/overlay display, only if hybrid checkpoint exists | `app/streamlit_app.py` | Partial | Current app falls back to MobileNet and hybrid heatmap is unavailable until hybrid training |
| Multi-crop dataset | Cassava/wheat in paper | Bell pepper/potato/tomato PlantVillage, 15 verified classes | `src/config.py`, `data_loader.py` | Partial | Different dataset and 3 crops; 18 target not verified |

## Honest Assessment

The repository follows the paper at a moderate architectural level:

- Strong: two pretrained backbones, branch attention, feature concatenation,
  dense Swish/BatchNorm head, Adam/categorical crossentropy, split isolation,
  metrics, and Grad-CAM++ module.
- Moderate: DenseNet transition and MobileNet attention are implemented with
  practical details derived from the paper.
- Weak: no measured LeafDoc-Net hybrid run or hybrid checkpoint exists.
- Weak: no baseline artifact comparison has actually been executed.
- Weak: actual preprocessing, backbone loading, and Grad-CAM++ should be
  validated after a real hybrid training run.
- Not matched: 18-class target remains 15 verified classes, and the paper's
  cassava/wheat datasets are replaced by PlantVillage.

Overall closeness is approximate rather than publication-grade. The repository
implements a credible LeafDoc-Net-inspired architecture and robust pipeline,
but not yet proven paper-level experimental results.
