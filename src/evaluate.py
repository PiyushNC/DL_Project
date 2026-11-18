import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from src.config import CONFIG
from src.data_loader import create_datasets

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def evaluate_model(
    model_path: Path,
    class_names_path: Path,
    dataset_dir: Path,
) -> dict:
    """Evaluate a trained model on the held-out test split."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}. Run src/train.py first."
        )
    if not class_names_path.exists():
        raise FileNotFoundError(
            f"Class names not found: {class_names_path}. Run src/train.py first."
        )

    model = tf.keras.models.load_model(model_path)
    class_names = json.loads(class_names_path.read_text(encoding="utf-8"))
    _, _, test_ds, dataset_class_names = create_datasets(
        dataset_dir=dataset_dir,
        image_size=CONFIG.image_size,
        batch_size=CONFIG.batch_size,
        seed=CONFIG.seed,
    )
    if class_names != dataset_class_names:
        raise RuntimeError(
            "Saved class-label mapping does not match dataset class ordering."
        )

    loss, accuracy = model.evaluate(test_ds, verbose=0)
    y_true = np.concatenate([np.argmax(labels.numpy(), axis=1) for _, labels in test_ds])
    y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)

    if len(y_true) != len(y_pred):
        raise RuntimeError("Prediction and label counts do not match.")

    cm = confusion_matrix(y_true, y_pred)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    CONFIG.plots_dir.mkdir(parents=True, exist_ok=True)
    CONFIG.metrics_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(max(10, len(class_names)), max(8, len(class_names))))
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.xticks(
        np.arange(len(class_names)) + 0.5,
        class_names,
        rotation=90,
    )
    plt.yticks(
        np.arange(len(class_names)) + 0.5,
        class_names,
        rotation=0,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    confusion_path = CONFIG.plots_dir / "confusion_matrix.png"
    plt.savefig(confusion_path, dpi=150)
    plt.close()

    metrics = {
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "model_parameters": int(model.count_params()),
        "model_file_size_bytes": int(model_path.stat().st_size),
        "model_file_size_mb": round(model_path.stat().st_size / (1024 * 1024), 3),
    }
    metrics_path = CONFIG.metrics_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    LOGGER.info("Test accuracy: %.4f", accuracy)
    LOGGER.info("Test loss: %.4f", loss)
    LOGGER.info("Saved metrics to %s", metrics_path)
    LOGGER.info("Saved confusion matrix to %s", confusion_path)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the trained classifier")
    parser.add_argument("--dataset", type=Path, default=CONFIG.dataset_dir)
    return parser.parse_args()


if __name__ == "__main__":
    configure_logging()
    args = parse_args()
    evaluate_model(
        model_path=CONFIG.model_path,
        class_names_path=CONFIG.class_names_path,
        dataset_dir=args.dataset,
    )
