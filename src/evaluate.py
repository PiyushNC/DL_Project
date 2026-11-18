import argparse
import csv
import json
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from src.config import CONFIG
from src.data_loader import create_datasets
from src.model import AttentiveTransition, MobileAttention

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_evaluation_model(model_path: Path) -> tf.keras.Model:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved model not found: {model_path}. Train or provide it first."
        )
    return tf.keras.models.load_model(
        model_path,
        custom_objects={
            "AttentiveTransition": AttentiveTransition,
            "MobileAttention": MobileAttention,
        },
    )


def collect_predictions(test_ds, model):
    y_true = np.concatenate(
        [np.argmax(labels.numpy(), axis=1) for _, labels in test_ds]
    )
    y_probability = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_probability, axis=1)
    if len(y_true) != len(y_pred):
        raise RuntimeError("Prediction and label counts do not match.")
    return y_true, y_pred, y_probability


def _safe_roc_auc(y_true: np.ndarray, y_probability: np.ndarray, num_classes: int):
    one_hot = np.zeros((len(y_true), num_classes), dtype=np.float32)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    curve_details = {}

    macro_auc = None
    micro_auc = None
    if y_probability.shape[1] == num_classes:
        try:
            macro_auc = float(
                roc_auc_score(one_hot, y_probability, average="macro", multi_class="ovr")
            )
        except ValueError as exc:
            LOGGER.warning("Macro multiclass ROC-AUC undefined: %s", exc)
        try:
            micro_auc = float(
                roc_auc_score(one_hot, y_probability, average="micro", multi_class="ovr")
            )
        except ValueError as exc:
            LOGGER.warning("Micro multiclass ROC-AUC undefined: %s", exc)

    for class_index in range(num_classes):
        binary_true = one_hot[:, class_index]
        if np.unique(binary_true).size < 2:
            curve_details[str(class_index)] = {
                "class_index": class_index,
                "roc_auc": None,
                "undefined_reason": "Class absent from held-out test labels",
            }
            continue
        false_positive, true_positive, _ = roc_curve(
            binary_true,
            y_probability[:, class_index],
        )
        curve_details[str(class_index)] = {
            "class_index": class_index,
            "false_positive_rate": false_positive.tolist(),
            "true_positive_rate": true_positive.tolist(),
            "roc_auc": float(auc(false_positive, true_positive)),
        }
    return macro_auc, micro_auc, curve_details


def save_confusion_matrix(cm: np.ndarray, class_names: list[str], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "confusion_matrix.npy", cm)

    with (output_dir / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["true\\predicted", *class_names])
        for class_name, row in zip(class_names, cm):
            writer.writerow([class_name, *row.tolist()])

    figure_size = max(8, len(class_names) * 0.55)
    plt.figure(figsize=(figure_size, figure_size * 0.82))
    sns.heatmap(
        cm,
        annot=len(class_names) <= 15,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=90)
    plt.tight_layout()
    confusion_path = output_dir / "confusion_matrix.png"
    plt.savefig(confusion_path, dpi=150)
    plt.close()
    return confusion_path


def save_roc_plot(
    curve_details: dict,
    class_names: list[str],
    plot_path: Path,
    macro_auc,
    micro_auc,
):
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    for class_index, class_name in enumerate(class_names):
        detail = curve_details[str(class_index)]
        if detail.get("false_positive_rate"):
            plt.plot(
                detail["false_positive_rate"],
                detail["true_positive_rate"],
                lw=1.5,
                label=f"{class_name} (AUC={detail['roc_auc']:.3f})",
            )
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    title = "One-vs-rest multiclass ROC"
    if macro_auc is not None:
        title += f" | Macro AUC={macro_auc:.3f}"
    if micro_auc is not None:
        title += f" | Micro AUC={micro_auc:.3f}"
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", fontsize="x-small")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


def measure_inference_latency(model, test_ds, warmup_batches: int = 5, measured_batches: int = 50):
    """Measure model.predict latency on real test batches."""
    batches = list(test_ds.take(max(warmup_batches + measured_batches, 1)).as_numpy_iterator())
    if not batches:
        raise RuntimeError("The held-out test dataset has no batches.")

    for images, _ in batches[:warmup_batches]:
        model.predict(images, verbose=0)

    durations = []
    samples = 0
    for images, _ in batches[warmup_batches:warmup_batches + measured_batches]:
        started = time.perf_counter()
        model.predict(images, verbose=0)
        durations.append(time.perf_counter() - started)
        samples += len(images)

    if not durations:
        raise RuntimeError("No test batches were available for latency measurement.")
    average_batch_seconds = float(np.mean(durations))
    total_seconds = float(np.sum(durations))
    average_sample_seconds = total_seconds / max(samples, 1)
    throughput = samples / total_seconds if total_seconds > 0 else 0.0
    return {
        "warmup_batches": warmup_batches,
        "measured_batches": len(durations),
        "measured_samples": samples,
        "average_batch_latency_ms": round(average_batch_seconds * 1000.0, 3),
        "average_inference_latency_ms": round(average_sample_seconds * 1000.0, 3),
        "throughput_samples_per_second": round(throughput, 3),
    }


def evaluate_model(
    model_path: Path,
    class_names_path: Path,
    dataset_dir: Path,
    model_name: str = "leafdoc_net",
    output_name: str | None = None,
) -> dict:
    """Evaluate one saved model exclusively on the held-out test split."""
    model = load_evaluation_model(model_path)
    class_names = json.loads(class_names_path.read_text(encoding="utf-8"))
    if not isinstance(class_names, list) or not class_names:
        raise ValueError("The class-name mapping is empty or invalid.")

    _train_ds, _val_ds, test_ds, dataset_class_names = create_datasets(
        dataset_dir=dataset_dir,
        image_size=CONFIG.image_size,
        batch_size=CONFIG.batch_size,
        seed=CONFIG.seed,
    )
    if class_names != dataset_class_names:
        raise RuntimeError(
            "Saved class-label mapping does not match dataset class ordering."
        )
    num_classes = len(class_names)
    if int(model.output_shape[-1]) != num_classes:
        raise RuntimeError(
            f"Model output width {model.output_shape[-1]} differs from "
            f"{num_classes} class mappings."
        )

    y_true, y_pred, y_probability = collect_predictions(test_ds, model)
    loss = None
    try:
        loss = float(model.evaluate(test_ds, verbose=0)[0])
    except Exception as exc:
        LOGGER.warning("Keras test loss could not be calculated: %s", exc)

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
        labels=list(range(num_classes)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    per_class_rows = []
    for class_index, class_name in enumerate(class_names):
        class_report = report.get(class_name, {})
        per_class_rows.append({
            "class_index": class_index,
            "class_name": class_name,
            "precision": class_report.get("precision"),
            "recall": class_report.get("recall"),
            "f1_score": class_report.get("f1-score"),
            "support": class_report.get("support", 0),
        })

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
    )
    macro_auc, micro_auc, curve_details = _safe_roc_auc(
        y_true,
        y_probability,
        num_classes,
    )

    output_name = output_name or model_name
    metrics_dir = CONFIG.metrics_dir / output_name
    plots_dir = CONFIG.plots_dir / output_name
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    confusion_path = save_confusion_matrix(cm, class_names, plots_dir)
    roc_path = plots_dir / "multiclass_roc_curve.png"
    save_roc_plot(
        curve_details,
        class_names,
        roc_path,
        macro_auc,
        micro_auc,
    )
    latency = measure_inference_latency(
        model,
        test_ds,
        warmup_batches=5,
        measured_batches=50,
    )

    pd.DataFrame(per_class_rows).to_csv(
        metrics_dir / "per_class_metrics.csv",
        index=False,
    )
    (metrics_dir / "roc_auc.json").write_text(
        json.dumps(
            {
                "macro_auc": macro_auc,
                "micro_auc": micro_auc,
                "one_vs_rest": curve_details,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    metrics = {
        "model_name": model_name,
        "test_loss": loss,
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "classification_report": report,
        "per_class_metrics": per_class_rows,
        "confusion_matrix": cm.tolist(),
        "macro_roc_auc": macro_auc,
        "micro_roc_auc": micro_auc,
        "model_parameters": int(model.count_params()),
        "model_file_size_bytes": int(model_path.stat().st_size),
        "model_file_size_mb": round(model_path.stat().st_size / (1024 * 1024), 3),
        "inference": latency,
        "test_sample_count": int(len(y_true)),
        "class_mapping": class_names,
        "held_out_test_only": True,
        "plot_paths": {
            "confusion_matrix": str(confusion_path),
            "multiclass_roc": str(roc_path),
        },
    }
    metrics_path = metrics_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame([metrics]).drop(
        columns=["classification_report", "per_class_metrics", "confusion_matrix", "plot_paths"],
        errors="ignore",
    ).to_csv(metrics_dir / "test_metrics.csv", index=False)

    LOGGER.info("Model: %s", model_name)
    LOGGER.info("Held-out test accuracy: %.4f", metrics["test_accuracy"])
    LOGGER.info("Held-out test loss: %s", loss)
    LOGGER.info("Macro F1: %.4f", macro_f1)
    LOGGER.info("Saved metrics to %s", metrics_path)
    return metrics


def compare_models(
    dataset_dir: Path,
    model_specs: list[dict],
    comparison_name: str = "model_comparison",
) -> dict:
    """Evaluate explicitly provided trained models on the identical test split."""
    results = []
    for spec in model_specs:
        model_path = Path(spec["model_path"])
        class_names_path = Path(spec["class_names_path"])
        model_name = spec["name"]
        result = evaluate_model(
            model_path=model_path,
            class_names_path=class_names_path,
            dataset_dir=dataset_dir,
            model_name=model_name,
            output_name=f"{comparison_name}/{model_name}",
        )
        results.append(result)

    comparison_dir = CONFIG.metrics_dir / comparison_name
    comparison_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in results:
        rows.append({
            "model_name": result["model_name"],
            "test_accuracy": result["test_accuracy"],
            "test_loss": result["test_loss"],
            "macro_precision": result["macro_precision"],
            "macro_recall": result["macro_recall"],
            "macro_f1": result["macro_f1"],
            "weighted_precision": result["weighted_precision"],
            "weighted_recall": result["weighted_recall"],
            "weighted_f1": result["weighted_f1"],
            "macro_roc_auc": result["macro_roc_auc"],
            "micro_roc_auc": result["micro_roc_auc"],
            "model_parameters": result["model_parameters"],
            "model_file_size_mb": result["model_file_size_mb"],
            "average_inference_latency_ms": result["inference"]["average_inference_latency_ms"],
            "throughput_samples_per_second": result["inference"]["throughput_samples_per_second"],
            "test_sample_count": result["test_sample_count"],
        })
    comparison_path = comparison_dir / "comparison.csv"
    pd.DataFrame(rows).to_csv(comparison_path, index=False)
    comparison_json = comparison_dir / "comparison.json"
    comparison_json.write_text(json.dumps({"models": rows}, indent=2), encoding="utf-8")

    if rows:
        names = [row["model_name"] for row in rows]
        accuracies = [row["test_accuracy"] for row in rows]
        macro_f1 = [row["macro_f1"] for row in rows]
        figure, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].bar(names, accuracies)
        axes[0].set_title("Held-out test accuracy")
        axes[0].set_ylabel("Accuracy")
        axes[1].bar(names, macro_f1)
        axes[1].set_title("Held-out test macro F1")
        axes[1].set_ylabel("Macro F1")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        comparison_plot = CONFIG.plots_dir / comparison_name / "model_comparison.png"
        comparison_plot.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(comparison_plot, dpi=150)
        plt.close()

    LOGGER.info("Saved model comparison to %s", comparison_path)
    return {"models": rows, "comparison_csv": str(comparison_path)}


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate trained plant classifiers")
    parser.add_argument("--dataset", type=Path, default=CONFIG.dataset_dir)
    parser.add_argument("--model", type=Path, default=CONFIG.leafdoc_model_path)
    parser.add_argument("--class-names", type=Path, default=CONFIG.class_names_path)
    parser.add_argument("--model-name", default="leafdoc_net")
    parser.add_argument("--comparison", type=Path)
    return parser


if __name__ == "__main__":
    configure_logging()
    arguments = parse_args().parse_args()
    try:
        if arguments.comparison:
            specs = json.loads(arguments.comparison.read_text(encoding="utf-8"))
            if not isinstance(specs, list):
                raise ValueError("Comparison JSON must be a list of model specifications.")
            comparison = compare_models(
                dataset_dir=arguments.dataset,
                model_specs=specs,
            )
            print(json.dumps(comparison, indent=2))
        else:
            result = evaluate_model(
                model_path=arguments.model,
                class_names_path=arguments.class_names,
                dataset_dir=arguments.dataset,
                model_name=arguments.model_name,
            )
            print(json.dumps(result, indent=2))
    except (FileNotFoundError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        LOGGER.error("Evaluation could not start: %s", exc)
        raise SystemExit(1)
