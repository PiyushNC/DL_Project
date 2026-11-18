import argparse
import json
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf

from src.config import CONFIG
from src.data_loader import create_datasets
from src.model import create_mobilenetv2_model, enable_fine_tuning

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def train(
    dataset_dir: Path,
    model_path: Path,
    class_names_path: Path,
    epochs: int,
    fine_tune: bool,
    batch_size: int = CONFIG.batch_size,
    learning_rate: float = CONFIG.learning_rate,
    fine_tune_epochs: int = CONFIG.fine_tune_epochs,
    fine_tune_learning_rate: float = CONFIG.fine_tune_learning_rate,
    fine_tune_last_layers: int = CONFIG.fine_tune_last_layers,
    seed: int = CONFIG.seed,
) -> tuple[tf.keras.Model, dict]:
    """Train the model and save reproducible artifacts."""
    tf.keras.utils.set_random_seed(seed)
    LOGGER.info("Discovering and splitting PlantVillage images...")
    train_ds, val_ds, _, class_names = create_datasets(
        dataset_dir=dataset_dir,
        image_size=CONFIG.image_size,
        batch_size=batch_size,
        seed=seed,
    )

    model = create_mobilenetv2_model(
        num_classes=len(class_names),
        image_size=CONFIG.image_size,
        dropout_rate=CONFIG.dropout_rate,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    class_names_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=CONFIG.early_stopping_patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            save_weights_only=False,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=CONFIG.reduce_lr_factor,
            patience=CONFIG.reduce_lr_patience,
            min_lr=CONFIG.reduce_lr_min,
            verbose=1,
        ),
    ]

    start_time = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    combined_history = dict(history.history)
    fine_tune_completed = False

    if fine_tune:
        enable_fine_tuning(
            model,
            trainable_last_layers=fine_tune_last_layers,
            learning_rate=fine_tune_learning_rate,
        )
        initial_epoch = len(history.history["accuracy"])
        fine_tune_history = model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=initial_epoch,
            epochs=initial_epoch + fine_tune_epochs,
            callbacks=callbacks,
            verbose=1,
        )
        for key, values in fine_tune_history.history.items():
            combined_history.setdefault(key, []).extend(values)
        fine_tune_completed = True

    duration_seconds = time.perf_counter() - start_time

    model.save(model_path)
    class_names_path.write_text(
        json.dumps(class_names, indent=2),
        encoding="utf-8",
    )

    CONFIG.plots_dir.mkdir(parents=True, exist_ok=True)
    CONFIG.metrics_dir.mkdir(parents=True, exist_ok=True)
    plot_training_history(combined_history)

    summary = {
        "training_duration_seconds": round(duration_seconds, 2),
        "initial_training_epochs": len(history.history["accuracy"]),
        "fine_tuning_completed": fine_tune_completed,
        "fine_tuning_epochs": fine_tune_epochs if fine_tune_completed else 0,
        "learning_rate": learning_rate,
        "fine_tune_learning_rate": fine_tune_learning_rate if fine_tune_completed else None,
        "batch_size": batch_size,
        "seed": seed,
        "number_of_classes": len(class_names),
        "class_names": class_names,
        "final_epoch_training_accuracy": float(combined_history["accuracy"][-1]),
        "final_epoch_validation_accuracy": float(combined_history["val_accuracy"][-1]),
        "final_epoch_training_loss": float(combined_history["loss"][-1]),
        "final_epoch_validation_loss": float(combined_history["val_loss"][-1]),
    }
    summary_path = CONFIG.metrics_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    LOGGER.info("Saved model to %s", model_path)
    LOGGER.info("Saved class names to %s", class_names_path)
    LOGGER.info("Training duration: %.2f seconds", duration_seconds)
    LOGGER.info("Saved training summary to %s", summary_path)
    return model, combined_history


def plot_training_history(history: dict[str, list[float]]) -> Path:
    """Plot and save accuracy and loss curves from real training history."""
    plot_path = CONFIG.plots_dir / "training_history.png"
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["accuracy"], label="Training")
    axes[0].plot(history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["loss"], label="Training")
    axes[1].plot(history["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close(figure)
    LOGGER.info("Saved training curves to %s", plot_path)
    return plot_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the PlantVillage classifier")
    parser.add_argument("--dataset", type=Path, default=CONFIG.dataset_dir)
    parser.add_argument("--epochs", type=int, default=CONFIG.epochs)
    parser.add_argument("--fine-tune", action="store_true")
    parser.add_argument("--batch-size", type=int, default=CONFIG.batch_size)
    parser.add_argument("--learning-rate", type=float, default=CONFIG.learning_rate)
    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=CONFIG.fine_tune_epochs,
    )
    parser.add_argument(
        "--fine-tune-learning-rate",
        type=float,
        default=CONFIG.fine_tune_learning_rate,
    )
    parser.add_argument(
        "--fine-tune-layers",
        type=int,
        default=CONFIG.fine_tune_last_layers,
    )
    parser.add_argument("--seed", type=int, default=CONFIG.seed)
    return parser.parse_args()


if __name__ == "__main__":
    configure_logging()
    args = parse_args()
    train(
        dataset_dir=args.dataset,
        model_path=CONFIG.model_path,
        class_names_path=CONFIG.class_names_path,
        epochs=args.epochs,
        fine_tune=args.fine_tune,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        fine_tune_epochs=args.fine_tune_epochs,
        fine_tune_learning_rate=args.fine_tune_learning_rate,
        fine_tune_last_layers=args.fine_tune_layers,
        seed=args.seed,
    )
