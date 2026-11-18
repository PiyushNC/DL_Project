import argparse
import csv
import json
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf

from src.config import CONFIG
from src.data_loader import create_datasets
from src.model import (
    AttentiveTransition,
    MobileAttention,
    create_leafdoc_net_model,
    enable_leafdoc_fine_tuning,
    get_leafdoc_model_report,
)

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


class TrainingHistoryLogger(tf.keras.callbacks.Callback):
    """Append each completed epoch to JSON and CSV files."""

    def __init__(self, json_path: Path, csv_path: Path) -> None:
        super().__init__()
        self.json_path = json_path
        self.csv_path = csv_path
        self.started_at = time.perf_counter()
        self.rows = []
        json_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        if json_path.exists():
            try:
                self.rows = json.loads(json_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                LOGGER.warning("Could not parse existing history JSON; starting new history.")
                self.rows = []

    def on_epoch_begin(self, epoch, logs=None):
        del logs
        self.started_at = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        lr = float(
            tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        )
        row = {
            "epoch": int(epoch) + 1,
            "accuracy": float(logs.get("accuracy", 0.0)),
            "loss": float(logs.get("loss", 0.0)),
            "val_accuracy": float(logs.get("val_accuracy", 0.0)),
            "val_loss": float(logs.get("val_loss", 0.0)),
            "learning_rate": lr,
            "epoch_duration_seconds": round(time.perf_counter() - self.started_at, 3),
        }
        self.rows.append(row)
        self.json_path.write_text(
            json.dumps(self.rows, indent=2),
            encoding="utf-8",
        )
        with self.csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerows(self.rows)
        LOGGER.info(
            "Epoch %d: loss=%.4f val_loss=%.4f accuracy=%.4f val_accuracy=%.4f lr=%g",
            row["epoch"],
            row["loss"],
            row["val_loss"],
            row["accuracy"],
            row["val_accuracy"],
            lr,
        )


def configure_hardware_memory() -> dict:
    """Enable GPU memory growth where available; the CPU remains supported."""
    report = {"gpus": []}
    try:
        gpus = tf.config.list_physical_devices("GPU")
        report["gpus"] = [gpu.name for gpu in gpus]
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as exc:
        LOGGER.warning("GPU memory growth could not be configured: %s", exc)
    return report


def build_callbacks(
    model_path: Path,
    metrics_dir: Path,
    early_stopping_patience: int = 3,
    reduce_lr_patience: int = 2,
    reduce_lr_factor: float = 0.2,
    reduce_lr_min: float = 1e-7,
) -> list[tf.keras.callbacks.Callback]:
    """Create validation-loss checkpointing and recovery callbacks."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=reduce_lr_min,
            verbose=1,
        ),
        TrainingHistoryLogger(
            json_path=metrics_dir / "leafdoc_training_history.json",
            csv_path=metrics_dir / "leafdoc_training_history.csv",
        ),
    ]


def save_training_config(
    path: Path,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    LOGGER.info("Saved training configuration to %s", path)


def plot_training_history(history_path: Path, plot_path: Path) -> None:
    if not history_path.exists():
        return
    rows = json.loads(history_path.read_text(encoding="utf-8"))
    if not rows:
        return
    epochs = [row["epoch"] for row in rows]
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [row["accuracy"] for row in rows], label="Training")
    axes[0].plot(epochs, [row["val_accuracy"] for row in rows], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, [row["loss"] for row in rows], label="Training")
    axes[1].plot(epochs, [row["val_loss"] for row in rows], label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close(figure)
    LOGGER.info("Saved training curves to %s", plot_path)


def train_leafdoc(
    dataset_dir: Path,
    model_path: Path,
    class_names_path: Path,
    epochs: int = 50,
    batch_size: int = 8,
    learning_rate: float = 1e-5,
    seed: int = 42,
    freeze_backbones: bool = True,
    fine_tune: bool = False,
    fine_tune_epochs: int = 5,
    fine_tune_learning_rate: float = 1e-6,
    densenet_last_layers: int = 30,
    mobilenet_last_layers: int = 20,
    early_stopping_patience: int = 3,
    reduce_lr_patience: int = 2,
    reduce_lr_factor: float = 0.2,
    reduce_lr_min: float = 1e-7,
    resume: bool = False,
) -> tuple[tf.keras.Model, dict]:
    """Train LeafDoc-Net using train/validation only; test data is never touched."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    tf.keras.utils.set_random_seed(seed)
    hardware = configure_hardware_memory()
    LOGGER.info("Detected devices: %s", hardware)
    LOGGER.info(
        "Preparing reproducible train/validation/test splits "
        "(test is reserved exclusively for final evaluation)"
    )
    train_ds, val_ds, _test_ds, class_names = create_datasets(
        dataset_dir=dataset_dir,
        image_size=CONFIG.image_size,
        batch_size=batch_size,
        seed=seed,
    )

    initial_epoch = 0
    if resume:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Cannot resume: saved checkpoint not found at {model_path}"
            )
        LOGGER.info("Loading checkpoint for resume: %s", model_path)
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "AttentiveTransition": AttentiveTransition,
                "MobileAttention": MobileAttention,
            },
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        history_path = CONFIG.metrics_dir / "leafdoc_training_history.json"
        if history_path.exists():
            try:
                existing_rows = json.loads(history_path.read_text(encoding="utf-8"))
                initial_epoch = len(existing_rows)
            except json.JSONDecodeError:
                LOGGER.warning("Existing history JSON is invalid; resume starts epoch 1.")
        LOGGER.info("Resuming at epoch %d", initial_epoch)
    else:
        model = create_leafdoc_net_model(
            num_classes=len(class_names),
            image_size=CONFIG.image_size,
            weights="imagenet",
            freeze_backbones=freeze_backbones,
            dropout_rate=CONFIG.dropout_rate,
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    class_names_path.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.metrics_dir.mkdir(parents=True, exist_ok=True)

    history_json = CONFIG.metrics_dir / "leafdoc_training_history.json"
    history_csv = CONFIG.metrics_dir / "leafdoc_training_history.csv"
    if not resume:
        for history_file in (history_json, history_csv):
            if history_file.exists():
                history_file.unlink()

    callbacks = build_callbacks(
        model_path=model_path,
        metrics_dir=CONFIG.metrics_dir,
        early_stopping_patience=early_stopping_patience,
        reduce_lr_patience=reduce_lr_patience,
        reduce_lr_factor=reduce_lr_factor,
        reduce_lr_min=reduce_lr_min,
    )

    training_config = {
        "architecture": "LeafDoc-Net-inspired DenseNet121 + MobileNetV2",
        "optimizer": "Adam",
        "loss": "categorical_crossentropy",
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "random_seed": seed,
        "freeze_backbones": freeze_backbones,
        "fine_tuning": {
            "enabled": fine_tune,
            "epochs": fine_tune_epochs,
            "learning_rate": fine_tune_learning_rate,
            "densenet_last_layers": densenet_last_layers,
            "mobilenet_last_layers": mobilenet_last_layers,
        },
        "early_stopping": {
            "monitor": "val_loss",
            "patience": early_stopping_patience,
        },
        "reduce_lr_on_plateau": {
            "monitor": "val_loss",
            "patience": reduce_lr_patience,
            "factor": reduce_lr_factor,
            "min_lr": reduce_lr_min,
        },
        "test_set_usage": "Excluded from fit, model selection, and hyperparameter tuning",
        "class_names": class_names,
        "model_report": get_leafdoc_model_report(model),
        "memory_strategy": (
            "Batch size 8, no dataset caching, prefetch, and GPU memory growth "
            "without architectural changes"
        ),
    }
    save_training_config(
        CONFIG.metrics_dir / "leafdoc_training_config.json",
        training_config,
    )
    class_names_path.write_text(
        json.dumps(class_names, indent=2),
        encoding="utf-8",
    )

    LOGGER.info(
        "Starting LeafDoc-Net training: epochs=%d batch_size=%d lr=%g",
        epochs,
        batch_size,
        learning_rate,
    )
    LOGGER.info("The test split will not be evaluated or used in fit().")
    started_at = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=initial_epoch,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    if fine_tune:
        LOGGER.info("Applying configurable upper-backbone fine-tuning.")
        enable_leafdoc_fine_tuning(
            model,
            densenet_last_layers=densenet_last_layers,
            mobilenet_last_layers=mobilenet_last_layers,
            learning_rate=fine_tune_learning_rate,
        )
        fine_initial_epoch = len(history.history["accuracy"]) + initial_epoch
        fine_history = model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=fine_initial_epoch,
            epochs=fine_initial_epoch + fine_tune_epochs,
            callbacks=callbacks,
            verbose=1,
        )
        training_config["fine_tuning"]["completed"] = True
        training_config["fine_tuning"]["epochs_completed"] = len(
            fine_history.history["accuracy"]
        )
    else:
        training_config["fine_tuning"]["completed"] = False
        training_config["fine_tuning"]["epochs_completed"] = 0

    duration = time.perf_counter() - started_at
    training_config["training_duration_seconds"] = round(duration, 3)
    training_config["completed_initial_epochs"] = len(history.history["accuracy"])
    save_training_config(
        CONFIG.metrics_dir / "leafdoc_training_config.json",
        training_config,
    )
    model.save(model_path)
    plot_training_history(
        CONFIG.metrics_dir / "leafdoc_training_history.json",
        CONFIG.plots_dir / "leafdoc_training_history.png",
    )

    LOGGER.info("Training duration: %.2f seconds", duration)
    LOGGER.info("Saved best model to %s", model_path)
    LOGGER.info("Saved class names to %s", class_names_path)
    return model, dict(history.history)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the LeafDoc-Net-inspired hybrid architecture"
    )
    parser.add_argument("--dataset", type=Path, default=CONFIG.dataset_dir)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=CONFIG.seed)
    parser.add_argument("--freeze-backbones", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fine-tune", action="store_true")
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-6)
    parser.add_argument("--densenet-last-layers", type=int, default=30)
    parser.add_argument("--mobilenet-last-layers", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--reduce-lr-patience", type=int, default=2)
    parser.add_argument("--reduce-lr-factor", type=float, default=0.2)
    parser.add_argument("--reduce-lr-min", type=float, default=1e-7)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", type=Path, default=CONFIG.leafdoc_model_path)
    parser.add_argument("--class-names", type=Path, default=CONFIG.class_names_path)
    return parser.parse_args()


if __name__ == "__main__":
    configure_logging()
    arguments = parse_args()
    try:
        train_leafdoc(
            dataset_dir=arguments.dataset,
            model_path=arguments.model,
            class_names_path=arguments.class_names,
            epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate,
            seed=arguments.seed,
            freeze_backbones=arguments.freeze_backbones,
            fine_tune=arguments.fine_tune,
            fine_tune_epochs=arguments.fine_tune_epochs,
            fine_tune_learning_rate=arguments.fine_tune_learning_rate,
            densenet_last_layers=arguments.densenet_last_layers,
            mobilenet_last_layers=arguments.mobilenet_last_layers,
            early_stopping_patience=arguments.early_stopping_patience,
            reduce_lr_patience=arguments.reduce_lr_patience,
            reduce_lr_factor=arguments.reduce_lr_factor,
            reduce_lr_min=arguments.reduce_lr_min,
            resume=arguments.resume,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        LOGGER.error("Training could not start: %s", exc)
        raise SystemExit(1)
