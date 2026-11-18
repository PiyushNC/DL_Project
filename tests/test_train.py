import json

from src.train import build_callbacks, save_training_config


def test_leafdoc_callbacks_monitor_validation_loss(tmp_path):
    model_path = tmp_path / "leafdoc.keras"
    metrics_dir = tmp_path / "metrics"
    callbacks = build_callbacks(
        model_path=model_path,
        metrics_dir=metrics_dir,
    )

    early_stopping = next(
        callback
        for callback in callbacks
        if callback.__class__.__name__ == "EarlyStopping"
    )
    checkpoint = next(
        callback
        for callback in callbacks
        if callback.__class__.__name__ == "ModelCheckpoint"
    )
    reduce_lr = next(
        callback
        for callback in callbacks
        if callback.__class__.__name__ == "ReduceLROnPlateau"
    )

    assert early_stopping.monitor == "val_loss"
    assert early_stopping.patience == 3
    assert checkpoint.monitor == "val_loss"
    assert checkpoint.save_best_only
    assert reduce_lr.monitor == "val_loss"


def test_training_config_is_saved_without_test_metrics(tmp_path):
    path = tmp_path / "leafdoc_training_config.json"
    save_training_config(
        path,
        {
            "architecture": "LeafDoc-Net",
            "test_set_usage": "Excluded from fit",
            "learning_rate": 1e-5,
        },
    )

    config = json.loads(path.read_text(encoding="utf-8"))
    assert config["learning_rate"] == 1e-5
    assert "Excluded" in config["test_set_usage"]
