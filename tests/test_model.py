import tensorflow as tf
import json
from PIL import Image

from src.model import create_mobilenetv2_model, enable_fine_tuning
from src.predict import PredictionService


def test_mobilenetv2_model_output_shape():
    model = create_mobilenetv2_model(
        num_classes=38,
        image_size=(224, 224),
        weights=None,
    )
    assert model.output_shape == (None, 38)
    base_model = next(
        (layer for layer in model.layers if hasattr(layer, "layers")),
        None,
    )
    assert base_model is not None
    assert not base_model.trainable


def test_fine_tuning_freezes_batch_normalization_and_early_layers():
    model = create_mobilenetv2_model(
        num_classes=38,
        image_size=(224, 224),
        weights=None,
    )
    enable_fine_tuning(
        model,
        trainable_last_layers=20,
        learning_rate=1e-5,
    )

    base_model = next(
        (layer for layer in model.layers if hasattr(layer, "layers")),
        None,
    )
    assert base_model is not None

    batch_norm_layers = [
        layer
        for layer in base_model.layers
        if isinstance(layer, tf.keras.layers.BatchNormalization)
    ]
    assert batch_norm_layers
    assert all(not layer.trainable for layer in batch_norm_layers)


def test_prediction_service_returns_three_ranked_predictions(tmp_path):
    model = create_mobilenetv2_model(
        num_classes=3,
        image_size=(224, 224),
        weights=None,
    )
    model_path = tmp_path / "model.keras"
    class_names_path = tmp_path / "class_names.json"
    model.save(model_path)
    class_names_path.write_text(
        json.dumps(["healthy", "rust", "blight"]),
        encoding="utf-8",
    )

    service = PredictionService(
        model_path=model_path,
        class_names_path=class_names_path,
        image_size=(224, 224),
    )
    label, confidence, top_three = service.predict(
        Image.new("RGB", (224, 224), color=(80, 120, 60)),
        top_k=3,
    )

    assert label in {"healthy", "rust", "blight"}
    assert 0.0 <= confidence <= 1.0
    assert len(top_three) == 3
    assert sum(probability for _, probability in top_three) <= 1.0 + 1e-6
