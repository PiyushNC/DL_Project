import tensorflow as tf
import json
import numpy as np
from PIL import Image

from src.model import create_mobilenetv2_model, enable_fine_tuning
from src.predict import PredictionService
from src.model import (
    AttentiveTransition,
    MobileAttention,
    create_leafdoc_net_model,
    enable_leafdoc_fine_tuning,
    get_leafdoc_model_report,
)


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


def test_leafdoc_net_output_shape_and_probability_sum():
    model = create_leafdoc_net_model(
        num_classes=15,
        image_size=(224, 224),
        weights=None,
        freeze_backbones=True,
    )
    images = np.random.default_rng(42).uniform(
        0,
        255,
        size=(2, 224, 224, 3),
    ).astype(np.float32)
    probabilities = model(images, training=False).numpy()

    assert model.output_shape == (None, 15)
    assert probabilities.shape == (2, 15)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)


def test_leafdoc_net_backbones_frozen_and_report_is_consistent():
    model = create_leafdoc_net_model(
        num_classes=15,
        weights=None,
        freeze_backbones=True,
    )
    densenet = model.get_layer("densenet121_base")
    mobilenet = model.get_layer("mobilenetv2_base")

    assert not densenet.trainable
    assert not mobilenet.trainable
    report = get_leafdoc_model_report(model)
    assert report["total_parameters"] > 0
    assert report["trainable_parameters"] + report["non_trainable_parameters"] == report["total_parameters"]


def test_leafdoc_net_fine_tuning_freezes_batch_normalization(tmp_path):
    del tmp_path
    model = create_leafdoc_net_model(
        num_classes=15,
        weights=None,
        freeze_backbones=True,
    )
    enable_leafdoc_fine_tuning(
        model,
        densenet_last_layers=30,
        mobilenet_last_layers=20,
        learning_rate=1e-5,
    )

    for base_name in ("densenet121_base", "mobilenetv2_base"):
        base = model.get_layer(base_name)
        batch_norm_layers = [
            layer
            for layer in base.layers
            if isinstance(layer, tf.keras.layers.BatchNormalization)
        ]
        assert batch_norm_layers
        assert all(not layer.trainable for layer in batch_norm_layers)


def test_leafdoc_net_save_load_and_inference(tmp_path):
    model = create_leafdoc_net_model(
        num_classes=15,
        weights=None,
        freeze_backbones=True,
    )
    model_path = tmp_path / "leafdoc_net.keras"
    model.save(model_path)

    loaded = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "AttentiveTransition": AttentiveTransition,
            "MobileAttention": MobileAttention,
        },
    )
    image = np.random.default_rng(7).uniform(
        0,
        255,
        size=(1, 224, 224, 3),
    ).astype(np.float32)
    probabilities = loaded(image, training=False).numpy()

    assert loaded.output_shape == (None, 15)
    assert probabilities.shape == (1, 15)
    assert np.isclose(probabilities.sum(), 1.0, atol=1e-5)
