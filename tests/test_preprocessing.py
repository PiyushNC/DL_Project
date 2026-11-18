from PIL import Image
import pytest
import numpy as np

from src.preprocessing import (
    load_validated_image,
    preprocess_for_model,
    preprocess_image,
)


def test_preprocess_image_has_mobilenetv2_shape():
    image = Image.new("RGB", (500, 300), color=(120, 80, 40))
    batch = preprocess_image(image, (224, 224))
    assert batch.shape == (1, 224, 224, 3)


def test_invalid_image_extension_is_rejected(tmp_path):
    invalid = tmp_path / "notes.txt"
    invalid.write_text("not an image", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported|valid supported"):
        load_validated_image(invalid)


def test_dense_and_mobilenet_preprocessing_are_distinct():
    pixels = np.full((224, 224, 3), 128.0, dtype=np.float32)
    mobile = preprocess_for_model(pixels, "mobilenetv2")
    dense = preprocess_for_model(pixels, "densenet121")

    assert mobile.dtype == np.float32
    assert dense.dtype == np.float32
    assert not np.allclose(mobile, dense)


def test_preprocess_image_converts_grayscale_to_rgb_and_resizes():
    grayscale = Image.new("L", (640, 480), color=100)
    batch = preprocess_image(grayscale, (224, 224))

    assert batch.shape == (1, 224, 224, 3)


def test_corrupted_image_is_rejected(tmp_path):
    corrupt = tmp_path / "broken.jpg"
    corrupt.write_bytes(b"not a jpeg image")

    with pytest.raises(ValueError, match="valid supported"):
        load_validated_image(corrupt)
