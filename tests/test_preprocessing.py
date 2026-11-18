from PIL import Image
import pytest

from src.preprocessing import load_validated_image, preprocess_image


def test_preprocess_image_has_mobilenetv2_shape():
    image = Image.new("RGB", (500, 300), color=(120, 80, 40))
    batch = preprocess_image(image, (224, 224))
    assert batch.shape == (1, 224, 224, 3)


def test_invalid_image_extension_is_rejected(tmp_path):
    invalid = tmp_path / "notes.txt"
    invalid.write_text("not an image", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported|valid supported"):
        load_validated_image(invalid)
