import numpy as np
from PIL import Image

from src.data_loader import create_splits, validate_three_crop_classes
from src.data_loader import _exact_hash
import pytest


def test_splits_are_disjoint_and_stratified(tmp_path):
    for class_index, class_name in enumerate(
        ("Pepper__bell___healthy", "Tomato_healthy")
    ):
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for image_index in range(12):
            rng = np.random.default_rng(class_index * 100 + image_index)
            pixels = rng.integers(
                low=20,
                high=240,
                size=(40, 40, 3),
                dtype=np.uint8,
            )
            pixels[:, :, 1] = np.clip(
                pixels[:, :, 1].astype(int) + 30,
                0,
                255,
            ).astype(np.uint8)
            image = Image.fromarray(pixels, mode="RGB")
            image.save(class_dir / f"leaf_{image_index}.jpg")

    train, validation, test, class_names = create_splits(
        tmp_path,
        seed=42,
        manifest_path=tmp_path / "splits.json",
    )

    assert class_names == ["Pepper__bell___healthy", "Tomato_healthy"]
    assert train and validation and test

    train_paths = {record.path.resolve() for record in train}
    validation_paths = {record.path.resolve() for record in validation}
    test_paths = {record.path.resolve() for record in test}

    assert train_paths.isdisjoint(validation_paths)
    assert train_paths.isdisjoint(test_paths)
    assert validation_paths.isdisjoint(test_paths)


def test_target_count_is_reported_without_false_18_class_claim():
    result = validate_three_crop_classes([
        "Pepper__bell___Bacterial_spot",
        "Pepper__bell___healthy",
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Potato___healthy",
        "Tomato_Bacterial_spot",
        "Tomato_Early_blight",
        "Tomato_Late_blight",
        "Tomato_Leaf_Mold",
        "Tomato_Septoria_leaf_spot",
        "Tomato_Spider_mites_Two_spotted_spider_mite",
        "Tomato__Target_Spot",
        "Tomato__Tomato_YellowLeaf__Curl_Virus",
        "Tomato__Tomato_mosaic_virus",
        "Tomato_healthy",
    ])
    assert result["detected_class_count"] == 15
    assert result["target_class_count"] == 18
    assert result["target_18_verified"] is False


def test_unexpected_crop_class_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unexpected classes"):
        validate_three_crop_classes([
            "Apple___Apple_scab",
            "Tomato_healthy",
        ])


def test_exact_duplicate_is_removed_from_split_records(tmp_path):
    class_dir = tmp_path / "Tomato_healthy"
    class_dir.mkdir()
    image = Image.new("RGB", (40, 40), (50, 120, 70))
    first = class_dir / "first.jpg"
    second = class_dir / "second.jpg"
    image.save(first)
    image.save(second)

    assert _exact_hash(first) == _exact_hash(second)
