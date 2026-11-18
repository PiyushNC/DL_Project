import numpy as np
from PIL import Image

from src.data_loader import create_splits


def test_splits_are_disjoint_and_stratified(tmp_path):
    for class_index, class_name in enumerate(("Apple", "Tomato")):
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
    )

    assert class_names == ["Apple", "Tomato"]
    assert train and validation and test

    train_paths = {record.path.resolve() for record in train}
    validation_paths = {record.path.resolve() for record in validation}
    test_paths = {record.path.resolve() for record in test}

    assert train_paths.isdisjoint(validation_paths)
    assert train_paths.isdisjoint(test_paths)
    assert validation_paths.isdisjoint(test_paths)
