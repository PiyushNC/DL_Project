import hashlib
import json
import logging
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from src.config import CONFIG
from src.preprocessing import create_training_augmentation

LOGGER = logging.getLogger(__name__)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def find_dataset_directory(dataset_dir: Path) -> Path:
    """Find an image-class directory, including nested extracted Kaggle layouts."""
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}. "
            "Download and extract PlantVillage before training."
        )

    candidates: list[tuple[int, Path]] = []
    for candidate in [dataset_dir, *dataset_dir.rglob("*")]:
        if not candidate.is_dir():
            continue
        try:
            children = [
                child
                for child in candidate.iterdir()
                if child.is_dir() and not child.name.startswith(".")
            ]
        except OSError:
            continue
        if len(children) < 2:
            continue

        if all(
            any(
                file.is_file()
                and file.suffix.lower() in VALID_EXTENSIONS
                for file in child.rglob("*")
            )
            for child in children
        ):
            candidates.append((len(candidate.relative_to(dataset_dir).parts), candidate))

    if candidates:
        _, resolved = max(candidates, key=lambda item: item[0])
        LOGGER.info("Resolved PlantVillage class root: %s", resolved)
        return resolved

    raise FileNotFoundError(
        f"No PlantVillage-style class folders containing images were found in {dataset_dir}"
    )


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    class_name: str
    class_index: int
    exact_hash: str
    perceptual_hash: int


def _exact_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _difference_hash(path: Path) -> int:
    with Image.open(path) as image:
        image = image.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.uint8)

    value = 0
    for row in pixels:
        for left, right in zip(row[:-1], row[1:]):
            value = (value << 1) | int(left > right)
    return int(value)


def _hamming_distance(left: int, right: int) -> int:
    return int((left ^ right).bit_count())


def _group_near_duplicates(records: list[ImageRecord], threshold: int = 5) -> list[list[ImageRecord]]:
    """Keep near-identical images together so one split does not leak them."""
    buckets: dict[int, list[int]] = {}
    representative_groups: list[list[ImageRecord]] = []

    for record in records:
        bucket = record.perceptual_hash >> 48
        candidate_indexes = buckets.setdefault(bucket, [])
        matching_group = None

        for representative_index in candidate_indexes:
            representative = representative_groups[representative_index][0]
            if _hamming_distance(record.perceptual_hash, representative.perceptual_hash) <= threshold:
                matching_group = representative_index
                break

        if matching_group is None:
            matching_group = len(representative_groups)
            representative_groups.append([])
            candidate_indexes.append(matching_group)

        representative_groups[matching_group].append(record)

    return representative_groups


def inspect_dataset_directory(dataset_dir: Path) -> tuple[list[str], list[Path], Counter]:
    """Validate class folders and collect usable image files without data leakage."""
    dataset_dir = find_dataset_directory(dataset_dir)
    class_names = sorted(
        child.name
        for child in dataset_dir.iterdir()
        if child.is_dir() and not child.name.startswith(".")
    )
    if not class_names:
        raise ValueError(f"No class folders were found in {dataset_dir}")

    usable: list[Path] = []
    class_counts: Counter = Counter()
    unsupported_count = 0
    corrupt_count = 0

    for class_name in class_names:
        class_dir = dataset_dir / class_name
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in VALID_EXTENSIONS:
                unsupported_count += 1
                LOGGER.warning("Skipping unsupported file: %s", path)
                continue

            try:
                with Image.open(path) as image:
                    image.verify()
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                corrupt_count += 1
                LOGGER.warning("Skipping corrupted image %s: %s", path, exc)
                continue

            usable.append(path)
            class_counts[class_name] += 1

    if unsupported_count:
        LOGGER.warning(
            "Skipped %d unsupported files in total.", unsupported_count
        )
    if corrupt_count:
        LOGGER.warning("Skipped %d corrupted images in total.", corrupt_count)
    if not usable:
        raise ValueError(
            f"No usable supported images were found in {dataset_dir}. "
            f"Unsupported: {unsupported_count}, corrupted: {corrupt_count}"
        )

    LOGGER.info(
        "Validated %d usable images across %d classes",
        len(usable),
        len(class_names),
    )
    return class_names, usable, class_counts


def create_splits(
    dataset_dir: Path,
    seed: int = CONFIG.seed,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> tuple[list[ImageRecord], list[ImageRecord], list[ImageRecord], list[str]]:
    """Create deterministic, duplicate-safe stratified file splits."""
    if train_ratio + validation_ratio >= 1.0:
        raise ValueError("train_ratio and validation_ratio must sum to less than 1.")

    dataset_dir = find_dataset_directory(dataset_dir)
    class_names, usable_paths, _ = inspect_dataset_directory(dataset_dir)
    rng = random.Random(seed)
    split_manifest: dict[str, dict[str, list[str]]] = {}
    train_records: list[ImageRecord] = []
    val_records: list[ImageRecord] = []
    test_records: list[ImageRecord] = []

    for class_index, class_name in enumerate(class_names):
        class_paths = [
            path for path in usable_paths
            if path.parent.name == class_name
        ]
        exact_hashes: set[str] = set()
        records: list[ImageRecord] = []

        for path in class_paths:
            try:
                exact_hash = _exact_hash(path)
                if exact_hash in exact_hashes:
                    LOGGER.warning(
                        "Skipping exact duplicate in class %s: %s",
                        class_name,
                        path,
                    )
                    continue
                exact_hashes.add(exact_hash)
                perceptual_hash = _difference_hash(path)
            except (OSError, ValueError, UnidentifiedImageError) as exc:
                LOGGER.warning(
                    "Skipping image that failed duplicate detection %s: %s",
                    path,
                    exc,
                )
                continue

            records.append(
                ImageRecord(
                    path=path,
                    class_name=class_name,
                    class_index=class_index,
                    exact_hash=exact_hash,
                    perceptual_hash=perceptual_hash,
                )
            )

        groups = _group_near_duplicates(records)
        rng.shuffle(groups)
        if len(groups) < 3:
            raise ValueError(
                f"Class '{class_name}' has only {len(groups)} duplicate-safe "
                "groups. At least 3 groups are required for 70/15/15 splits."
            )

        train_count = int(len(groups) * train_ratio)
        val_count = int(len(groups) * validation_ratio)
        train_count = max(1, train_count)
        val_count = max(1, val_count)
        if train_count + val_count >= len(groups):
            val_count = max(1, len(groups) - train_count - 1)

        train_groups = groups[:train_count]
        val_groups = groups[train_count:train_count + val_count]
        test_groups = groups[train_count + val_count:]

        for destination, destination_groups in (
            (train_records, train_groups),
            (val_records, val_groups),
            (test_records, test_groups),
        ):
            for group in destination_groups:
                destination.extend(group)

        split_manifest[class_name] = {
            "train": [
                str(record.path)
                for group in train_groups
                for record in group
            ],
            "validation": [
                str(record.path)
                for group in val_groups
                for record in group
            ],
            "test": [
                str(record.path)
                for group in test_groups
                for record in group
            ],
        }

        LOGGER.info(
            "Class %s: train=%d, validation=%d, test=%d",
            class_name,
            sum(len(group) for group in train_groups),
            sum(len(group) for group in val_groups),
            sum(len(group) for group in test_groups),
        )

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    rng.shuffle(test_records)

    CONFIG.splits_path.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.splits_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "train_ratio": train_ratio,
                "validation_ratio": validation_ratio,
                "classes": class_names,
                "splits": split_manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    LOGGER.info("Saved split manifest to %s", CONFIG.splits_path)

    train_paths = {record.path.resolve() for record in train_records}
    val_paths = {record.path.resolve() for record in val_records}
    test_paths = {record.path.resolve() for record in test_records}
    if train_paths & val_paths or train_paths & test_paths or val_paths & test_paths:
        raise RuntimeError("Split overlap detected. No dataset should be shared across splits.")

    LOGGER.info(
        "Split sizes: train=%d, validation=%d, test=%d",
        len(train_records),
        len(val_records),
        len(test_records),
    )
    return train_records, val_records, test_records, class_names


def create_datasets(
    dataset_dir: Path,
    image_size: tuple[int, int] | None = None,
    batch_size: int = CONFIG.batch_size,
    seed: int = CONFIG.seed,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    """Create deterministic 70/15/15 train, validation, and test datasets."""
    image_size = image_size or CONFIG.image_size
    train_records, val_records, test_records, class_names = create_splits(
        dataset_dir=dataset_dir,
        seed=seed,
    )
    num_classes = len(class_names)

    def decode_record(path: tf.Tensor, label: tf.Tensor):
        image = tf.io.read_file(path)
        image = tf.io.decode_image(
            image,
            channels=3,
            expand_animations=False,
        )
        image.set_shape((None, None, 3))
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32)
        return image, tf.one_hot(label, depth=num_classes)

    def make_dataset(records: list[ImageRecord], shuffle: bool):
        paths = [str(record.path) for record in records]
        labels = [record.class_index for record in records]
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                tf.constant(paths),
                tf.constant(labels, dtype=tf.int32),
            )
        )
        if shuffle:
            dataset = dataset.shuffle(
                buffer_size=len(records),
                seed=seed,
                reshuffle_each_iteration=True,
            )
        dataset = dataset.map(
            decode_record,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = make_dataset(train_records, shuffle=True)
    val_ds = make_dataset(val_records, shuffle=False)
    test_ds = make_dataset(test_records, shuffle=False)
    augmentation = create_training_augmentation(seed)

    def augment_images(images, labels):
        return augmentation(images, training=True), labels

    train_ds = train_ds.map(
        augment_images,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    train_counts = Counter(record.class_name for record in train_records)
    val_counts = Counter(record.class_name for record in val_records)
    test_counts = Counter(record.class_name for record in test_records)
    LOGGER.info("Per-class samples:")
    for class_name in class_names:
        LOGGER.info(
            "%s | train=%d | validation=%d | test=%d",
            class_name,
            train_counts[class_name],
            val_counts[class_name],
            test_counts[class_name],
        )

    LOGGER.info(
        "Prepared datasets from %s with %d classes",
        find_dataset_directory(dataset_dir),
        len(class_names),
    )
    return train_ds, val_ds, test_ds, class_names
