import argparse
import logging
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
from PIL import Image

from src.config import CONFIG
from src.data_loader import (
    create_splits,
    inspect_dataset_directory,
)


def display_samples(
    dataset_dir: Path,
    samples_per_class: int = 3,
    output_path: Path | None = None,
) -> None:
    class_names, _, class_counts = inspect_dataset_directory(dataset_dir)
    train_records, val_records, test_records, _ = create_splits(
        dataset_dir=dataset_dir,
        seed=CONFIG.seed,
    )

    print(f"Dataset: {dataset_dir}")
    print(f"Classes: {len(class_names)}")
    print(f"Usable images: {sum(class_counts.values())}")
    print(
        f"Split sizes: train={len(train_records)}, "
        f"validation={len(val_records)}, test={len(test_records)}"
    )

    selected = []
    all_records = train_records + val_records + test_records
    for class_name in class_names:
        class_records = [
            record for record in all_records
            if record.class_name == class_name
        ]
        selected.extend(class_records[:samples_per_class])

    if not selected:
        raise ValueError("No valid sample images are available.")

    columns = min(samples_per_class, 4)
    rows = math.ceil(len(selected) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 3, rows * 3))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for axis, record in zip(axes, selected):
        with Image.open(record.path) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(record.class_name)
        axis.axis("off")
    for axis in axes[len(selected):]:
        axis.axis("off")

    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"Saved sample grid to {output_path}")
    else:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Inspect a PlantVillage dataset and display sample images"
    )
    parser.add_argument("dataset", type=Path, nargs="?", default=CONFIG.dataset_dir)
    parser.add_argument("--samples-per-class", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    display_samples(
        dataset_dir=args.dataset,
        samples_per_class=args.samples_per_class,
        output_path=args.output,
    )
