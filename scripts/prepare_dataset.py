import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import find_dataset_directory, validate_three_crop_classes


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def count_images(directory: Path) -> dict[str, int]:
    if not directory.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {directory}")

    directory = find_dataset_directory(directory)
    counts: dict[str, int] = {}
    for child in sorted(directory.iterdir()):
        if not child.is_dir():
            continue
        image_count = sum(
            1
            for file in child.rglob("*")
            if file.is_file() and file.suffix.lower() in VALID_EXTENSIONS
        )
        if image_count:
            counts[child.name] = image_count
    if not counts:
        raise ValueError(
            f"No class directories with JPG, JPEG, PNG, or BMP images found in {directory}"
        )
    validate_three_crop_classes(sorted(counts))
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate a downloaded PlantVillage dataset"
    )
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    try:
        counts = count_images(args.dataset)
        validation = validate_three_crop_classes(sorted(counts))
        print(f"Dataset: {args.dataset}")
        print(f"Classes: {len(counts)}")
        print(f"Target classes: {validation['target_class_count']}")
        print(
            "Target 18 verified:",
            validation["target_18_verified"],
        )
        print(f"Images: {sum(counts.values())}")
        for class_name, count in counts.items():
            print(f"{class_name}: {count}")
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
