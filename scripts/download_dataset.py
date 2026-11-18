import argparse
import subprocess
import sys
from pathlib import Path


def download(dataset_slug: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        "kaggle",
        "datasets",
        "download",
        "--dataset",
        dataset_slug,
        "--path",
        str(destination),
        "--unzip",
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "The Kaggle CLI was not found. Install dependencies and configure "
            "~/.kaggle/kaggle.json or KAGGLE_API_TOKEN."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Kaggle download failed with exit code {exc.returncode}. "
            "Verify your Kaggle authentication and dataset access."
        ) from exc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PlantVillage from Kaggle")
    parser.add_argument(
        "--dataset",
        default="emmarex/plantdisease",
        help="Kaggle dataset slug",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/raw"),
    )
    args = parser.parse_args()
    try:
        download(args.dataset, args.destination)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
