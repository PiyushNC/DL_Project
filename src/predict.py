import json
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from src.config import CONFIG
from src.data_loader import create_splits
from src.preprocessing import load_validated_image, preprocess_image
from src.model import AttentiveTransition, MobileAttention

LOGGER = logging.getLogger(__name__)


class PredictionService:
    """Load one trained model and provide real-time image predictions."""

    def __init__(
        self,
        model_path: Path,
        class_names_path: Path,
        image_size: tuple[int, int],
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found: {model_path}. Train the model first."
            )
        if not class_names_path.exists():
            raise FileNotFoundError(
                f"Class names not found: {class_names_path}. Train the model first."
            )

        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "AttentiveTransition": AttentiveTransition,
                "MobileAttention": MobileAttention,
            },
        )
        self.image_size = image_size
        self.class_names = json.loads(
            class_names_path.read_text(encoding="utf-8")
        )
        if not isinstance(self.class_names, list) or not self.class_names:
            raise ValueError("The class-label mapping is empty or invalid.")
        output_width = int(self.model.output_shape[-1])
        if output_width != len(self.class_names):
            raise ValueError(
                "Class-label mapping length does not match the model output: "
                f"{len(self.class_names)} labels, {output_width} outputs."
            )
        LOGGER.info("Loaded model from %s", model_path)

    def predict(
        self,
        image: Image.Image,
        top_k: int = 3,
    ) -> tuple[str, float, list[tuple[str, float]]]:
        if image is None:
            raise ValueError("An image is required for prediction.")
        if image.format and image.format.lower() not in {"jpeg", "png", "bmp", "gif", "webp"}:
            raise ValueError(f"Unsupported image format: {image.format}")

        batch = preprocess_image(image, self.image_size)
        probabilities = self.model.predict(batch, verbose=0)[0]
        best_index = int(np.argmax(probabilities))
        confidence = float(probabilities[best_index])
        if len(probabilities) != len(self.class_names):
            raise RuntimeError("Model output width does not match the class mapping.")

        ordered_indexes = np.argsort(probabilities)[::-1][:max(1, top_k)]
        scores = [
            (self.class_names[index], float(probabilities[index]))
            for index in ordered_indexes
        ]
        return self.class_names[best_index], confidence, scores

    def predict_file(self, path_or_file, top_k: int = 3):
        image = load_validated_image(path_or_file)
        return self.predict(image, top_k=top_k)


def predict_test_images(
    model_path: Path,
    class_names_path: Path,
    dataset_dir: Path,
    image_size: tuple[int, int],
    count: int = 3,
) -> list[dict]:
    """Run deterministic inference on the first held-out test images."""
    service = PredictionService(model_path, class_names_path, image_size)
    _, _, test_records, class_names = create_splits(dataset_dir)
    if not test_records:
        raise ValueError("The held-out test split contains no images.")

    results = []
    for record in test_records[:max(1, count)]:
        label, confidence, top_three = service.predict_file(record.path, top_k=3)
        results.append(
            {
                "path": str(record.path),
                "actual_class": record.class_name,
                "predicted_class": label,
                "confidence": confidence,
                "top_three": top_three,
                "correct": label == record.class_name,
                "confidence_disclaimer": "Confidence is a model score, not scientific certainty.",
            }
        )
    LOGGER.info("Completed %d held-out test predictions", len(results))
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test predictions on held-out images")
    parser.add_argument("--dataset", type=Path, default=CONFIG.dataset_dir)
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()

    try:
        test_results = predict_test_images(
            model_path=CONFIG.model_path,
            class_names_path=CONFIG.class_names_path,
            dataset_dir=args.dataset,
            image_size=CONFIG.image_size,
            count=args.count,
        )
        print(json.dumps(test_results, indent=2))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
