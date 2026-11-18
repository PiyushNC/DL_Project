import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def create_training_augmentation(seed: int) -> tf.keras.Sequential:
    """Create augmentation applied only to training samples."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(0.08, seed=seed),
            tf.keras.layers.RandomZoom(0.08, seed=seed),
        ],
        name="training_augmentation",
    )


def preprocess_for_model(
    pixels: np.ndarray,
    model_name: str,
) -> np.ndarray:
    """Apply the framework preprocessing required by a supported backbone."""
    normalized_name = model_name.strip().lower().replace("-", "").replace("_", "")
    pixels = np.asarray(pixels, dtype=np.float32)

    if normalized_name in {"mobilenetv2", "mobilenet"}:
        return np.asarray(
            tf.keras.applications.mobilenet_v2.preprocess_input(pixels)
        )
    if normalized_name in {"densenet121", "densenet"}:
        scaled = pixels / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        return (scaled - mean) / std

    raise ValueError(
        f"Unsupported model preprocessing '{model_name}'. "
        "Supported values: mobilenetv2, densenet121."
    )


def preprocess_image(image: Image.Image, image_size: tuple[int, int]) -> np.ndarray:
    """Convert a PIL image to a raw inference batch for the model pipeline."""
    if image is None:
        raise ValueError("An image is required.")
    if not hasattr(image, "size"):
        raise ValueError("Invalid PIL image.")

    if image.mode != "RGB":
        image = image.convert("RGB")

    resized = image.resize(image_size, Image.Resampling.BILINEAR)
    pixels = tf.keras.preprocessing.image.img_to_array(resized)
    return np.expand_dims(pixels, axis=0)


def load_validated_image(path_or_file) -> Image.Image:
    """Open and verify an uploaded or test image format."""
    name = getattr(path_or_file, "name", None)
    if name is not None:
        name = str(name)
        suffix = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format '{suffix or 'unknown'}'. "
                f"Use: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}"
            )

    try:
        image = Image.open(path_or_file)
        image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"The file is not a valid supported image: {exc}") from exc

    if hasattr(path_or_file, "seek"):
        path_or_file.seek(0)
    image = Image.open(path_or_file)
    image.load()
    return image.convert("RGB")
