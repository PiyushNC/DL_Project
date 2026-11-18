import argparse
import json
import logging
from pathlib import Path

from matplotlib import colormaps
import numpy as np
import tensorflow as tf
from PIL import Image

from src.config import CONFIG
from src.data_loader import create_splits
from src.model import AttentiveTransition, MobileAttention
from src.preprocessing import load_validated_image

LOGGER = logging.getLogger(__name__)

SUPPORTED_BRANCHES = {"densenet", "mobilenet", "both"}


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_leafdoc_model(model_path: Path) -> tf.keras.Model:
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found: {model_path}")
    return tf.keras.models.load_model(
        model_path,
        custom_objects={
            "AttentiveTransition": AttentiveTransition,
            "MobileAttention": MobileAttention,
        },
    )


def find_branch_feature_layer(model: tf.keras.Model, branch: str) -> tuple[tf.keras.Model, str]:
    """Locate a valid final convolutional feature map in a backbone."""
    if branch == "densenet":
        backbone_name = "densenet121_base"
        preferred_names = ("conv5_block16_2_conv",)
    elif branch == "mobilenet":
        backbone_name = "mobilenetv2_base"
        preferred_names = ("Conv_1", "Conv_1_bn", "Conv_1_relu")
    else:
        raise ValueError("branch must be densenet or mobilenet")

    backbone = model.get_layer(backbone_name)
    for name in preferred_names:
        try:
            layer = backbone.get_layer(name)
            if isinstance(
                layer,
                (
                    tf.keras.layers.Conv2D,
                    tf.keras.layers.BatchNormalization,
                    tf.keras.layers.ReLU,
                ),
            ):
                return backbone, name
        except ValueError:
            continue

    convolutional_layers = [
        layer
        for layer in backbone.layers
        if isinstance(layer, tf.keras.layers.Conv2D)
    ]
    if not convolutional_layers:
        raise ValueError(f"No convolutional feature layer found in {backbone_name}")
    fallback = convolutional_layers[-1]
    LOGGER.warning(
        "Preferred %s feature layer was unavailable; using fallback %s.",
        backbone_name,
        fallback.name,
    )
    return backbone, fallback.name


def build_gradcam_gradient_model(
    model: tf.keras.Model,
    branch: str,
) -> tuple[tf.keras.Model, str]:
    """Build one connected graph for Grad-CAM++ gradient calculation."""
    if branch not in {"densenet", "mobilenet"}:
        raise ValueError("branch must be densenet or mobilenet")

    inputs = tf.keras.Input(
        shape=tuple(model.input_shape[1:]),
        name="gradcam_input",
    )
    densenet = model.get_layer("densenet121_base")
    mobilenet = model.get_layer("mobilenetv2_base")

    dense_feature_name = "conv5_block16_2_conv"
    mobile_feature_name = "Conv_1"
    try:
        dense_feature_output = densenet.get_layer(dense_feature_name).output
    except ValueError:
        _, dense_feature_name = find_branch_feature_layer(model, "densenet")
        dense_feature_output = densenet.get_layer(dense_feature_name).output
    try:
        mobile_feature_output = mobilenet.get_layer(mobile_feature_name).output
    except ValueError:
        _, mobile_feature_name = find_branch_feature_layer(model, "mobilenet")
        mobile_feature_output = mobilenet.get_layer(mobile_feature_name).output

    dense_branch_model = tf.keras.Model(
        inputs=densenet.input,
        outputs=[densenet.output, dense_feature_output],
        name="gradcam_densenet_branch",
    )
    mobile_branch_model = tf.keras.Model(
        inputs=mobilenet.input,
        outputs=[mobilenet.output, mobile_feature_output],
        name="gradcam_mobilenet_branch",
    )

    dense_preprocessed = tf.keras.applications.densenet.preprocess_input(inputs)
    dense_output, dense_feature_maps = dense_branch_model(
        dense_preprocessed,
        training=False,
    )
    dense_output = model.get_layer("densenet_attentive_transition")(
        dense_output,
        training=False,
    )
    dense_output = tf.keras.layers.GlobalAveragePooling2D(
        name="gradcam_densenet_gap"
    )(dense_output)

    mobile_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    mobile_output, mobile_feature_maps = mobile_branch_model(
        mobile_preprocessed,
        training=False,
    )
    mobile_output = model.get_layer("mobilenet_sigmoid_attention")(
        mobile_output,
        training=False,
    )
    mobile_output = tf.keras.layers.GlobalAveragePooling2D(
        name="gradcam_mobilenet_gap"
    )(mobile_output)

    fused = tf.keras.layers.Concatenate(
        name="gradcam_feature_fusion"
    )([dense_output, mobile_output])
    x = model.get_layer("dense_1024")(fused)
    x = model.get_layer("swish_1024")(x)
    x = model.get_layer("batch_norm_1024")(x, training=False)
    x = model.get_layer("dense_512")(x)
    x = model.get_layer("swish_512")(x)
    x = model.get_layer("batch_norm_512")(x, training=False)
    x = model.get_layer("dense_128")(x)
    x = model.get_layer("swish_128")(x)
    x = model.get_layer("batch_norm_128")(x, training=False)
    x = model.get_layer("classifier_dropout")(x, training=False)
    output_layer = model.get_layer("leafdoc_output")
    predictions = tf.keras.ops.matmul(x, output_layer.kernel)
    if output_layer.bias is not None:
        predictions = predictions + output_layer.bias

    selected_feature_maps = (
        dense_feature_maps
        if branch == "densenet"
        else mobile_feature_maps
    )
    feature_name = dense_feature_name if branch == "densenet" else mobile_feature_name
    gradient_model = tf.keras.Model(
        inputs=inputs,
        outputs=[selected_feature_maps, predictions],
        name=f"gradcam_pp_{branch}",
    )
    return gradient_model, feature_name


def grad_cam_plus_plus(
    model: tf.keras.Model,
    image_batch: np.ndarray,
    class_index: int,
    branch: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate Grad-CAM++ weights for one backbone and class."""
    if branch not in {"densenet", "mobilenet"}:
        raise ValueError("Grad-CAM++ branch must be densenet or mobilenet")
    if image_batch.ndim != 4:
        raise ValueError("image_batch must have shape (1, height, width, 3)")
    if class_index < 0 or class_index >= int(model.output_shape[-1]):
        raise ValueError("class_index is outside the model output range")

    with tf.GradientTape(persistent=False) as tape:
        image_tensor = tf.cast(image_batch, tf.float32)
        tape.watch(image_tensor)
        gradient_model, feature_layer_name = build_gradcam_gradient_model(model, branch)
        feature_maps, predictions = gradient_model(image_tensor, training=False)
        tape.watch(feature_maps)
        target_score = predictions[:, class_index]

    gradients = tape.gradient(target_score, feature_maps)
    if gradients is None:
        raise RuntimeError(
            f"Could not calculate gradients for {branch} feature layer "
            f"{feature_layer_name}"
        )

    gradients = tf.nn.relu(gradients)
    gradient_squared = gradients * gradients
    gradient_cubed = gradient_squared * gradients
    spatial_sum = tf.reduce_sum(feature_maps * gradient_cubed, axis=(1, 2))
    spatial_sum = tf.reshape(
        spatial_sum,
        (
            tf.shape(feature_maps)[0],
            1,
            1,
            tf.shape(feature_maps)[-1],
        ),
    )
    alpha = gradient_squared / (
        2.0 * gradient_squared + spatial_sum + 1e-8
    )
    weights = tf.reduce_sum(
        alpha * tf.nn.relu(gradients),
        axis=(1, 2),
    )
    weights = tf.reshape(
        weights,
        (
            tf.shape(feature_maps)[0],
            1,
            1,
            tf.shape(feature_maps)[-1],
        ),
    )
    heatmap = tf.reduce_sum(weights * feature_maps, axis=-1)
    heatmap = tf.nn.relu(heatmap)

    heatmap_array = heatmap[0].numpy()
    maximum = float(np.max(heatmap_array))
    if maximum > 0:
        heatmap_array = heatmap_array / maximum
    else:
        heatmap_array = np.zeros_like(heatmap_array, dtype=np.float32)

    return heatmap_array, np.asarray(feature_maps[0].numpy())


def resize_heatmap(
    heatmap: np.ndarray,
    image_size: tuple[int, int],
) -> np.ndarray:
    heatmap_image = Image.fromarray(
        np.uint8(np.clip(heatmap, 0.0, 1.0) * 255),
        mode="L",
    ).resize(image_size, Image.Resampling.BILINEAR)
    return np.asarray(heatmap_image, dtype=np.float32) / 255.0


def overlay_heatmap(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    if original_image.mode != "RGB":
        original_image = original_image.convert("RGB")
    original_array = np.asarray(original_image, dtype=np.uint8)
    resized = resize_heatmap(heatmap, (original_image.width, original_image.height))
    colored = np.asarray(
        colormaps["jet"](resized)[..., :3] * 255,
        dtype=np.uint8,
    )
    blended = np.uint8(
        (1.0 - alpha) * original_array + alpha * colored
    )
    return Image.fromarray(blended, mode="RGB")


def generate_gradcam_pp(
    model: tf.keras.Model,
    original_image: Image.Image,
    class_index: int,
    branch: str = "both",
    alpha: float = 0.45,
):
    """Generate Grad-CAM++ overlays for DenseNet, MobileNet, or both."""
    if branch not in SUPPORTED_BRANCHES:
        raise ValueError(
            f"Unsupported branch '{branch}'. Use densenet, mobilenet, or both."
        )
    if original_image.mode != "RGB":
        original_image = original_image.convert("RGB")

    image_resized = original_image.resize((224, 224), Image.Resampling.BILINEAR)
    batch = np.expand_dims(
        np.asarray(image_resized, dtype=np.float32),
        axis=0,
    )

    if branch == "both":
        dense_heatmap, _ = grad_cam_plus_plus(
            model,
            batch,
            class_index,
            "densenet",
        )
        mobile_heatmap, _ = grad_cam_plus_plus(
            model,
            batch,
            class_index,
            "mobilenet",
        )
        dense_resized = resize_heatmap(dense_heatmap, (224, 224))
        mobile_resized = resize_heatmap(mobile_heatmap, (224, 224))
        heatmap = (dense_resized + mobile_resized) / 2.0
        heatmap = heatmap / max(float(np.max(heatmap)), 1e-8)
        overlays = {
            "densenet": overlay_heatmap(original_image, dense_heatmap, alpha),
            "mobilenet": overlay_heatmap(original_image, mobile_heatmap, alpha),
            "both": overlay_heatmap(original_image, heatmap, alpha),
        }
        return heatmap, overlays

    heatmap, _ = grad_cam_plus_plus(
        model,
        batch,
        class_index,
        branch,
    )
    return heatmap, {
        branch: overlay_heatmap(original_image, heatmap, alpha),
    }


def save_test_gradcam_samples(
    model_path: Path,
    class_names_path: Path,
    dataset_dir: Path,
    output_dir: Path,
    count: int = 5,
    class_index: int | None = None,
    branch: str = "both",
) -> list[dict]:
    model = load_leafdoc_model(model_path)
    class_names = json.loads(class_names_path.read_text(encoding="utf-8"))
    _, _, test_records, dataset_classes = create_splits(dataset_dir)
    if class_names != dataset_classes:
        raise RuntimeError("Class mapping does not match the held-out test split.")
    if not test_records:
        raise ValueError("The held-out test split has no samples.")

    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for sample_index, record in enumerate(test_records[:max(1, count)]):
        original = load_validated_image(record.path)
        batch = np.expand_dims(
            np.asarray(
                original.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR),
                dtype=np.float32,
            ),
            axis=0,
        )
        probabilities = model(batch, training=False).numpy()[0]
        predicted_index = int(np.argmax(probabilities))
        selected_index = predicted_index if class_index is None else class_index
        if selected_index < 0 or selected_index >= len(class_names):
            raise ValueError("The selected class index is invalid.")

        _, overlays = generate_gradcam_pp(
            model,
            original,
            selected_index,
            branch=branch,
        )
        stem = f"{sample_index:03d}_{record.class_name}"
        saved = {}
        for branch_name, overlay in overlays.items():
            path = output_dir / f"{stem}_{branch_name}.png"
            overlay.save(path)
            saved[branch_name] = str(path)

        records.append({
            "source": str(record.path),
            "actual_class": record.class_name,
            "predicted_class": class_names[predicted_index],
            "selected_class": class_names[selected_index],
            "predicted_confidence": float(probabilities[predicted_index]),
            "saved_overlays": saved,
            "feature_layers": {
                "densenet": "conv5_block16_2_conv",
                "mobilenet": "Conv_1",
            },
        })
    LOGGER.info("Saved %d Grad-CAM++ explainability samples", len(records))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM++ overlays for the LeafDoc-Net hybrid"
    )
    parser.add_argument("--dataset", type=Path, default=CONFIG.dataset_dir)
    parser.add_argument("--model", type=Path, default=CONFIG.leafdoc_model_path)
    parser.add_argument("--class-names", type=Path, default=CONFIG.class_names_path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--class-name")
    parser.add_argument("--class-index", type=int)
    parser.add_argument("--branch", choices=sorted(SUPPORTED_BRANCHES), default="both")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--output", type=Path, default=CONFIG.plots_dir / "gradcampp")
    return parser


if __name__ == "__main__":
    configure_logging()
    arguments = parse_args().parse_args()
    try:
        model = load_leafdoc_model(arguments.model)
        class_names = json.loads(arguments.class_names.read_text(encoding="utf-8"))

        if arguments.class_name:
            if arguments.class_name not in class_names:
                raise ValueError(
                    f"Unknown class '{arguments.class_name}'. "
                    f"Available: {', '.join(class_names)}"
                )
            selected_index = class_names.index(arguments.class_name)
        elif arguments.class_index is not None:
            selected_index = arguments.class_index
        else:
            selected_index = None

        if arguments.image:
            original = load_validated_image(arguments.image)
            batch = np.expand_dims(
                np.asarray(
                    original.resize((224, 224), Image.Resampling.BILINEAR),
                    dtype=np.float32,
                ),
                axis=0,
            )
            probabilities = model(batch, training=False).numpy()[0]
            predicted_index = int(np.argmax(probabilities))
            selected_index = predicted_index if selected_index is None else selected_index
            if selected_index < 0 or selected_index >= len(class_names):
                raise ValueError("The selected class index is invalid.")
            _, overlays = generate_gradcam_pp(
                model,
                original,
                selected_index,
                branch=arguments.branch,
            )
            arguments.output.mkdir(parents=True, exist_ok=True)
            saved = {}
            for branch_name, overlay in overlays.items():
                path = arguments.output / f"{Path(arguments.image).stem}_{branch_name}.png"
                overlay.save(path)
                saved[branch_name] = str(path)
            print(json.dumps({
                "predicted_class": class_names[predicted_index],
                "predicted_confidence": float(probabilities[predicted_index]),
                "selected_class": class_names[selected_index],
                "saved_overlays": saved,
            }, indent=2))
        else:
            records = save_test_gradcam_samples(
                model_path=arguments.model,
                class_names_path=arguments.class_names,
                dataset_dir=arguments.dataset,
                output_dir=arguments.output,
                count=arguments.count,
                class_index=selected_index,
                branch=arguments.branch,
            )
            print(json.dumps(records, indent=2))
    except (FileNotFoundError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        LOGGER.error("Grad-CAM++ could not start: %s", exc)
        raise SystemExit(1)
