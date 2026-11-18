import numpy as np
from PIL import Image

from src.gradcam_pp import (
    grad_cam_plus_plus,
    generate_gradcam_pp,
    overlay_heatmap,
)
from src.model import create_leafdoc_net_model


def test_gradcam_plus_plus_dense_and_mobile_output_dimensions():
    model = create_leafdoc_net_model(
        num_classes=15,
        weights=None,
        freeze_backbones=True,
    )
    image = np.random.default_rng(42).uniform(
        0,
        255,
        size=(1, 224, 224, 3),
    ).astype(np.float32)

    dense_heatmap, dense_features = grad_cam_plus_plus(
        model,
        image,
        class_index=0,
        branch="densenet",
    )
    mobile_heatmap, mobile_features = grad_cam_plus_plus(
        model,
        image,
        class_index=0,
        branch="mobilenet",
    )

    assert dense_heatmap.shape[0] > 0
    assert dense_heatmap.shape == (7, 7)
    assert mobile_heatmap.shape == (7, 7)
    assert dense_features.shape[-1] == 32
    assert mobile_features.shape[-1] == 1280
    assert float(dense_heatmap.min()) >= 0.0
    assert float(dense_heatmap.max()) <= 1.0


def test_gradcam_overlay_preserves_original_dimensions():
    model = create_leafdoc_net_model(
        num_classes=15,
        weights=None,
        freeze_backbones=True,
    )
    original = Image.new("RGB", (320, 180), color=(70, 130, 60))
    heatmap = np.random.default_rng(3).random((7, 7), dtype=np.float32)
    heatmap, overlays = generate_gradcam_pp(
        model,
        original,
        class_index=4,
        branch="both",
    )

    assert heatmap.shape == (224, 224)
    assert set(overlays) == {"densenet", "mobilenet", "both"}
    assert overlays["both"].size == (320, 180)


def test_overlay_heatmap_returns_rgb_image():
    original = Image.new("RGB", (224, 224), color=(100, 150, 80))
    heatmap = np.ones((7, 7), dtype=np.float32)
    overlay = overlay_heatmap(original, heatmap)

    assert overlay.mode == "RGB"
    assert overlay.size == (224, 224)
