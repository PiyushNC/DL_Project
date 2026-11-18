import numpy as np

from src.evaluate import _safe_roc_auc, save_confusion_matrix


def test_multiclass_roc_helper_handles_missing_class():
    y_true = np.array([0, 0, 1, 1])
    y_probability = np.array([
        [0.8, 0.2, 0.0],
        [0.7, 0.3, 0.0],
        [0.4, 0.6, 0.0],
        [0.2, 0.8, 0.0],
    ])

    macro_auc, micro_auc, curves = _safe_roc_auc(
        y_true,
        y_probability,
        num_classes=3,
    )
    assert curves["2"]["undefined_reason"]
    assert macro_auc is not None
    assert micro_auc is not None


def test_confusion_matrix_save_outputs(tmp_path):
    cm = np.array([[3, 1], [2, 4]])
    plot_dir = tmp_path / "plots"

    plot_path = save_confusion_matrix(
        cm,
        ["healthy", "diseased"],
        plot_dir,
    )

    assert plot_path.exists()
    assert (plot_dir / "confusion_matrix.npy").exists()
    assert (plot_dir / "confusion_matrix.csv").exists()
