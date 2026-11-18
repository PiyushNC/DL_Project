from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ProjectConfig:
    image_height: int = 224
    image_width: int = 224
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 15
    fine_tune_learning_rate: float = 1e-5
    fine_tune_epochs: int = 5
    fine_tune_last_layers: int = 20
    dropout_rate: float = 0.25
    early_stopping_patience: int = 3
    reduce_lr_patience: int = 2
    reduce_lr_factor: float = 0.2
    reduce_lr_min: float = 1e-7
    seed: int = 42
    dataset_dir: Path = PROJECT_ROOT / "data" / "raw"
    model_dir: Path = PROJECT_ROOT / "outputs" / "models"
    plots_dir: Path = PROJECT_ROOT / "outputs" / "plots"
    metrics_dir: Path = PROJECT_ROOT / "outputs" / "metrics"
    splits_path: Path = PROJECT_ROOT / "data" / "splits.json"
    model_filename: str = "plant_disease_mobilenetv2.keras"
    class_names_filename: str = "class_names.json"

    @property
    def image_size(self) -> tuple[int, int]:
        return (self.image_height, self.image_width)

    @property
    def model_path(self) -> Path:
        return self.model_dir / self.model_filename

    @property
    def class_names_path(self) -> Path:
        return self.model_dir / self.class_names_filename


CONFIG = ProjectConfig()
