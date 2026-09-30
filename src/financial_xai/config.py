"""Configuration settings for the project."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = project_root / "data"
    notebooks_dir: Path = project_root / "notebooks"
    models_dir: Path = project_root / "models"
    logs_dir: Path = project_root / "logs"


DEFAULT_CONFIG = Config()
