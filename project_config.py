from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
INSTANCE_DIR = BASE_DIR / "instance"
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = STATIC_DIR / "uploads"
UPLOADS_WEB_DIR = "static/uploads"
DATABASE_PATH = INSTANCE_DIR / "users.db"
DEFAULT_DATASET_DIR = BASE_DIR / r"C:\Users\sudhakar gajare\Desktop\ad\Dataset"

ANEMIA_CLASS_NAMES = ["Anemia", "NoAnemia"]
CATARACT_CLASS_NAMES = ["1_normal", "2_cataract"]

ANEMIA_MODEL_CANDIDATES = [
    MODELS_DIR / "anemia_model_best.keras",
    MODELS_DIR / "anemia_model_best.h5",
]

CATARACT_MODEL_CANDIDATES = [
    MODELS_DIR / "efficientnet_cataract_best.keras",
    MODELS_DIR / "efficientnet_cataract_tf.keras",
]

ANEMIA_LEGACY_ARTIFACTS = [
    MODELS_DIR / "anemia_model_best.pkl",
    MODELS_DIR / "mobilenet_anemia_tf.h5",
    BASE_DIR / "anemia_model_best.keras",
    BASE_DIR / "mobilenet_anemia_tf.h5",
    BASE_DIR / "_model" / "anemia_model_best.h5",
]

CATARACT_LEGACY_ARTIFACTS = [
    BASE_DIR / "efficientnet_cataract_tf.h5",
]


def ensure_runtime_directories() -> None:
    for path in (MODELS_DIR, INSTANCE_DIR, UPLOADS_DIR, DEFAULT_DATASET_DIR):
        path.mkdir(parents=True, exist_ok=True)


def get_dataset_dir() -> Path:
    raw_path = os.getenv("MEDISCAN_DATASET_DIR")
    return Path(raw_path).expanduser() if raw_path else DEFAULT_DATASET_DIR


def resolve_first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def list_existing_paths(paths: Iterable[Path]) -> list[str]:
    return [str(path) for path in paths if path.exists()]
