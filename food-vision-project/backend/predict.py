from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import tensorflow as tf

from preprocessing import prepare_image_batch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "backend" / "model"
MODEL_PATH = MODEL_DIR / "food_model.keras"
MAPPING_PATH = MODEL_DIR / "class_mapping.json"


@dataclass(frozen=True)
class PredictionResult:
    class_name: str
    display_name_tr: str
    confidence: float


def load_model(model_path: Path = MODEL_PATH) -> tf.keras.Model:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Train the model first: py notebooks/train_food_model.py"
        )
    return tf.keras.models.load_model(model_path)


def load_class_mapping(mapping_path: Path = MAPPING_PATH) -> Dict[str, Any]:
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"class_mapping.json not found at {mapping_path}. Train the model first: py notebooks/train_food_model.py"
        )
    return json.loads(mapping_path.read_text(encoding="utf-8"))


def predict_image(image_path: Path) -> PredictionResult:
    model = load_model()
    mapping = load_class_mapping()

    id_to_class_name: Dict[str, str] = mapping["id_to_class_name"]
    id_to_display_name_tr: Dict[str, str] = mapping["id_to_display_name_tr"]

    batch = prepare_image_batch(image_path)
    probs = model.predict(batch, verbose=0)[0]
    predicted_id = int(np.argmax(probs))
    confidence = float(probs[predicted_id]) * 100.0

    class_name = id_to_class_name[str(predicted_id)]
    display_name_tr = id_to_display_name_tr[str(predicted_id)]
    return PredictionResult(class_name=class_name, display_name_tr=display_name_tr, confidence=confidence)

