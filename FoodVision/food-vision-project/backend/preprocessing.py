from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image


IMAGE_SIZE = (224, 224)


PathLike = Union[str, Path]


def load_image(path: PathLike) -> Image.Image:
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path)


def to_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def resize(image: Image.Image, size: tuple[int, int] = IMAGE_SIZE) -> Image.Image:
    return image.resize(size, resample=Image.BILINEAR)


def normalize(image_array: np.ndarray) -> np.ndarray:
    return image_array.astype(np.float32) / 255.0


def prepare_image_batch(path: PathLike) -> np.ndarray:
    image = load_image(path)
    image = to_rgb(image)
    image = resize(image, IMAGE_SIZE)
    image_array = np.array(image)
    image_array = normalize(image_array)
    return np.expand_dims(image_array, axis=0)

