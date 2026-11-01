"""visium_hd() image loading for PNGs that are not plain RGB (GEO deposits)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from spatialdata.models import get_channel_names

from spatialdata_io.readers.visium_hd import _load_image


@pytest.mark.parametrize(
    ("array", "mode", "channels"),
    [
        (np.full((20, 30, 4), 255, dtype=np.uint8), "RGBA", 3),  # GSM9083960: alpha fully opaque, dropped
        (np.dstack([np.zeros((20, 30, 3), np.uint8), np.arange(600, dtype=np.uint8).reshape(20, 30)]), "RGBA", 4),
        (np.zeros((20, 30), dtype=np.uint8), "L", 1),  # grayscale
        (np.zeros((20, 30, 3), dtype=np.uint8), "RGB", 3),
    ],
    ids=["rgba-opaque", "rgba-alpha", "grayscale", "rgb"],
)
def test_load_image_channel_layouts(tmp_path: Path, array: np.ndarray, mode: str, channels: int) -> None:
    path = tmp_path / "tissue_hires_image.png"
    Image.fromarray(array, mode=mode).save(path)
    images: dict = {}
    _load_image(path, images, "_hires_image", "S1", imread_kwargs={}, image_models_kwargs={}, scale_factors=None)
    image = images["S1_hires_image"]
    assert image.shape == (channels, 20, 30) and len(get_channel_names(image)) == channels
