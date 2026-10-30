"""The full-resolution image check of visium() on a synthetic Space Ranger output."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest
import tifffile

from spatialdata_io import visium

# spots in the full-resolution frame (x = pxl_col_in_fullres, y = pxl_row_in_fullres), like a section in a
# whole-slide image: they sit far from the origin
_SPOTS = {"AAAC-1": (5000, 3000), "AAAG-1": (5200, 3000), "AAAT-1": (5000, 3200), "AACA-1": (5200, 3200)}


def _space_ranger(root: Path, image_shape: tuple[int, int]) -> Path:
    (root / "spatial").mkdir(parents=True)
    genes = ["G1", "G2"]
    rows = [",".join(["", *genes])] + [f"{b},{i},{i + 1}" for i, b in enumerate(_SPOTS)]
    (root / "counts.csv").write_text("\n".join(rows) + "\n")
    positions = [f"{b},1,{i // 2},{i % 2},{y},{x}" for i, (b, (x, y)) in enumerate(_SPOTS.items())]
    (root / "spatial" / "tissue_positions_list.csv").write_text("\n".join(positions) + "\n")
    scalefactors = {"spot_diameter_fullres": 90.0, "tissue_hires_scalef": 0.1, "tissue_lowres_scalef": 0.03}
    (root / "spatial" / "scalefactors_json.json").write_text(json.dumps(scalefactors))
    tifffile.imwrite(root / "image.tif", np.zeros((*image_shape, 3), dtype=np.uint8))
    return root


def _read(root: Path) -> object:
    return visium(
        root,
        dataset_id="S1",
        counts_file="counts.csv",
        fullres_image_file="image.tif",
        delimiter=",",
        first_column_names=True,
    )


def test_fullres_image_containing_the_spots_is_kept(tmp_path: Path) -> None:
    sdata = _read(_space_ranger(tmp_path, (4000, 6000)))
    assert "S1_full_image" in sdata.images


@pytest.mark.parametrize(
    "image_shape",
    [
        (1000, 1000),  # a crop of the frame around the capture area, shipped without its offset
        (3000, 3200),  # the CytAssist capture image (3000 x 3200 px), in another frame than the spots
    ],
    ids=["crop", "cytassist"],
)
def test_image_not_containing_the_spots_is_skipped(
    tmp_path: Path, image_shape: tuple[int, int], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        sdata = _read(_space_ranger(tmp_path, image_shape))
    assert "S1_full_image" not in sdata.images
    assert "S1" in sdata.shapes and "table" in sdata.tables
    assert "of the spots fall inside image.tif" in caplog.text
