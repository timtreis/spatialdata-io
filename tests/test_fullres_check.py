"""The one full-resolution placement check, shared by visium, visium_hd and callers that pick among images."""

from __future__ import annotations

import numpy as np
import pytest
from spatialdata.models import Image2DModel

from spatialdata_io.readers._utils._fullres import (
    MAX_HIRES_SIZE_ERROR,
    MIN_SPOTS_INSIDE_FULLRES,
    fraction_inside,
    not_fullres_reason,
)
from spatialdata_io.readers.visium_hd import _drop_fullres_in_another_frame

_XY = np.array([[10.0, 10.0], [20.0, 10.0]])  # bin / spot centres (x, y) in full-resolution px


def test_thresholds():
    assert (MIN_SPOTS_INSIDE_FULLRES, MAX_HIRES_SIZE_ERROR) == (0.99, 0.01)
    assert fraction_inside(np.array([[0, 0], [10, 10], [10, 11]]), (10, 10)) == 2 / 3  # bounds inclusive


@pytest.mark.parametrize(
    ("shape_yx", "in_tissue", "hires", "reason"),
    [
        ((30, 30), None, None, None),
        ((15, 15), None, None, "15 x 15 px: 50.0% of spots inside (need 99%)"),
        ((15, 15), [True, False], None, None),  # the off-tissue spot may fall outside a tissue crop
        ((15, 15), [False, False], None, "15 x 15 px: 50.0% of in-tissue spots inside"),  # none in tissue: all
        ((30, 30), None, ((3, 3), 0.1), None),
        ((30, 40), None, ((3, 3), 0.1), "40 x 30 px scaled by tissue_hires_scalef 0.1 is off from the 3 x 3 px"),
        ((15, 15), None, ((3, 3), 0.2), "50.0% of spots inside"),  # containment is checked first
    ],
)
def test_not_fullres_reason(shape_yx, in_tissue, hires, reason):
    hires_shape, scalef = hires or (None, None)
    got = not_fullres_reason(shape_yx, _XY, in_tissue, scalef, hires_shape)
    assert got == reason if reason is None else reason in got


def _image(h: int, w: int, multiscale: bool = False):
    return Image2DModel.parse(
        np.zeros((3, h, w), dtype=np.uint8), dims=("c", "y", "x"), scale_factors=[2] if multiscale else None
    )


@pytest.mark.parametrize(("full", "kept"), [((30, 30), True), ((30, 40), False)])
def test_visium_hd_skips_a_full_image_in_another_frame(full, kept):
    images = {"S1_full_image": _image(*full, multiscale=True), "S1_hires_image": _image(30, 30)}
    spots = (_XY, np.array([True, True]))
    if kept:
        _drop_fullres_in_another_frame(images, "S1", spots, hires_scalef=1.0)
    else:
        with pytest.warns(UserWarning, match="is off from the 30 x 30 px hires image"):
            _drop_fullres_in_another_frame(images, "S1", spots, hires_scalef=1.0)
    assert ("S1_full_image" in images) is kept
    assert "S1_hires_image" in images


def test_visium_hd_without_bins_checks_nothing():
    images = {"S1_full_image": _image(5, 5)}
    _drop_fullres_in_another_frame(images, "S1", None, hires_scalef=None)
    assert "S1_full_image" in images
