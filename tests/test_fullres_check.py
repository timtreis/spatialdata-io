"""The one full-resolution placement check, shared by visium, visium_hd and callers that pick among images."""

from __future__ import annotations

import numpy as np
import pytest
from spatialdata.models import Image2DModel

from spatialdata_io.readers._utils._fullres import (
    MAX_HIRES_SIZE_ERROR,
    MIN_BINS_INSIDE_FULLRES_HD,
    MIN_SPOTS_INSIDE_FULLRES,
    fraction_inside,
    fullres_placement,
    not_fullres_reason,
)
from spatialdata_io.readers.visium_hd import _drop_fullres_in_another_frame

_XY = np.array([[10.0, 10.0], [20.0, 10.0]])  # bin / spot centres (x, y) in full-resolution px


def test_thresholds():
    assert (MIN_SPOTS_INSIDE_FULLRES, MIN_BINS_INSIDE_FULLRES_HD, MAX_HIRES_SIZE_ERROR) == (0.99, 0.95, 0.01)
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


def _spots_with_inside(pct: int) -> np.ndarray:
    """100 in-tissue centres, ``pct`` of them inside a 100 x 100 px image and the rest just past its right edge."""
    return np.array([[50.0 if i < pct else 101.0, 50.0] for i in range(100)])


@pytest.mark.parametrize(("pct", "passes"), [(96, True), (94, False)])
def test_visium_hd_floor_with_a_matching_size(pct, passes):
    # an HD image cropped to the tissue: some edge bins fall past it, the size matches the hires image
    images = {"S1_full_image": _image(100, 100), "S1_hires_image": _image(10, 10)}
    in_tissue = np.ones(100, dtype=bool)
    if passes:
        placement = _drop_fullres_in_another_frame(images, "S1", (_spots_with_inside(pct), in_tissue), 0.1)
    else:
        with pytest.warns(UserWarning, match="94.0% of in-tissue spots inside \\(need 95%\\)"):
            placement = _drop_fullres_in_another_frame(images, "S1", (_spots_with_inside(pct), in_tissue), 0.1)
    assert ("S1_full_image" in images) is passes
    assert placement["inside_fraction"] == pct / 100  # recorded either way
    assert (placement["min_inside"], placement["size_error"], placement["size_match"]) == (0.95, 0.0, True)
    assert (placement["reason"] is None) is passes


def test_visium_floor_stays_at_99_percent():
    placement = fullres_placement((100, 100), _spots_with_inside(98), np.ones(100, dtype=bool), 0.1, (10, 10))
    assert placement["reason"] == "100 x 100 px: 98.0% of in-tissue spots inside (need 99%)"
    assert (placement["inside_fraction"], placement["size_match"]) == (0.98, True)


def test_placement_record_without_a_hires_image():
    placement = fullres_placement((30, 30), _XY)
    assert placement == {
        "shape_yx": [30, 30],
        "inside_fraction": 1.0,
        "min_inside": 0.99,
        "size_error": None,
        "max_size_error": 0.01,
        "size_match": None,
        "reason": None,
    }
