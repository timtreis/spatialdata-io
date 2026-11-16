"""Whether an image is the full-resolution frame that Space Ranger measured the spots (or bins) in.

The one placement check for ``visium``, ``visium_hd`` and callers that pick a full-resolution image among several
deposited ones (CytAssist capture, crops, second stains). It reads no file: it takes the image's ``(y, x)`` size.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

# a full-resolution image must contain (almost) all in-tissue spots; deposited crops and CytAssist images do not
MIN_SPOTS_INSIDE_FULLRES = 0.99
# Visium HD: the size check decides; in-tissue bins at the edge of an image cropped to the tissue may fall outside
MIN_BINS_INSIDE_FULLRES_HD = 0.95
# Space Ranger makes the hires image by scaling the full-resolution image by tissue_hires_scalef (then rounding)
MAX_HIRES_SIZE_ERROR = 0.01
# where the readers record the check of the full-resolution image they were given (``sdata.attrs``)
FULLRES_PLACEMENT_ATTR = "spatialdata_io_fullres_placement"


def fraction_inside(spot_xy: ArrayLike, shape_yx: tuple[int, ...]) -> float:
    """Fraction of the (x, y) spot centres, ignoring missing ones, inside an image of shape ``(y, x)`` (inclusive)."""
    xy = np.asarray(spot_xy, dtype=float).reshape(-1, 2)
    xy = xy[~np.isnan(xy).any(axis=1)]
    if not len(xy):
        return 1.0
    h, w = shape_yx[-2:]
    return float(((xy[:, 0] >= 0) & (xy[:, 0] <= w) & (xy[:, 1] >= 0) & (xy[:, 1] <= h)).mean())


def fullres_placement(
    shape_yx: tuple[int, ...],
    spot_xy: ArrayLike,
    in_tissue: ArrayLike | None = None,
    hires_scalef: float | None = None,
    hires_shape_yx: tuple[int, ...] | None = None,
    *,
    min_inside: float = MIN_SPOTS_INSIDE_FULLRES,
) -> dict[str, Any]:
    """Whether an image of size ``shape_yx`` is the full-resolution image the spots (or bins) were measured in.

    First, at least ``min_inside`` of the in-tissue spots (``in_tissue`` mask; all spots without one, or when none is
    in tissue) must fall inside it, so an image cropped to the tissue still passes: :data:`MIN_SPOTS_INSIDE_FULLRES`
    for Visium spots, :data:`MIN_BINS_INSIDE_FULLRES_HD` for Visium HD bins. Second, when the hires image size and
    ``tissue_hires_scalef`` are both known, the image scaled by ``tissue_hires_scalef`` must match the hires size
    within :data:`MAX_HIRES_SIZE_ERROR`: an image merely larger than the spot array fails here, and an image that
    matches is the frame even when a few edge spots fall past it.

    Returns a JSON-ready record: ``shape_yx``, ``inside_fraction``, ``min_inside``, ``size_error`` (``None`` when
    the hires size or scale factor is unknown), ``max_size_error``, ``size_match`` (``None`` when not checked) and
    ``reason`` (``None`` when the image is the full-resolution frame).
    """
    h, w = (int(n) for n in shape_yx[-2:])
    xy = np.asarray(spot_xy, dtype=float).reshape(-1, 2)
    label = "spots"
    if in_tissue is not None:
        mask = np.asarray(in_tissue, dtype=bool)
        label = "in-tissue spots"
        if mask.any():
            xy = xy[mask]
    inside = fraction_inside(xy, (h, w))
    size_error = None
    if hires_shape_yx is not None and hires_scalef:
        hh, hw = (int(n) for n in hires_shape_yx[-2:])
        size_error = float(max(abs(n * hires_scalef - m) / m for n, m in ((h, hh), (w, hw))))
    reason = None
    if inside < min_inside:
        reason = f"{w} x {h} px: {100 * inside:.1f}% of {label} inside (need {100 * min_inside:g}%)"
    elif size_error is not None and size_error > MAX_HIRES_SIZE_ERROR:
        reason = (
            f"{w} x {h} px scaled by tissue_hires_scalef {hires_scalef:g} is off from the "
            f"{hw} x {hh} px hires image by {size_error:.1%}"
        )
    return {
        "shape_yx": [h, w],
        "inside_fraction": round(inside, 6),
        "min_inside": min_inside,
        "size_error": None if size_error is None else round(size_error, 6),
        "max_size_error": MAX_HIRES_SIZE_ERROR,
        "size_match": None if size_error is None else size_error <= MAX_HIRES_SIZE_ERROR,
        "reason": reason,
    }


def not_fullres_reason(
    shape_yx: tuple[int, ...],
    spot_xy: ArrayLike,
    in_tissue: ArrayLike | None = None,
    hires_scalef: float | None = None,
    hires_shape_yx: tuple[int, ...] | None = None,
    *,
    min_inside: float = MIN_SPOTS_INSIDE_FULLRES,
) -> str | None:
    """Why an image is not the full-resolution frame, or ``None`` if it is (:func:`fullres_placement`'s ``reason``)."""
    placement = fullres_placement(shape_yx, spot_xy, in_tissue, hires_scalef, hires_shape_yx, min_inside=min_inside)
    return placement["reason"]
