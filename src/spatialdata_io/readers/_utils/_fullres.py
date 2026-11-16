"""Whether an image is the full-resolution frame that Space Ranger measured the spots (or bins) in.

The one placement check for ``visium``, ``visium_hd`` and callers that pick a full-resolution image among several
deposited ones (CytAssist capture, crops, second stains). It reads no file: it takes the image's ``(y, x)`` size.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

# a full-resolution image must contain (almost) all in-tissue spots; deposited crops and CytAssist images do not
MIN_SPOTS_INSIDE_FULLRES = 0.99
# Space Ranger makes the hires image by scaling the full-resolution image by tissue_hires_scalef (then rounding)
MAX_HIRES_SIZE_ERROR = 0.01


def fraction_inside(spot_xy: ArrayLike, shape_yx: tuple[int, ...]) -> float:
    """Fraction of the (x, y) spot centres, ignoring missing ones, inside an image of shape ``(y, x)`` (inclusive)."""
    xy = np.asarray(spot_xy, dtype=float).reshape(-1, 2)
    xy = xy[~np.isnan(xy).any(axis=1)]
    if not len(xy):
        return 1.0
    h, w = shape_yx[-2:]
    return float(((xy[:, 0] >= 0) & (xy[:, 0] <= w) & (xy[:, 1] >= 0) & (xy[:, 1] <= h)).mean())


def not_fullres_reason(
    shape_yx: tuple[int, ...],
    spot_xy: ArrayLike,
    in_tissue: ArrayLike | None = None,
    hires_scalef: float | None = None,
    hires_shape_yx: tuple[int, ...] | None = None,
) -> str | None:
    """Why an image of size ``shape_yx`` is not the full-resolution image the spots were measured in, else ``None``.

    First, at least :data:`MIN_SPOTS_INSIDE_FULLRES` of the in-tissue spots (``in_tissue`` mask; all spots without
    one, or when none is in tissue) must fall inside it, so an image cropped to the tissue still passes. Second, when
    the hires image size and ``tissue_hires_scalef`` are both known, the image scaled by ``tissue_hires_scalef`` must
    match the hires size within :data:`MAX_HIRES_SIZE_ERROR`: an image merely larger than the spot array (a
    CytAssist capture that happens to hold the spots) fails here.
    """
    h, w = shape_yx[-2:]
    xy = np.asarray(spot_xy, dtype=float).reshape(-1, 2)
    label = "spots"
    if in_tissue is not None:
        mask = np.asarray(in_tissue, dtype=bool)
        label = "in-tissue spots"
        if mask.any():
            xy = xy[mask]
    inside = fraction_inside(xy, (h, w))
    if inside < MIN_SPOTS_INSIDE_FULLRES:
        return f"{w} x {h} px: {100 * inside:.1f}% of {label} inside (need {100 * MIN_SPOTS_INSIDE_FULLRES:g}%)"
    if hires_shape_yx is not None and hires_scalef:
        hh, hw = hires_shape_yx[-2:]
        err = max(abs(n * hires_scalef - m) / m for n, m in ((h, hh), (w, hw)))
        if err > MAX_HIRES_SIZE_ERROR:
            return (
                f"{w} x {h} px scaled by tissue_hires_scalef {hires_scalef:g} is off from the "
                f"{hw} x {hh} px hires image by {err:.1%}"
            )
    return None
