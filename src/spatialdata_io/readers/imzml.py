from __future__ import annotations

import math
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.sparse as sp
from shapely.geometry import box
from spatialdata import SpatialData
from spatialdata._logging import logger
from spatialdata.models import ShapesModel, TableModel
from spatialdata.transformations import Identity, Scale

from spatialdata_io.readers._utils._utils import _set_reader_metadata

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["imzml"]

_SHAPES = "msi_pixels"
_BATCH = 2000  # spectra per vectorised step
# m/z accuracy by mass analyser, used when `ppm` is not given
_PPM_BY_ANALYSER = (("orbitrap", 5.0), ("fourier transform ion cyclotron", 5.0), ("ft-icr", 5.0), ("fticr", 5.0))
_PPM_TOF = 20.0
_PPM_DEFAULT = 10.0
# header pixel sizes above this are FlexImaging pixel areas in µm²
_AREA_THRESHOLD = 1000.0


def imzml(
    path: str | Path,
    *,
    features: str | Path | None = None,
    n_features: int = 1000,
    ppm: float | None = None,
    pixel_size_um: float | None = None,
) -> SpatialData:
    """Read a 2-D *imzML* imaging mass spectrometry dataset (``.imzML`` + ``.ibd``) into pixels x m/z features.

    A full spectrum per pixel is too large for a table, so the spectra are summarised on one m/z feature axis:

        - ``features``: a deposited m/z list (CSV with an ``mz`` column, else its first column). Each pixel's peaks
          within ``ppm`` of a feature (the nearest one) are summed into it. ``feature_space="deposited"``.
        - otherwise, a lossy summary: a first streaming pass bins all peaks on a log-spaced grid ``ppm`` wide; the
          ``n_features`` local maxima of the total intensity become the features, each counting the peaks within one
          bin of its maximum (``mz_low``/``mz_high``), so an ion drifting across neighbouring bins stays one feature
          (``feature_space="mz_top{n}_{ppm}ppm"``).

    ``ppm`` defaults to the mass analyser named in the header: 5 for Orbitrap and FT-ICR, 20 for time-of-flight,
    else 10. Processed and continuous imzML are read the same way; files with several z planes raise an error.

    Parameters
    ----------
    path
        The ``.imzML`` file; its ``.ibd`` must sit next to it.
    features
        Deposited m/z feature list.
    n_features
        Number of m/z bins kept without ``features``.
    ppm
        m/z tolerance (bin width) in ppm; see above for the default.
    pixel_size_um
        Pixel size in µm, used when the header states none (e.g. from the repository text). A header value above
        1000 is read as the pixel area in µm² (FlexImaging) and its square root is used; a header value of exactly 1
        is treated as a converter placeholder, i.e. as no value.

    Returns
    -------
    :class:`spatialdata.SpatialData` with the table ``"table"`` (pixels with at least one peak x features, raw summed
    intensities as CSR float32; ``obs`` ``x``/``y`` grid indices from 1 and ``tic``; ``var`` ``mz``, ``mz_low``,
    ``mz_high``) annotating one square per pixel, the shapes ``"msi_pixels"``. With a pixel size, the shapes are
    scaled to µm in ``"global"``; without, they are in grid units and ``uns["spatialdata_io"]["pitch_unknown"]`` is
    ``True``. ``uns["spatialdata_io"]`` also records the mode, polarity, analyser, m/z range, ``ppm`` and their sources.
    """
    try:
        from pyimzml.ImzMLParser import ImzMLParser
    except ImportError as e:  # pragma: no cover
        raise ImportError("imzml() needs pyimzml: pip install 'spatialdata-io[imzml]'.") from e

    path = Path(path)
    head = _header(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # pyimzml warns about outdated ontology term names
        parser = ImzMLParser(str(path))
    coords = np.asarray(parser.coordinates)
    if (coords[:, 2] > 1).any():
        raise ValueError(f"{path.name} has {coords[:, 2].max()} z planes; only 2-D imzML is supported.")

    analyser = _analyser(head)
    if ppm is not None:
        ppm_source = "kwarg"
    else:
        ppm = dict(_PPM_BY_ANALYSER).get(analyser or "", _PPM_TOF if analyser == "time-of-flight" else _PPM_DEFAULT)
        ppm_source = "analyser" if analyser else "default"
    tol = ppm * 1e-6

    if features is not None:
        feat = _read_features(Path(features))
        lookup = _deposited_lookup(feat, tol)
        var = pd.DataFrame({"mz": feat, "mz_low": feat * (1 - tol), "mz_high": feat * (1 + tol)})
        space = "deposited"
    else:
        step = math.log1p(tol)
        totals = _bin_totals(parser, step)
        # local maxima only, so an ion whose m/z drifts across neighbouring bins between pixels is one feature
        left, right = np.r_[0.0, totals[:-1]], np.r_[totals[1:], 0.0]
        peaks = np.flatnonzero((totals > 0) & (totals >= left) & (totals > right))
        top = np.sort(peaks[np.argsort(totals[peaks])[::-1][:n_features]])
        lookup = _bin_lookup(top, step)
        var = pd.DataFrame(
            {"mz": np.exp((top + 0.5) * step), "mz_low": np.exp((top - 1) * step), "mz_high": np.exp((top + 2) * step)}
        )
        space = f"mz_top{len(top)}_{ppm:g}ppm"
    var.index = pd.Index([f"mz_{mz:.4f}" for mz in var["mz"]])

    X, tic, n_peaks, mz_range = _count(parser, lookup, len(var))
    keep = np.flatnonzero(n_peaks > 0)
    xy = coords[keep, :2]

    header_px, header_source = _pixel_size(parser), "imzml header"
    if header_px is not None and header_px > _AREA_THRESHOLD:
        # FlexImaging writes the pixel area in µm² (10000 for 100 µm pixels)
        header_px, header_source = math.sqrt(header_px), "imzml header (area->sqrt)"
    if header_px == 1.0:
        header_px = None  # a converter placeholder, not a 1 µm pixel
    if header_px is not None:
        px, px_source = header_px, header_source
        if pixel_size_um is not None and not math.isclose(header_px, pixel_size_um, rel_tol=0.01):
            logger.warning(f"{path.name} states {header_px} µm per pixel, not {pixel_size_um}; using the header.")
    elif pixel_size_um is not None:
        px, px_source = pixel_size_um, "repository text"
    else:
        px, px_source = None, None

    uns = {
        "reader": "imzml",
        "imzml": str(path),
        "imzml_mode": "continuous" if "IMS:1000030" in head else "processed" if "IMS:1000031" in head else None,
        "polarity": "positive" if "MS:1000130" in head else "negative" if "MS:1000129" in head else None,
        "analyser": analyser,
        "mz_range": [float(mz_range[0]), float(mz_range[1])],
        "ppm": ppm,
        "ppm_source": ppm_source,
        "feature_space": space,
        "n_features": len(var),
        "n_pixels_empty": int(len(coords) - len(keep)),
        "pixel_size_um": px,
        "pixel_size_source": px_source,
        "pitch_unknown": px is None,
    }
    obs = pd.DataFrame(
        {
            "x": xy[:, 0],
            "y": xy[:, 1],
            "tic": tic[keep],
            "region": pd.Categorical([_SHAPES] * len(keep)),
            "instance_id": keep,
        },
        index=pd.Index([f"{x}_{y}" for x, y in xy]),
    )
    adata = ad.AnnData(X=X[keep], obs=obs, var=var)
    adata.obsm["spatial"] = xy.astype(float) - 0.5
    adata.uns["spatialdata_io"] = uns
    table = TableModel.parse(adata, region=_SHAPES, region_key="region", instance_key="instance_id")

    transform = Scale([px, px], axes=("x", "y")) if px is not None else Identity()
    squares = gpd.GeoDataFrame({"geometry": [box(x - 1, y - 1, x, y) for x, y in xy]}, index=keep)
    shapes = ShapesModel.parse(squares, transformations={"global": transform})
    return _set_reader_metadata(SpatialData(shapes={_SHAPES: shapes}, tables={"table": table}), "imzml")


def _header(path: Path, size: int = 4 << 20) -> str:
    """The start of the imzML XML, which holds the file description and instrument configuration."""
    with open(path, "rb") as f:
        return f.read(size).decode("utf-8", errors="replace")


def _analyser(head: str) -> str | None:
    names = " ".join(re.findall(r'<cvParam [^>]*name="([^"]+)"', head)).lower()
    for key, _ in _PPM_BY_ANALYSER:
        if key in names:
            return "orbitrap" if key == "orbitrap" else "fticr"
    return "time-of-flight" if "time-of-flight" in names or "tof" in names.split() else None


def _pixel_size(parser: Any) -> float | None:
    d = parser.imzmldict
    x, y = d.get("pixel size x"), d.get("pixel size y")
    if x is None:
        return None
    if y is not None and not math.isclose(float(x), float(y), rel_tol=1e-3):
        raise ValueError(f"Non-square pixels ({x} x {y} µm) are not supported.")
    return float(x)


def _read_features(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    col = df["mz"] if "mz" in df.columns else df.iloc[:, 0]
    return np.unique(pd.to_numeric(col, errors="coerce").dropna().to_numpy(dtype=float))


def _spectra(parser: Any) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Batches of ``(pixel index, m/z, intensity)`` over all spectra."""
    n = len(parser.coordinates)
    for start in range(0, n, _BATCH):
        pix, mzs, ints = [], [], []
        for i in range(start, min(start + _BATCH, n)):
            mz, it = parser.getspectrum(i)
            pix.append(np.full(len(mz), i, dtype=np.int64))
            mzs.append(np.asarray(mz, dtype=float))
            ints.append(np.asarray(it, dtype=float))
        yield np.concatenate(pix), np.concatenate(mzs), np.concatenate(ints)


def _bin_totals(parser: Any, step: float) -> np.ndarray:
    """Total intensity per log-spaced m/z bin (bin ``i`` spans ``exp(i * step)`` to ``exp((i + 1) * step)``)."""
    totals = np.zeros(0)
    for _, mz, it in _spectra(parser):
        ok = mz > 0
        b = np.floor(np.log(mz[ok]) / step).astype(np.int64)
        part = np.bincount(b, weights=it[ok])
        if len(part) > len(totals):
            totals = np.pad(totals, (0, len(part) - len(totals)))
        totals[: len(part)] += part
    return totals


def _bin_lookup(top: np.ndarray, step: float) -> Any:
    """Map m/z to the feature whose peak bin is at most one bin away (maxima are never adjacent)."""

    def lookup(mz: np.ndarray) -> np.ndarray:
        if not len(top):
            return np.full(len(mz), -1)
        b = np.floor(np.log(np.maximum(mz, 1e-12)) / step).astype(np.int64)
        i = np.minimum(np.searchsorted(top, b - 1), len(top) - 1)
        return np.where(np.abs(top[i] - b) <= 1, i, -1)

    return lookup


def _deposited_lookup(feat: np.ndarray, tol: float) -> Any:
    padded = np.r_[-np.inf, feat, np.inf]

    def lookup(mz: np.ndarray) -> np.ndarray:
        i = np.searchsorted(feat, mz)  # feat[i - 1] < mz <= feat[i], as padded[i] and padded[i + 1]
        nearest = np.where(mz - padded[i] <= padded[i + 1] - mz, i - 1, i)
        nearest = np.clip(nearest, 0, len(feat) - 1)
        return np.where(np.abs(feat[nearest] - mz) <= feat[nearest] * tol, nearest, -1)

    return lookup


def _count(parser: Any, lookup: Any, n_feat: int) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, tuple[float, float]]:
    """Pixels x features CSR of summed intensities, the TIC and number of peaks per pixel, and the m/z range."""
    n = len(parser.coordinates)
    tic, n_peaks = np.zeros(n), np.zeros(n, dtype=np.int64)
    blocks: list[sp.csr_matrix] = []
    lo, hi = np.inf, -np.inf
    for pix, mz, it in _spectra(parser):
        tic += np.bincount(pix, weights=it, minlength=n)
        n_peaks += np.bincount(pix, minlength=n)
        if len(mz):
            lo, hi = min(lo, float(mz.min())), max(hi, float(mz.max()))
        f = lookup(mz)
        m = f >= 0
        blocks.append(sp.csr_matrix((it[m].astype(np.float32), (pix[m], f[m])), shape=(n, n_feat)))
    X = sum(blocks[1:], blocks[0]) if blocks else sp.csr_matrix((n, n_feat), dtype=np.float32)
    return X.tocsr(), tic, n_peaks, (lo, hi)
