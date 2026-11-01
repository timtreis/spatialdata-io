from __future__ import annotations

import io
import math
import mmap
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree
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
_MZML_NS = "http://psi.hupo.org/ms/mzml"
_BATCH = 2000  # spectra per vectorised step
# m/z accuracy by mass analyser, used when `ppm` is not given
_PPM_BY_ANALYSER = (("orbitrap", 5.0), ("fourier transform ion cyclotron", 5.0), ("ft-icr", 5.0), ("fticr", 5.0))
_PPM_TOF = 20.0
_PPM_DEFAULT = 10.0
# header pixel sizes above this are FlexImaging pixel areas in µm²
_AREA_THRESHOLD = 1000.0
# max dimension x may span count or count - 1 pixels
_DIM_TOL = 0.05


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
        Pixel size in µm from the repository text, used when the file gives none. The header value is checked
        against the file's dimensions and units; ``uns["spatialdata_io"]["pixel_size_source"]`` records the outcome:
        ``"imzml header"``, ``"imzml header (area->sqrt)"`` (FlexImaging writes µm²), ``"imzml dimensions"`` (a 1.0
        placeholder replaced by max dimension / pixels), ``"repository text"``, ``"inconsistent"`` or ``"ambiguous
        unitless header"`` (both without a pixel size), or ``None``.

    Returns
    -------
    :class:`spatialdata.SpatialData` with the table ``"table"`` (pixels with at least one peak x features, raw summed
    intensities as CSR float32; ``obs`` ``x``/``y`` grid indices from 1 and ``tic``; ``var`` ``mz``, ``mz_low``,
    ``mz_high``) annotating one square per pixel, the shapes ``"msi_pixels"``. With a pixel size, the shapes are
    scaled to µm in ``"global"``; without, they are in grid units and ``uns["spatialdata_io"]["pitch_unknown"]`` is
    ``True``. ``uns["spatialdata_io"]`` also records the mode, polarity, analyser, m/z range, ``ppm`` and their sources.
    """
    try:
        from pyimzml.ImzMLParser import INFER_IBD_FROM_IMZML, ImzMLParser
    except ImportError as e:  # pragma: no cover
        raise ImportError("imzml() needs pyimzml: pip install 'spatialdata-io[imzml]'.") from e

    path = Path(path)
    head = _header(path)

    source: Any = str(path)
    ibd: Any = INFER_IBD_FROM_IMZML
    root_tag = head[head.find("<mzML") : head.find(">", head.find("<mzML"))]
    if f'xmlns="{_MZML_NS}"' not in root_tag:
        # pyimzml matches tags in the mzML namespace only; some exports (e.g. PXD053247) omit it, so no spectrum is found
        source = io.BytesIO(path.read_bytes().replace(b"<mzML", f'<mzML xmlns="{_MZML_NS}"'.encode(), 1))
        ibd = open(_ibd_path(path), "rb")  # kept open: the parser reads spectra from it lazily
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # pyimzml warns about outdated ontology term names
        parser = ImzMLParser(source, ibd_file=ibd)
    coords = np.asarray(parser.coordinates)
    # continuous mode: one shared m/z array, so the spectra can be sliced straight from a memmap of the .ibd
    ibd_path = _ibd_path(path) if "IMS:1000030" in head and len(set(parser.mzOffsets)) == 1 else None
    pos3d = _positions_3d(path, len(coords))
    if pos3d is None and (coords[:, 2] > 1).any():
        raise ValueError(f"{path.name} has {coords[:, 2].max()} z planes; only 2-D imzML is supported.")
    if pos3d is None and (n_dup := len(coords) - len(np.unique(coords[:, :2], axis=0))) > 0:
        # e.g. Bruker 3D exports: the planes share one x/y grid and only the 3DPositionZ parameter tells them apart
        raise ValueError(f"{path.name}: {n_dup} spectra repeat an x/y position; only 2-D imzML is supported.")

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
        totals = _bin_totals(parser, step, ibd_path)
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

    X, tic, n_peaks, mz_range = _count(parser, lookup, len(var), ibd_path)
    keep = np.flatnonzero(n_peaks > 0)
    xy = coords[keep, :2]

    raw_px = _pixel_size(parser)
    d = parser.imzmldict
    dim, count = d.get("max dimension x"), d.get("max count of pixels x")
    dim_px = float(dim) / float(count) if dim and count else None
    px, px_source = _resolve_pixel_size(raw_px, dim_px, _header_has_unit(head), pixel_size_um)
    measured = None
    if pos3d is not None:
        # registered voxel centres in µm: the spot spacing within a plane is the pixel size (MTBLS176 microbes: 400
        # µm, as stated, while the header says 100); a stated pixel size wins, as registration can stretch it
        # (MTBLS176 pancreas: 60 µm stated, 64.2 µm measured)
        measured = _plane_spacing(pos3d)
        if pixel_size_um is not None:
            px, px_source = pixel_size_um, "repository text"
            if not math.isclose(measured, pixel_size_um, rel_tol=0.05):
                logger.warning(
                    f"{path.name}: the 3DPosition spot spacing is {measured:.1f} µm, not the stated {pixel_size_um} µm; "
                    "using the stated pixel size."
                )
        else:
            px, px_source = measured, "3DPosition spacing"
        spacing = px
    if pos3d is None and px_source not in ("repository text", None) and px is not None and pixel_size_um is not None:
        if not math.isclose(px, pixel_size_um, rel_tol=0.01):
            logger.warning(
                f"{path.name}: {px} µm per pixel from the file ({px_source}), not {pixel_size_um}; using the file."
            )

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
        "pixel_size_header": raw_px,
        "pixel_size_from_dimensions": dim_px,
        "pitch_unknown": px is None,
    }
    if measured is not None:
        uns["measured_spacing_um"] = measured
    if pos3d is not None:
        return _set_reader_metadata(
            _sdata_3d(X[keep], var, coords[keep], pos3d[keep], tic[keep], keep, spacing, uns), "imzml"
        )
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


def _positions_3d(path: Path, n: int) -> np.ndarray | None:
    """Per-spectrum ``3DPositionX/Y/Z`` userParams (µm) of Bruker/SCiLS 3D exports, if every spectrum has them."""
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
        if m.find(b'name="3DPositionZ"') < 0:
            return None
        cols = [
            [float(v) for v in re.findall(rb'name="3DPosition%s" value="([-0-9.eE+]+)"' % a.encode(), m)]  # type: ignore[call-overload]
            for a in "XYZ"
        ]
    if any(len(c) != n for c in cols):
        return None
    return np.array(cols).T


def _plane_spacing(pos: np.ndarray) -> float:
    """Median distance to the nearest spectrum of the same plane."""
    d = [cKDTree(p := pos[pos[:, 2] == z, :2]).query(p, k=2)[0][:, 1] for z in np.unique(pos[:, 2])]
    return float(np.median(np.concatenate(d)))


def _sdata_3d(
    X: sp.csr_matrix,
    var: pd.DataFrame,
    grid: np.ndarray,
    pos: np.ndarray,
    tic: np.ndarray,
    keep: np.ndarray,
    px: float,
    uns: dict[str, Any],
) -> SpatialData:
    """One ``msi_pixels_z<µm>`` squares element per plane in µm, all annotated by one table with ``z_um``."""
    regions = np.array([f"{_SHAPES}_z{z:g}" for z in pos[:, 2]])
    obs = pd.DataFrame(
        {
            "x": grid[:, 0],
            "y": grid[:, 1],
            "x_um": pos[:, 0],
            "y_um": pos[:, 1],
            "z_um": pos[:, 2],
            "tic": tic,
            "region": pd.Categorical(regions),
            "instance_id": keep,
        },
        index=pd.Index([str(i) for i in keep]),
    )
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = pos[:, :2]
    adata.uns["spatialdata_io"] = uns | {"n_planes": len(set(regions))}
    names = sorted({str(r) for r in regions}, key=lambda r: float(r.rsplit("_z", 1)[1]))
    table = TableModel.parse(adata, region=names, region_key="region", instance_key="instance_id")
    h = px / 2
    shapes = {
        name: ShapesModel.parse(
            gpd.GeoDataFrame(
                {"geometry": [box(x - h, y - h, x + h, y + h) for x, y in pos[regions == name, :2]]},
                index=keep[regions == name],
            ),
            transformations={"global": Identity()},
        )
        for name in names
    }
    return SpatialData(shapes=shapes, tables={"table": table})


def _ibd_path(path: Path) -> Path:
    """The ``.ibd`` next to ``path``, matching the extension case-insensitively."""
    hits = [p for p in path.parent.glob(f"{path.stem}.*") if p.suffix.lower() == ".ibd"]
    if not hits:
        raise FileNotFoundError(f"No .ibd next to {path}.")
    return hits[0]


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


def _header_has_unit(head: str) -> bool:
    """Whether the header's pixel size cvParam (IMS:1000046) carries a unit attribute."""
    m = re.search(r'<cvParam [^>]*accession="IMS:1000046"[^>]*>', head)
    return m is not None and "unit" in m.group(0)


def _resolve_pixel_size(
    raw: float | None, dim_px: float | None, has_unit: bool, text: float | None
) -> tuple[float | None, str | None]:
    """The pixel size in µm and its source, from the header, the file's dimensions or the repository ``text``.

    FlexImaging writes the pixel area in µm² and some converters a 1.0 placeholder, so the header is checked:

        - 1.0: ``max dimension x / max count of pixels x`` if given (``"imzml dimensions"``), else ``text``;
        - with those dimensions: the side if it matches their ratio, the square root if it matches its square,
          else ``"inconsistent"`` (no pixel size);
        - without them: above 1000 an area; at most 1000 and without a unit ambiguous (no pixel size, unless
          ``text`` agrees within 5% with the header read as a side or as an area); otherwise the side.
    """
    from_text = (text, "repository text") if text is not None else (None, None)
    if raw is None:
        return from_text
    if raw == 1.0:
        return (dim_px, "imzml dimensions") if dim_px is not None else from_text
    if dim_px is not None:
        if math.isclose(raw, dim_px, rel_tol=_DIM_TOL):
            return raw, "imzml header"
        if math.isclose(raw, dim_px**2, rel_tol=2 * _DIM_TOL):
            return math.sqrt(raw), "imzml header (area->sqrt)"
        logger.warning(
            f"Header pixel size {raw} matches neither max dimension / pixels ({dim_px:.4g} µm) nor its square."
        )
        return None, "inconsistent"
    if raw > _AREA_THRESHOLD:
        return math.sqrt(raw), "imzml header (area->sqrt)"
    if not has_unit:
        if text is not None and any(math.isclose(text, v, rel_tol=_DIM_TOL) for v in (raw, math.sqrt(raw))):
            return from_text
        return None, "ambiguous unitless header"
    return raw, "imzml header"


def _read_features(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    col = df["mz"] if "mz" in df.columns else df.iloc[:, 0]
    return np.unique(pd.to_numeric(col, errors="coerce").dropna().to_numpy(dtype=float))


def _spectra(parser: Any, ibd_path: Path | None = None) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Batches of ``(pixel index, m/z, intensity)`` over all spectra.

    With ``ibd_path`` (continuous mode), the arrays are sliced from a memmap of the ``.ibd``: the same bytes and dtypes
    as :meth:`pyimzml.ImzMLParser.getspectrum` reads, so the results are identical, without a Python read per spectrum.
    """
    n = len(parser.coordinates)
    if ibd_path is not None:
        yield from _spectra_memmap(parser, ibd_path)
        return
    for start in range(0, n, _BATCH):
        pix, mzs, ints = [], [], []
        for i in range(start, min(start + _BATCH, n)):
            mz, it = parser.getspectrum(i)
            pix.append(np.full(len(mz), i, dtype=np.int64))
            mzs.append(np.asarray(mz, dtype=float))
            ints.append(np.asarray(it, dtype=float))
        yield np.concatenate(pix), np.concatenate(mzs), np.concatenate(ints)


def _spectra_memmap(parser: Any, ibd_path: Path) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    n = len(parser.coordinates)
    mm = np.memmap(ibd_path, dtype=np.uint8, mode="r")
    mz_type, it_type = np.dtype(parser.mzPrecision), np.dtype(parser.intensityPrecision)
    off, length = parser.mzOffsets[0], parser.mzLengths[0]
    mz = np.asarray(mm[off : off + length * mz_type.itemsize].view(mz_type), dtype=float)
    it_off, it_len = np.asarray(parser.intensityOffsets), np.asarray(parser.intensityLengths)
    for start in range(0, n, _BATCH):
        stop = min(start + _BATCH, n)
        ints = [mm[o : o + k * it_type.itemsize] for o, k in zip(it_off[start:stop], it_len[start:stop], strict=True)]
        counts = it_len[start:stop]
        yield (
            np.repeat(np.arange(start, stop, dtype=np.int64), counts),
            np.tile(mz, stop - start) if (counts == len(mz)).all() else np.concatenate([mz[:k] for k in counts]),
            np.asarray(np.concatenate(ints).view(it_type), dtype=float),
        )


def _bin_totals(parser: Any, step: float, ibd_path: Path | None = None) -> np.ndarray:
    """Total intensity per log-spaced m/z bin (bin ``i`` spans ``exp(i * step)`` to ``exp((i + 1) * step)``)."""
    totals = np.zeros(0)
    for _, mz, it in _spectra(parser, ibd_path):
        ok = mz >= 1  # m/z below 1 is no ion, and its log bin would be negative
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
        return np.where((np.abs(top[i] - b) <= 1) & (mz >= 1), i, -1)

    return lookup


def _deposited_lookup(feat: np.ndarray, tol: float) -> Any:
    padded = np.r_[-np.inf, feat, np.inf]

    def lookup(mz: np.ndarray) -> np.ndarray:
        i = np.searchsorted(feat, mz)  # feat[i - 1] < mz <= feat[i], as padded[i] and padded[i + 1]
        nearest = np.where(mz - padded[i] <= padded[i + 1] - mz, i - 1, i)
        nearest = np.clip(nearest, 0, len(feat) - 1)
        return np.where(np.abs(feat[nearest] - mz) <= feat[nearest] * tol, nearest, -1)

    return lookup


def _count(
    parser: Any, lookup: Any, n_feat: int, ibd_path: Path | None = None
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, tuple[float, float]]:
    """Pixels x features CSR of summed intensities, the TIC and number of peaks per pixel, and the m/z range."""
    n = len(parser.coordinates)
    tic, n_peaks = np.zeros(n), np.zeros(n, dtype=np.int64)
    blocks: list[sp.csr_matrix] = []
    lo, hi, done = np.inf, -np.inf, 0
    for pix, mz, it in _spectra(parser, ibd_path):
        tic += np.bincount(pix, weights=it, minlength=n)
        n_peaks += np.bincount(pix, minlength=n)
        if len(mz):
            lo, hi = min(lo, float(mz.min())), max(hi, float(mz.max()))
        f = lookup(mz)
        m = f >= 0
        # stacked row blocks: adding n x n_feat blocks was quadratic in the number of batches. Each block runs from
        # the rows done so far to its last pixel, so empty spectra at either end of a batch keep their rows.
        end = int(pix.max()) + 1 if len(pix) else done
        blocks.append(sp.csr_matrix((it[m].astype(np.float32), (pix[m] - done, f[m])), shape=(end - done, n_feat)))
        done = end
    blocks.append(sp.csr_matrix((n - done, n_feat), dtype=np.float32))  # trailing spectra without peaks
    return sp.vstack(blocks, format="csr"), tic, n_peaks, (lo, hi)
