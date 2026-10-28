from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
from spatialdata import SpatialData
from spatialdata._logging import logger
from spatialdata.models import ShapesModel, TableModel
from spatialdata.transformations import BaseTransformation, Identity, Scale

from spatialdata_io.readers._utils._utils import _set_reader_metadata

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["cell_table"]

VIZGEN_METADATA_COLUMNS: tuple[str, ...] = (
    "fov",
    "volume",
    "center_x",
    "center_y",
    "min_x",
    "max_x",
    "min_y",
    "max_y",
    "anisotropy",
    "transcript_count",
    "perimeter_area_ratio",
    "solidity",
    "EntityID",
)

_UNITS = ("um", "mm", "px")
_DEFAULT_RADIUS_UM = 5.0
_REGION_KEY = "region"
_INSTANCE_KEY = "cell_id"


def _read_delimited(path: str | Path, skiprows: int = 0, **kwargs: Any) -> pd.DataFrame:
    """Read a delimited text file; the delimiter follows the suffix (``.gz`` ignored).

    Extra keyword arguments (e.g. ``dtype``, ``nrows``) go to :func:`pandas.read_csv`.
    """
    path = Path(path)
    suffixes = [s.lower() for s in path.suffixes]
    if suffixes and suffixes[-1] == ".gz":
        suffixes = suffixes[:-1]
    suffix = suffixes[-1] if suffixes else ""
    if suffix == ".csv":
        sep = ","
    elif suffix in (".tsv", ".txt"):
        sep = "\t"
    else:
        raise ValueError(f"Unsupported file type {path.name!r}: expected .csv, .tsv or .txt (optionally .gz).")
    return pd.read_csv(path, sep=sep, skiprows=skiprows, **kwargs)


def _transformation(unit: str, pixel_size: float | None) -> tuple[BaseTransformation, float]:
    """Return the transformation to ``global`` (µm) and the µm per coordinate unit."""
    if unit == "um":
        return Identity(), 1.0
    factor = 1000.0 if unit == "mm" else float(pixel_size)  # type: ignore[arg-type]
    return Scale([factor, factor], axes=("x", "y")), factor


def _resolve_radius(obs: pd.DataFrame, unit: str, cell_radius: float | None) -> tuple[str, float | np.ndarray]:
    if cell_radius is not None:
        return "stated", float(cell_radius)
    if unit == "um":
        if "volume" in obs.columns:
            volume = obs["volume"].to_numpy(dtype=float)
            return "derived:volume", np.cbrt(3 * volume / (4 * np.pi))
        if "area" in obs.columns:
            area = obs["area"].to_numpy(dtype=float)
            return "derived:area", np.sqrt(area / np.pi)
    return "default", _DEFAULT_RADIUS_UM


def _build(
    X: pd.DataFrame | sp.spmatrix,
    var_names: pd.Index,
    obs: pd.DataFrame,
    *,
    unit: str,
    pixel_size: float | None,
    x: str,
    y: str,
    split_by: str | None,
    cell_radius: float | None,
    n_coordinates_unmatched: int,
) -> SpatialData:
    """Validate counts and assemble the SpatialData object.

    Parameters
    ----------
    X
        Cells x genes counts, in the row order of ``obs``.
    var_names
        Gene names; columns containing ``blank`` (case-insensitive) go to ``obsm["blank"]``.
    obs
        Per-cell metadata indexed by the string cell id; must contain the ``x`` and ``y`` columns.
    unit, pixel_size, x, y, split_by, cell_radius
        See :func:`cell_table`.
    n_coordinates_unmatched
        Number of coordinate rows dropped because they have no counts; recorded in ``uns``.
    """
    if not obs.index.is_unique:
        dup = obs.index[obs.index.duplicated()].unique()[:5].tolist()
        raise ValueError(f"Found duplicate cell ids, for example {dup}.")
    for col in (x, y):
        if col not in obs.columns:
            raise ValueError(f"Coordinate column {col!r} not found; available columns: {list(obs.columns)[:10]}.")

    genes: Any
    blank: pd.DataFrame
    is_gene = ~var_names.str.lower().str.contains("blank")
    if isinstance(X, pd.DataFrame):
        values = X.to_numpy()
        genes = values[:, is_gene]
        blank = pd.DataFrame(values[:, ~is_gene], index=obs.index, columns=var_names[~is_gene])
    else:
        csr = X.tocsr()  # type: ignore[attr-defined]  # stub lacks tocsr on spmatrix
        values = csr.data
        genes = csr[:, np.flatnonzero(is_gene)]
        blank = pd.DataFrame(csr[:, np.flatnonzero(~is_gene)].toarray(), index=obs.index, columns=var_names[~is_gene])
    if not np.all(np.mod(values, 1) == 0):
        raise ValueError(
            "Found non-integer values in the counts; this looks like a normalized matrix. "
            "cell_table() only reads raw integer counts."
        )

    obs = obs.copy()
    n = len(obs)
    obs[_INSTANCE_KEY] = np.arange(n)
    transformation, factor = _transformation(unit, pixel_size)
    source, radius_um = _resolve_radius(obs, unit, cell_radius)

    adata = ad.AnnData(X=genes, obs=obs, var=pd.DataFrame(index=var_names[is_gene]))
    adata.obsm["blank"] = blank

    if split_by is not None:
        raise NotImplementedError("split_by is not implemented yet.")
    region = "cells"
    adata.obs[_REGION_KEY] = pd.Categorical([region] * n)

    coords = obs[[x, y]].to_numpy(dtype=float)
    shapes = ShapesModel.parse(
        coords,
        geometry=0,
        radius=np.asarray(radius_um) / factor,
        index=pd.RangeIndex(n),
        transformations={"global": transformation},
    )

    table = TableModel.parse(adata, region=region, region_key=_REGION_KEY, instance_key=_INSTANCE_KEY)
    table.uns["spatialdata_io"] = {
        "reader": "cell_table",
        "coordinates": {"x": x, "y": y, "unit": unit, "pixel_size": pixel_size},
        "radius": {
            "source": source,
            "value_um": radius_um.tolist() if isinstance(radius_um, np.ndarray) else radius_um,
        },
        "n_coordinates_unmatched": n_coordinates_unmatched,
    }
    sdata = SpatialData(shapes={region: shapes}, tables={"table": table})
    return _set_reader_metadata(sdata, "cell_table")


def _read_wide(
    counts: Path,
    *,
    x: str,
    y: str,
    cell_id: str | None,
    split_by: str | None,
    metadata_columns: Sequence[str] | None,
    skiprows: int,
) -> tuple[pd.DataFrame, pd.Index, pd.DataFrame]:
    header = _read_delimited(counts, skiprows, nrows=0).columns
    id_col = cell_id if cell_id is not None else header[0]
    df = _read_delimited(counts, skiprows, dtype={id_col: str} if id_col in header else None)
    required = [id_col, x, y] + ([split_by] if split_by is not None else [])
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not found in {counts.name}; available columns: {list(df.columns)[:10]}.")
    if metadata_columns is None:
        meta = [c for c in VIZGEN_METADATA_COLUMNS if c in df.columns]
    else:
        meta = list(metadata_columns)
    meta = [c for c in dict.fromkeys([*meta, x, y, *([split_by] if split_by else [])]) if c != id_col]
    missing = [c for c in meta if c not in df.columns]
    if missing:
        raise ValueError(f"Metadata columns {missing} not found in {counts.name}.")
    gene_cols = [c for c in df.columns if c != id_col and c not in meta]
    ids = df[id_col].astype(str)
    obs = df[meta].copy()
    obs.index = pd.Index(ids, name=None)
    genes = df[gene_cols]
    return genes, pd.Index(gene_cols), obs


_FEATURE_NAMES = ("features", "genes", "feats")
_BARCODE_NAMES = ("barcodes", "cells", "obs")
_SIBLING_SUFFIXES = (".tsv", ".tsv.gz", ".txt", ".txt.gz", ".csv", ".csv.gz")


def _mtx_stem(path: Path) -> str:
    name = path.name
    for ext in (".mtx.gz", ".mtx"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return path.stem


def _find_sibling(path: Path, bases: tuple[str, ...], stem_suffix: str) -> Path | None:
    candidates = [f"{base}{ext}" for base in bases for ext in _SIBLING_SUFFIXES]
    candidates += [f"{_mtx_stem(path)}_{stem_suffix}{ext}" for ext in _SIBLING_SUFFIXES]
    for cand in candidates:
        if (path.parent / cand).is_file():
            return path.parent / cand
    return None


def _read_names(path: Path) -> pd.Index:
    """Read the first column of a header-less name list."""
    sep = "," if path.name.lower().replace(".gz", "").endswith(".csv") else "\t"
    return pd.Index(pd.read_csv(path, sep=sep, header=None, usecols=[0], dtype=str, keep_default_na=False).iloc[:, 0])


def _infer_rows_are_genes(
    row_ids: pd.Index, col_ids: pd.Index, coord_ids: pd.Index, genes_as_rows: bool | None
) -> bool:
    if genes_as_rows is not None:
        return genes_as_rows
    rows_hit = bool(row_ids.isin(coord_ids).any())
    cols_hit = bool(col_ids.isin(coord_ids).any())
    if rows_hit == cols_hit:
        reason = "both axes" if rows_hit else "neither axis"
        raise ValueError(
            f"Cannot infer the orientation of the counts: {reason} match the coordinate cell ids. "
            "Set genes_as_rows=True (genes in rows) or genes_as_rows=False (cells in rows)."
        )
    return cols_hit


def _read_counts(
    path: Path, *, genes_as_rows: bool | None, coord_ids: pd.Index, skiprows: int
) -> tuple[pd.DataFrame | sp.spmatrix, pd.Index, pd.Index]:
    """Read a counts file with its own axis labels and return cells x genes.

    Parameters
    ----------
    path
        Delimited file (either orientation, first column holds the row labels) or ``.mtx(.gz)``. For
        ``.mtx`` the labels come from sibling files named ``{features,genes}`` and ``{barcodes,cells}``
        with a ``.tsv``/``.txt``/``.csv`` suffix (optionally ``.gz``), or ``<stem>_feats`` and ``<stem>_obs``.
    genes_as_rows
        Orientation; ``None`` infers it from the overlap of the axis labels with ``coord_ids``.
    coord_ids
        String cell ids of the coordinates file.
    skiprows
        Leading lines to skip in a delimited file.

    Returns
    -------
    The cells x genes counts (dense frame or sparse matrix), the gene names and the string cell ids.
    """
    if path.name.lower().endswith((".mtx", ".mtx.gz")):
        feats_path = _find_sibling(path, _FEATURE_NAMES, "feats")
        obs_path = _find_sibling(path, _BARCODE_NAMES, "obs")
        if feats_path is None or obs_path is None:
            raise ValueError(
                f"Could not find the label files next to {path.name}: expected features.tsv(.gz) or genes.tsv(.gz) "
                f"and barcodes.tsv(.gz) or cells.tsv(.gz) (or {_mtx_stem(path)}_feats.* and {_mtx_stem(path)}_obs.*) "
                f"in {path.parent}."
            )
        feats, barcodes = _read_names(feats_path), _read_names(obs_path)
        mat = sp.csr_matrix(scipy.io.mmread(path, spmatrix=True))
        shape = mat.shape
        if genes_as_rows is None and len(feats) != len(barcodes):
            genes_as_rows = shape == (len(feats), len(barcodes))
        if genes_as_rows is None:
            genes_as_rows = _infer_rows_are_genes(feats, barcodes, coord_ids, None)
        rows, cols = (feats, barcodes) if genes_as_rows else (barcodes, feats)
        if shape != (len(rows), len(cols)):
            raise ValueError(
                f"{path.name} has shape {shape} but the label files give {len(rows)} rows x {len(cols)} columns."
            )
        return (mat.T.tocsr(), feats, barcodes) if genes_as_rows else (mat, feats, barcodes)

    df = _read_delimited(path, skiprows, dtype={0: str})
    if isinstance(df.index, pd.RangeIndex):
        df = df.set_index(df.columns[0])
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    rows_are_genes = _infer_rows_are_genes(df.index, df.columns, coord_ids, genes_as_rows)
    if rows_are_genes:
        df = df.T
    return df, pd.Index(df.columns), pd.Index(df.index)


def _read_coordinates(path: Path, *, cell_id: str | None, skiprows: int) -> pd.DataFrame:
    header = _read_delimited(path, skiprows, nrows=0).columns
    named = [c for c in header if not str(c).startswith("Unnamed:")]
    id_col = cell_id if cell_id is not None else (named[0] if named else header[0])
    if id_col not in header:
        raise ValueError(f"Column {id_col!r} not found in {path.name}; available columns: {list(header)[:10]}.")
    df = _read_delimited(path, skiprows, dtype={id_col: str})
    df = df.loc[:, [c for c in df.columns if c == id_col or not str(c).startswith("Unnamed:")]]
    df.index = pd.Index(df[id_col].astype(str))
    df = df.drop(columns=id_col)
    if not df.index.is_unique:
        dup = df.index[df.index.duplicated()].unique()[:5].tolist()
        raise ValueError(f"Found duplicate cell ids in {path.name}, for example {dup}.")
    return df


def _read_split(
    counts: Path,
    coordinates: Path,
    *,
    cell_id: str | None,
    genes_as_rows: bool | None,
    skiprows: int,
) -> tuple[pd.DataFrame | sp.spmatrix, pd.Index, pd.DataFrame, int]:
    coords = _read_coordinates(coordinates, cell_id=cell_id, skiprows=skiprows)
    X, var_names, cell_ids = _read_counts(
        counts, genes_as_rows=genes_as_rows, coord_ids=coords.index, skiprows=skiprows
    )
    missing = ~cell_ids.isin(coords.index)
    if missing.any():
        examples = cell_ids[missing][:5].tolist()
        raise ValueError(
            f"{int(missing.sum())} cells in {counts.name} have no coordinates in {coordinates.name}, "
            f"for example {examples}. If the counts are oriented the other way, set genes_as_rows."
        )
    n_unmatched = int((~coords.index.isin(cell_ids)).sum())
    if n_unmatched > 0.1 * len(coords):
        logger.warning(f"Dropped {n_unmatched} of {len(coords)} coordinate rows that have no counts.")
    obs = coords.reindex(cell_ids)
    return X, var_names, obs, n_unmatched


def cell_table(
    counts: str | Path,
    coordinates: str | Path | None = None,
    *,
    unit: Literal["um", "mm", "px"],
    pixel_size: float | None = None,
    x: str = "center_x",
    y: str = "center_y",
    cell_id: str | None = None,
    genes_as_rows: bool | None = None,
    split_by: str | None = None,
    metadata_columns: Sequence[str] | None = None,
    skiprows: int = 0,
    layer: str | None = None,
    cell_radius: float | None = None,
) -> SpatialData:
    """Read a cell x gene counts table with per-cell centroids.

    Reads either a single wide delimited file holding the id, the centroids, optional metadata and one
    column per gene (cells x genes), or a counts file (delimited in either orientation, or ``.mtx(.gz)``
    with label files next to it) plus a separate ``coordinates`` file. The raw integer counts go to ``table.X``, columns whose
    name contains ``blank`` (case-insensitive) go to ``table.obsm["blank"]``, and each cell becomes a
    circle in the ``cells`` shapes element.

    Parameters
    ----------
    counts
        Path to a ``.csv``, ``.tsv`` or ``.txt`` file (``.gz`` allowed). The delimiter follows the
        extension; ``.txt`` means tab.
    coordinates
        Separate delimited coordinates file with one row per cell. Its ``cell_id`` column is joined to
        the counts (ids compared as strings); all other columns go to ``obs``. Counts cells without
        coordinates raise; coordinate rows without counts are dropped (a warning is logged above 10%).
    unit
        Unit of the coordinate columns: ``"um"``, ``"mm"`` or ``"px"``. Required, never inferred. The
        values are kept as deposited; the ``global`` coordinate system is in µm via a ``Scale``
        transformation (``Identity`` for ``"um"``).
    pixel_size
        Micrometres per pixel. Required for ``unit="px"``.
    x, y
        Names of the centroid columns.
    cell_id
        Name of the id column (of the wide file, or of the coordinates file). Defaults to the first column.
        For the wide file the first column is used even if unnamed (Vizgen exports keep the EntityID there);
        for a coordinates file unnamed index columns are skipped.
    genes_as_rows
        Orientation of a separate counts file (``True``: genes in rows). ``None`` infers it from which axis
        matches the coordinate cell ids. Not used by the wide layout.
    split_by
        Column splitting the cells into several shapes elements. Not implemented yet.
    metadata_columns
        Non-gene columns, copied to ``obs``. Defaults to the Vizgen ``cell_metadata`` columns that are
        present; ``x``, ``y`` and ``split_by`` are always included. All other columns are genes.
    skiprows
        Number of leading lines to skip before the header (of each delimited file).
    layer
        Counts layer of an h5ad file. Not implemented yet.
    cell_radius
        Circle radius in µm. If ``None`` and ``unit="um"``, it is derived per cell from ``obs["volume"]``
        as the radius of the sphere of equal volume, ``(3V / 4π)^(1/3)``, or else from ``obs["area"]``
        as the radius of the disc of equal area, ``sqrt(A / π)``. Otherwise 5 µm is used. The source
        (``stated``, ``derived:<column>`` or ``default``) is recorded in
        ``table.uns["spatialdata_io"]["radius"]``.

    Returns
    -------
    A :class:`spatialdata.SpatialData` with the table ``table`` (annotating the shapes ``cells``;
    ``region_key="region"``, ``instance_key="cell_id"``, an integer range equal to the shapes index; the
    string cell id is ``table.obs.index``).

    Raises
    ------
    ValueError
        If ``unit`` is invalid, ``unit="px"`` has no ``pixel_size``, the counts are not integral, cell ids
        are duplicated, or the id, ``x`` or ``y`` columns are missing.
    NotImplementedError
        For h5ad input.
    """
    if unit not in _UNITS:
        raise ValueError(f"unit must be one of {_UNITS}, got {unit!r}.")
    if unit == "px" and pixel_size is None:
        raise ValueError("unit='px' requires pixel_size (micrometres per pixel).")
    counts = Path(counts)
    name = counts.name.lower()
    n_unmatched = 0
    if name.endswith(".h5ad"):
        raise NotImplementedError("h5ad input is not supported yet.")
    if coordinates is not None:
        genes, var_names, obs, n_unmatched = _read_split(
            counts, Path(coordinates), cell_id=cell_id, genes_as_rows=genes_as_rows, skiprows=skiprows
        )
    elif name.endswith((".mtx", ".mtx.gz")):
        raise ValueError("A .mtx counts file needs a separate coordinates file.")
    else:
        genes, var_names, obs = _read_wide(
            counts,
            x=x,
            y=y,
            cell_id=cell_id,
            split_by=split_by,
            metadata_columns=metadata_columns,
            skiprows=skiprows,
        )
    logger.info(f"Read {len(obs)} cells x {len(var_names)} columns from {counts.name}.")
    return _build(
        genes,
        var_names,
        obs,
        unit=unit,
        pixel_size=pixel_size,
        x=x,
        y=y,
        split_by=split_by,
        cell_radius=cell_radius,
        n_coordinates_unmatched=n_unmatched,
    )
