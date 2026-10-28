from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

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


def _positive(obs: pd.DataFrame, column: str) -> np.ndarray:
    values = obs[column].to_numpy(dtype=float)
    if not np.all(values > 0):  # also catches NaN
        raise ValueError(
            f"Column {column!r} has missing or non-positive values, so no cell radius can be derived from it; "
            "pass cell_radius."
        )
    return values


def _resolve_radius(obs: pd.DataFrame, unit: str, cell_radius: float | None) -> tuple[str, float | np.ndarray]:
    if cell_radius is not None:
        return "stated", float(cell_radius)
    if unit == "um":
        if "volume" in obs.columns:
            volume = _positive(obs, "volume")
            return "derived:volume", np.cbrt(3 * volume / (4 * np.pi))
        if "area" in obs.columns:
            area = _positive(obs, "area")
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
    coordinate_labels: tuple[str, str] | None = None,
    var: pd.DataFrame | None = None,
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
    coordinate_labels
        Names recorded in ``uns`` for ``x`` and ``y`` instead of the column names (e.g. for ``obsm`` sources).
    var
        Per-gene metadata indexed by ``var_names`` (e.g. from an h5ad file); gene rows are kept in ``table.var``.
    """
    for col in (x, y):
        if col not in obs.columns:
            raise ValueError(f"Coordinate column {col!r} not found; available columns: {list(obs.columns)[:10]}.")

    n_nan = int(obs[[x, y]].isna().to_numpy().any(axis=1).sum())
    if n_nan:
        raise ValueError(f"{n_nan} cells have NaN (missing) coordinates in columns {x!r} / {y!r}.")

    # Element name per cell; ids only need to be unique within a split.
    region_of = np.full(len(obs), "cells", dtype=object)
    if split_by is not None:
        if split_by not in obs.columns:
            raise ValueError(f"split_by column {split_by!r} not found; available columns: {list(obs.columns)[:10]}.")
        split_values = obs[split_by]
        if split_values.isna().any():
            raise ValueError(f"split_by column {split_by!r} has {int(split_values.isna().sum())} missing values.")
        safe = {v: re.sub("[^A-Za-z0-9_]", "_", str(v)) for v in split_values.unique()}
        by_name: dict[str, list[str]] = {}
        for v, sv in safe.items():
            by_name.setdefault(sv.lower(), []).append(str(v))  # case-insensitive: stores may lowercase paths
        clash = [vs for vs in by_name.values() if len(vs) > 1]
        if clash:
            raise ValueError(
                f"Values of {split_by!r} {clash[0]} map to the same element name after replacing characters "
                "other than letters, digits and '_' (compared case-insensitively)."
            )
        region_of = split_values.map({v: f"cells_{sv}" for v, sv in safe.items()}).to_numpy(dtype=object)
    for _, ids in obs.index.to_series().groupby(region_of):
        if not ids.is_unique:
            dup = ids[ids.duplicated()].unique()[:5].tolist()
            raise ValueError(f"Found duplicate cell ids, for example {dup}.")
    obs = obs.copy()
    for col in (_INSTANCE_KEY, _REGION_KEY):
        if col in obs.columns:
            logger.warning(f"Column {col!r} in the input is reserved by cell_table(); renamed to '{col}_original'.")
            obs = obs.rename(columns={col: f"{col}_original"})
    if not obs.index.is_unique:  # same id in several splits: keep AnnData's index unique
        prefix = region_of.astype(str)
        obs.index = pd.Index([f"{r[len('cells_') :]}_{i}" for r, i in zip(prefix, obs.index, strict=True)])
        if not obs.index.is_unique:
            dup = obs.index[obs.index.duplicated()].unique()[:5].tolist()
            raise ValueError(f"Cell ids are not unique after prefixing them with the split value, for example {dup}.")

    csr = sp.csr_matrix(cast("Any", X.to_numpy() if isinstance(X, pd.DataFrame) else X))
    if not np.all(np.mod(csr.data, 1) == 0):
        raise ValueError(
            "Found non-integer values in the counts; this looks like a normalized matrix. "
            "cell_table() only reads raw integer counts."
        )
    is_gene = ~var_names.str.lower().str.contains("blank")
    genes = csr[:, np.flatnonzero(is_gene)]
    i32 = np.iinfo(np.int32)
    fits = genes.data.size == 0 or (genes.data.min() >= 0 and genes.data.max() <= i32.max)
    x_dtype = "int32" if fits else "int64"
    genes = sp.csr_matrix(genes.astype(x_dtype))
    blank = pd.DataFrame(csr[:, np.flatnonzero(~is_gene)].toarray(), index=obs.index, columns=var_names[~is_gene])

    n = len(obs)
    obs[_INSTANCE_KEY] = np.arange(n)
    transformation, factor = _transformation(unit, pixel_size)
    source, radius_um = _resolve_radius(obs, unit, cell_radius)

    var_frame = var.loc[is_gene] if var is not None else pd.DataFrame(index=var_names[is_gene])
    adata = ad.AnnData(X=genes, obs=obs, var=var_frame)
    adata.obsm["spatial"] = obs[[x, y]].to_numpy(dtype=float)
    adata.obsm["blank"] = blank

    adata.obs[_REGION_KEY] = pd.Categorical(region_of)

    coords = obs[[x, y]].to_numpy(dtype=float)
    radius = np.broadcast_to(np.asarray(radius_um) / factor, (n,))
    shapes_by_region: dict[str, Any] = {}
    for region in adata.obs[_REGION_KEY].cat.categories:
        pos = np.flatnonzero(region_of == region)
        shapes_by_region[region] = ShapesModel.parse(
            coords[pos],
            geometry=0,
            radius=radius[pos],
            index=pd.Index(pos),
            transformations={"global": transformation},
        )

    table = TableModel.parse(adata, region=list(shapes_by_region), region_key=_REGION_KEY, instance_key=_INSTANCE_KEY)
    xy_labels = coordinate_labels or (x, y)
    table.uns["spatialdata_io"] = {
        "reader": "cell_table",
        "coordinates": {"x": xy_labels[0], "y": xy_labels[1], "unit": unit, "pixel_size": pixel_size},
        "radius": {
            "source": source,
            "median": float(np.median(radius_um)),
            "min": float(np.min(radius_um)),
            "max": float(np.max(radius_um)),
        },
        "x_dtype": x_dtype,
        "n_coordinates_unmatched": n_coordinates_unmatched,
    }
    sdata = SpatialData(shapes=shapes_by_region, tables={"table": table})
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
    gene_cols = [
        c
        for i, c in enumerate(df.columns)
        if c != id_col and c not in meta and not (i > 0 and str(c).startswith("Unnamed:"))
    ]
    bad = [c for c in gene_cols if not pd.api.types.is_numeric_dtype(df[c]) or df[c].isna().any()]
    if bad:
        raise ValueError(
            f"Gene columns {bad[:5]} in {counts.name} are non-numeric or contain missing values; "
            "pass the non-gene columns in metadata_columns."
        )
    ids = df[id_col].astype(str)
    obs = df[meta].copy()
    obs.index = pd.Index(ids, name=None)
    genes = df[gene_cols]
    return genes, pd.Index(gene_cols), obs


_FEATURE_NAMES = ("features", "genes", "feats")
_BARCODE_NAMES = ("barcodes", "cells", "obs")
_FEATURE_SUFFIXES = (".tsv", ".tsv.gz", ".txt", ".txt.gz", ".csv", ".csv.gz")
_BARCODE_SUFFIXES = _FEATURE_SUFFIXES[:4]  # no .csv: a cells.csv next to the matrix is usually the coordinates


def _mtx_stem(path: Path) -> str:
    name = path.name
    for ext in (".mtx.gz", ".mtx"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return path.stem


def _find_sibling(path: Path, bases: tuple[str, ...], stem_suffix: str, suffixes: tuple[str, ...]) -> Path | None:
    stem = _mtx_stem(path)
    prefixes = [""]
    if stem.lower().endswith("matrix") and (len(stem) == 6 or stem[-7] in "_."):
        prefixes.append(stem[:-6])  # GEO: <prefix>matrix.mtx.gz next to <prefix>features.tsv.gz
    candidates = [f"{prefix}{base}{ext}" for prefix in prefixes for base in bases for ext in suffixes]
    candidates += [f"{stem}_{stem_suffix}{ext}" for ext in suffixes]
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
        ``.mtx`` the labels come from sibling files named ``{features,genes}`` (``.tsv``/``.txt``/``.csv``) and
        ``{barcodes,cells}`` (``.tsv``/``.txt``), optionally ``.gz`` and optionally behind the prefix of a
        ``<prefix>matrix.mtx`` name, or ``<stem>_feats`` and ``<stem>_obs``.
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
        feats_path = _find_sibling(path, _FEATURE_NAMES, "feats", _FEATURE_SUFFIXES)
        obs_path = _find_sibling(path, _BARCODE_NAMES, "obs", _BARCODE_SUFFIXES)
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
    obs, n_unmatched = _join_coordinates(cell_ids, coords, counts.name, coordinates.name)
    return X, var_names, obs, n_unmatched


def _join_coordinates(
    cell_ids: pd.Index, coords: pd.DataFrame, counts_name: str, coords_name: str
) -> tuple[pd.DataFrame, int]:
    """Reindex ``coords`` to ``cell_ids``; raise if a cell has none, count (and warn on) unused rows."""
    missing = ~cell_ids.isin(coords.index)
    if missing.any():
        examples = cell_ids[missing][:5].tolist()
        raise ValueError(
            f"{int(missing.sum())} cells in {counts_name} have no coordinates in {coords_name}, "
            f"for example {examples}. If the counts are oriented the other way, set genes_as_rows."
        )
    n_unmatched = int((~coords.index.isin(cell_ids)).sum())
    if n_unmatched > 0.1 * len(coords):
        logger.warning(f"Dropped {n_unmatched} of {len(coords)} coordinate rows that have no counts.")
    return coords.reindex(cell_ids), n_unmatched


_OBSM_COORDINATE_KEYS = ("spatial", "X_spatial")


def _read_h5ad(
    path: Path, *, layer: str | None, coordinates: Path | None, cell_id: str | None, x: str, y: str, skiprows: int
) -> tuple[sp.spmatrix, pd.DataFrame, pd.DataFrame, int, str, str, dict[str, str]]:
    """Read counts, ``obs`` and a coordinate source from an h5ad file.

    Only ``obs`` is kept; ``obsm`` is used solely as a coordinate source (``spatial`` or ``X_spatial``)
    when ``x`` and ``y`` are not in ``obs`` and no ``coordinates`` file is given.

    Returns
    -------
    The counts, ``var`` (indexed by gene name), ``obs``, the number of unmatched coordinate rows, the ``x`` / ``y``
    column names and the labels to record in ``uns`` for columns taken from ``obsm``.
    """
    adata = ad.read_h5ad(path)
    var: pd.DataFrame
    if layer is None:
        X, var = adata.X, cast("Any", adata.var)
    elif layer == "raw":
        if adata.raw is None:
            raise ValueError(f"{path.name} has no .raw; set layer to None or one of {list(adata.layers.keys())}.")
        X, var = adata.raw.X, cast("Any", adata.raw.var)
    elif layer in adata.layers:
        X, var = adata.layers[layer], cast("Any", adata.var)
    else:
        raise ValueError(
            f"Layer {layer!r} not found in {path.name}; available layers: {list(adata.layers.keys())} "
            "(plus 'raw' for .raw.X, or None for X)."
        )
    if X is None:
        raise ValueError(f"{path.name} has no counts matrix for layer={layer!r}.")
    X = sp.csr_matrix(cast("Any", X))
    var = var.copy()
    var.index = pd.Index(var.index.astype(str))
    obs: pd.DataFrame = cast("Any", adata.obs).copy()
    obs.index = pd.Index(obs.index.astype(str))

    n_unmatched = 0
    labels: dict[str, str] = {}
    if coordinates is not None:
        coords = _read_coordinates(coordinates, cell_id=cell_id, skiprows=skiprows)
        joined, n_unmatched = _join_coordinates(obs.index, coords, path.name, coordinates.name)
        obs = pd.concat([obs.drop(columns=joined.columns.intersection(obs.columns)), joined], axis=1)
    elif (x in obs.columns) != (y in obs.columns):
        raise ValueError(
            f"Only one of the coordinate columns {x!r} / {y!r} is in obs of {path.name}; "
            f"available obs columns: {list(obs.columns)[:10]}."
        )
    elif x not in obs.columns:
        key = next((k for k in _OBSM_COORDINATE_KEYS if k in adata.obsm), None)
        if key is None:
            raise ValueError(
                f"Coordinate columns {x!r} / {y!r} not found in obs of {path.name} and obsm has none of "
                f"{_OBSM_COORDINATE_KEYS}; available obs columns: {list(obs.columns)[:10]}."
            )
        xy = np.asarray(adata.obsm[key])
        x, y = f"obsm_{key}_0", f"obsm_{key}_1"  # obs column names must be alphanumeric/underscore
        obs[x], obs[y] = xy[:, 0], xy[:, 1]
        labels = {x: f"obsm:{key}[0]", y: f"obsm:{key}[1]"}
    return X, var, obs, n_unmatched, x, y, labels


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

    Reads a single wide delimited file holding the id, the centroids, optional metadata and one column per
    gene (cells x genes); a counts file (delimited in either orientation, or ``.mtx(.gz)`` with label files
    next to it) plus a separate ``coordinates`` file; or an ``.h5ad`` file. The raw integer counts go to
    ``table.X`` (CSR), columns whose name contains ``blank`` (case-insensitive) go to ``table.obsm["blank"]``,
    the deposited centroids to ``table.obsm["spatial"]``, and each cell becomes a circle in the ``cells``
    shapes element.

    Parameters
    ----------
    counts
        Path to a ``.csv``, ``.tsv`` or ``.txt`` file (``.gz`` allowed), a ``.mtx(.gz)`` file (needs
        ``coordinates``) or an ``.h5ad`` file. The delimiter follows the extension; ``.txt`` means tab.
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
        Column splitting the cells into several shapes elements ``cells_<value>`` (characters other than
        letters, digits and ``_`` in the value become ``_``), all annotated by the single ``table``
        (``region`` lists them). For a wide CSV or an h5ad file, cell ids need only be unique within a split;
        when an id repeats across splits the ``table.obs`` index is prefixed with the split value. With a
        ``coordinates`` file the ids must be globally unique.
    metadata_columns
        Non-gene columns, copied to ``obs``. Defaults to the Vizgen ``cell_metadata`` columns that are
        present; ``x``, ``y`` and ``split_by`` are always included. All other columns are genes and must be
        numeric. Not allowed together with ``coordinates``.
    skiprows
        Number of leading lines to skip before the header (of each delimited file).
    layer
        Counts source of an h5ad file: ``None`` reads ``X``, ``"raw"`` reads ``.raw.X`` (with ``.raw.var``),
        any other value reads ``layers[layer]``. There is no automatic fallback. Only ``obs`` and ``var`` are
        kept; ``obsm`` is used only as a coordinate source (``spatial`` or ``X_spatial``) when ``x`` / ``y`` are
        not in ``obs`` and no ``coordinates`` file is given. The values are then added to ``obs`` as
        ``obsm_<key>_0`` and ``obsm_<key>_1``, and recorded in ``uns`` as ``obsm:<key>[0]`` and ``obsm:<key>[1]``.
        Only valid for ``.h5ad`` input.
    cell_radius
        Circle radius in µm. If ``None`` and ``unit="um"``, it is derived per cell from ``obs["volume"]``
        as the radius of the sphere of equal volume, ``(3V / 4π)^(1/3)``, or else from ``obs["area"]``
        as the radius of the disc of equal area, ``sqrt(A / π)``. Otherwise 5 µm is used. The source
        (``stated``, ``derived:<column>`` or ``default``) is recorded in
        ``table.uns["spatialdata_io"]["radius"]`` together with the median, minimum and maximum radius
        over all cells in µm (``median``, ``min``, ``max``). The per-cell radius is the ``radius`` column
        of the shapes.

    Returns
    -------
    A :class:`spatialdata.SpatialData` with the table ``table`` (annotating the shapes ``cells``, or
    ``cells_<value>`` with ``split_by``; ``region_key="region"``,
    ``instance_key="cell_id"``, an integer range equal to the shapes index; the string cell id is
    ``table.obs.index``).

    Raises
    ------
    ValueError
        If ``unit`` is invalid, ``unit="px"`` has no ``pixel_size``, the counts are not integral, cell ids
        are duplicated, the id, ``x``, ``y`` or ``split_by`` columns are missing, ``split_by`` has missing values,
        two values map to the same element name, coordinates are NaN, a gene column is non-numeric or has missing
        values, a derived radius is missing or non-positive, or ``layer`` / ``metadata_columns`` would be ignored.
        Input columns named ``cell_id`` or ``region`` are renamed to ``cell_id_original`` / ``region_original``
        with a warning.
    """
    if unit not in _UNITS:
        raise ValueError(f"unit must be one of {_UNITS}, got {unit!r}.")
    if unit == "px" and pixel_size is None:
        raise ValueError("unit='px' requires pixel_size (micrometres per pixel).")
    counts = Path(counts)
    name = counts.name.lower()
    n_unmatched = 0
    labels: dict[str, str] = {}
    genes: pd.DataFrame | sp.spmatrix
    var: pd.DataFrame | None = None
    if layer is not None and not name.endswith(".h5ad"):
        raise ValueError(f"layer={layer!r} only applies to .h5ad input, not {counts.name}.")
    if metadata_columns is not None and coordinates is not None:
        raise ValueError(
            "metadata_columns only applies to a wide counts file; with coordinates, all its columns go to obs."
        )
    if name.endswith(".h5ad"):
        genes, var, obs, n_unmatched, x, y, labels = _read_h5ad(
            counts,
            layer=layer,
            coordinates=Path(coordinates) if coordinates is not None else None,
            cell_id=cell_id,
            x=x,
            y=y,
            skiprows=skiprows,
        )
    elif coordinates is not None:
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
    if var is not None:
        var_names = var.index
    logger.info(f"Read {len(obs)} cells x {len(var_names)} columns from {counts.name}.")
    sdata = _build(
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
        coordinate_labels=(labels[x], labels[y]) if labels else None,
        var=var,
    )
    return sdata
