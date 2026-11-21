from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import shapely
from geopandas import GeoDataFrame
from scipy.sparse import coo_matrix, csr_matrix
from spatialdata import SpatialData
from spatialdata._core.validation import check_valid_name
from spatialdata.models import ShapesModel, TableModel
from spatialdata.transformations import Scale

from spatialdata_io.readers._utils._utils import _set_observation_unit, _set_reader_metadata
from spatialdata_io.readers.stereoseq import (
    _GEF_PLATFORM_RESOLUTION_NM,
    _bin_chunks,
    _bin_name,
    _bins_table,
    _default_dataset_id,
)

__all__ = ["stereoseq_gem"]

_HEADER = re.compile(r"^#\s*([^=]+?)\s*=\s*(.*?)\s*$")


def stereoseq_gem(
    gem: str | Path | None = None,
    cellbin_gem: str | Path | None = None,
    *,
    bin_size: int = 50,
    dataset_id: str | None = None,
) -> SpatialData:
    """Read *Stereo-seq* GEM text tables (``*.gem``, ``*.tissue.gem``, ``*.cellbin.gem``, plain or gzipped).

    A GEM lists one row per (gene, DNB): ``geneID x y MIDCount ExonCount``, and ``CellID`` in a cellbin GEM, after
    ``#key=value`` header lines (``FileFormat``, ``BinSize``, chip id, ``OffsetX``, ``OffsetY``). The output is the one
    :func:`spatialdata_io.stereoseq_gef` gives for the same inputs, with these differences:

        - every table has the ``ExonCount`` sums as the layer ``"exon"`` (``X`` holds the ``MIDCount`` sums);
        - a cell's outline is the convex hull of its DNB squares (a GEM has no cell polygons and lists only the DNBs
          that captured reads), ``uns["spatialdata_io"]["cell_geometry"]`` says so, and its ``obs`` has ``dnbCount``
          only;
        - the DNB pitch is the 500 nm platform value (a GEM states none), ``resolution_source="platform_default"``;
        - ``uns["spatialdata_io"]["gem_header"]`` keeps the header lines; x/y are kept as written (the offsets are
          not applied).

    Parameters
    ----------
    gem
        A bin1 or ``tissue`` GEM (square bins).
    cellbin_gem
        A ``cellbin`` GEM (cells). At least one of ``gem`` and ``cellbin_gem`` is required.
    bin_size
        Bin size in DNB (positive integer) for the square-bin table.
    dataset_id
        Defaults to the chip id parsed from the file name, as for :func:`spatialdata_io.stereoseq_gef`.

    Returns
    -------
    :class:`spatialdata.SpatialData`, laid out as :func:`spatialdata_io.stereoseq_gef`.
    """
    if gem is None and cellbin_gem is None:
        raise ValueError("Provide at least one of `gem` or `cellbin_gem`.")
    if isinstance(bin_size, bool) or not isinstance(bin_size, int | np.integer) or bin_size <= 0:
        raise ValueError(f"`bin_size` must be a positive integer, got {bin_size!r}.")
    primary = Path(gem if gem is not None else cellbin_gem)  # type: ignore[arg-type]
    dataset_id = dataset_id or _default_dataset_id(primary)
    check_valid_name(f"{dataset_id}_cell_boundaries")
    r = _GEF_PLATFORM_RESOLUTION_NM / 1000
    transform = {"global": Scale([r, r], axes=("x", "y"))}
    meta: dict[str, Any] = {
        "reader": "stereoseq_gem",
        "resolution_nm": _GEF_PLATFORM_RESOLUTION_NM,
        "resolution_source": "platform_default",
        "bin_size_dnb": int(bin_size),
        "bin_size_um": bin_size * r,
    }
    tables: dict[str, ad.AnnData] = {}
    shapes: dict[str, GeoDataFrame] = {}
    if cellbin_gem is not None:
        header, df = _read_gem(Path(cellbin_gem), cells=True)
        adata, cell_shapes = _cells(df, f"{dataset_id}_cell_boundaries", transform)
        adata.uns["spatialdata_io"] = meta | {"gem_header": header, "cell_geometry": "convex hull of DNB squares"}
        _set_observation_unit(adata, "cell", None)
        region = f"{dataset_id}_cell_boundaries"
        tables["table"] = TableModel.parse(adata, region=region, region_key="region", instance_key="cell_id")
        shapes[region] = cell_shapes
    if gem is not None:
        header, df = _read_gem(Path(gem), cells=False)
        region = f"{dataset_id}_{_bin_name(bin_size, _GEF_PLATFORM_RESOLUTION_NM)}"
        adata, squares = _bins(df, bin_size, region, transform)
        adata.uns["spatialdata_io"] = meta | {"gem_header": header}
        _set_observation_unit(adata, None, meta["bin_size_um"])
        name = _bin_name(bin_size, _GEF_PLATFORM_RESOLUTION_NM) if cellbin_gem is not None else "table"
        tables[name] = TableModel.parse(adata, region=region, region_key="region", instance_key="bin_id")
        shapes[region] = squares
    return _set_reader_metadata(SpatialData(tables=tables, shapes=shapes), "stereoseq_gem")


def _read_gem(path: Path, *, cells: bool) -> tuple[dict[str, str], pd.DataFrame]:
    """The ``#key=value`` header and the rows of a GEM, plain or gzipped."""
    import gzip

    header: dict[str, str] = {}
    with gzip.open(path, "rt") if path.suffix == ".gz" else open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            if m := _HEADER.match(line):
                header[m[1]] = m[2]
    dtype = {"geneID": "category", "x": np.int64, "y": np.int64, "MIDCount": np.int64, "ExonCount": np.int64}
    df = pd.read_csv(path, sep="\t", comment="#", dtype=dtype | ({"CellID": np.int64} if cells else {}))
    if cells and "CellID" not in df:
        raise ValueError(f"{path.name} has no CellID column; pass it as `gem=`.")
    if (df["x"] < 0).any() or (df["y"] < 0).any():
        raise ValueError(f"{path.name}: negative x/y are not supported.")
    return header, df


def _bins(df: pd.DataFrame, bin_size: int, region: str, transform: dict) -> tuple[ad.AnnData, GeoDataFrame]:
    """Square bins as :func:`stereoseq_gef` builds them: the GEM rows go through the same ``_bin_chunks``."""
    codes = df["geneID"].cat.codes.to_numpy()
    order = np.argsort(codes, kind="stable")
    n_genes = len(df["geneID"].cat.categories)
    counts = np.bincount(codes, minlength=n_genes)
    gene = np.zeros(n_genes, dtype=[("offset", np.int64), ("count", np.int64)])
    gene["count"] = counts
    gene["offset"] = np.cumsum(counts) - counts

    def matrix(column: str) -> tuple[np.ndarray, csr_matrix]:
        expression = np.zeros(len(df), dtype=[("x", np.int64), ("y", np.int64), ("count", np.int64)])
        for field, col in (("x", "x"), ("y", "y"), ("count", column)):
            expression[field] = df[col].to_numpy()[order]
        return _bin_chunks(expression, gene, bin_size, chunk_rows=2**22)  # type: ignore[arg-type]

    keys, x_mat = matrix("MIDCount")
    exon_keys, exon = matrix("ExonCount")  # a bin with no exon count is not in exon_keys
    exon = coo_matrix(exon)
    exon = csr_matrix((exon.data, (np.searchsorted(keys, exon_keys)[exon.row], exon.col)), shape=x_mat.shape)
    adata, squares = _bins_table(keys, x_mat, np.asarray(df["geneID"].cat.categories), bin_size, region, transform)
    adata.layers["exon"] = exon
    return adata, squares


def _cells(df: pd.DataFrame, region: str, transform: dict) -> tuple[ad.AnnData, GeoDataFrame]:
    """Cells x genes (``MIDCount`` and ``ExonCount`` sums) and each cell's outline, the convex hull of its DNB squares."""
    cell_ids, cell = np.unique(df["CellID"].to_numpy(), return_inverse=True)
    gene = df["geneID"].cat.codes.to_numpy()
    shape = (len(cell_ids), len(df["geneID"].cat.categories))
    x_mat = csr_matrix(coo_matrix((df["MIDCount"].to_numpy(), (cell, gene)), shape=shape))
    exon = csr_matrix(coo_matrix((df["ExonCount"].to_numpy(), (cell, gene)), shape=shape))
    dnb = pd.DataFrame({"cell": cell, "x": df["x"].to_numpy(), "y": df["y"].to_numpy()}).drop_duplicates()
    dnb = dnb.sort_values("cell", kind="stable")
    # a GEM lists only the DNBs that captured reads (a median 83 per cell on zenodo:19524969 O2), so their union is
    # scattered squares; the outline is the convex hull of the cell's DNB squares (their 4 corners each)
    corners = np.stack([np.c_[dnb["x"] + dx, dnb["y"] + dy] for dx in (0, 1) for dy in (0, 1)], axis=1).reshape(-1, 2)
    geoms = shapely.convex_hull(shapely.multipoints(corners, indices=np.repeat(dnb["cell"].to_numpy(), 4)))
    bounds = np.searchsorted(dnb["cell"].to_numpy(), np.arange(len(cell_ids) + 1))
    obs = pd.DataFrame(index=pd.Index(cell_ids.astype(str)))
    obs["cell_id"] = cell_ids
    obs["region"] = pd.Categorical([region] * len(cell_ids))
    obs["dnbCount"] = np.diff(bounds)
    adata = ad.AnnData(X=x_mat, obs=obs, var=pd.DataFrame(index=pd.Index(df["geneID"].cat.categories.astype(str))))
    adata.layers["exon"] = exon
    centroid = dnb.groupby("cell")[["x", "y"]].mean().to_numpy() + 0.5
    adata.obsm["spatial"] = centroid
    gdf = GeoDataFrame(geometry=geoms, index=pd.Index(cell_ids))
    return adata, ShapesModel.parse(gdf, transformations=dict(transform))
