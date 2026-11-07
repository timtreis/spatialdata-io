from __future__ import annotations

import re
from pathlib import Path

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
from shapely.geometry import box
from spatialdata import SpatialData
from spatialdata.models import ShapesModel, TableModel
from spatialdata.transformations import Identity

from spatialdata_io.readers._utils._utils import _set_observation_unit, _set_reader_metadata

__all__ = ["dbit_matrix"]

_SHAPES = "spots"
_SPOT = re.compile(r"^(\d+)x(\d+)$")
# CITE-seq antibody names carry their oligo barcode, e.g. "CD4.AACAAGACCCTTGAG"
_ADT_BARCODE = re.compile(r"\.[ACGT]{6,}$")


def dbit_matrix(
    counts: str | Path,
    *,
    pixel_width_um: float,
    pitch_um: float | None = None,
    protein: str | Path | None = None,
) -> SpatialData:
    """Read a *DBiT* spots x features count matrix (DBiT-seq, xDBiT, spatial-CITE-seq) onto its pixel grid.

    ``counts`` and ``protein`` are each one of:

        - a delimited table (``.tsv``/``.csv``, optionally ``.gz``) whose spots are named ``<A>x<B>`` by their two
          barcode indices, as rows or columns (e.g. GSE213264, GSE137986); ``A`` is taken as x and ``B`` as y;
        - an ``.h5ad`` file whose ``obs`` has ``array_row``/``array_col`` and, optionally, the pixel centres in µm as
          ``um_row``/``um_col`` (e.g. xDBiT, GSE207843). Its ``uns`` (which may hold full-size images) is dropped;
        - a folder with ``matrix.mtx`` (features x spots), ``features.tsv`` (``name`` and ``assay`` columns),
          ``barcodes.tsv`` and ``positions.csv`` (``barcode``, ``array_row``, ``array_col``), as written from a
          Seurat ``.rds`` (e.g. Patho-DBiT, GSE274641) by a conversion script; ``assay`` goes to ``var``.

    Each pixel becomes a ``pixel_width_um`` square in the ``spots`` shapes element, centred at the deposited µm
    position, else at the barcode indices times ``pitch_um``; elements are in µm in ``"global"``. Without either,
    the pixels are one-unit squares at their barcode indices (grid units, ``pitch_unknown`` in ``uns``), and
    ``pixel_width_um`` is kept as metadata only. The RNA counts are
    ``table`` and the antibody counts ``table_protein`` (oligo barcode suffixes stripped from their names); both
    annotate ``spots``.

    Parameters
    ----------
    counts
        RNA counts.
    pixel_width_um
        Width of one pixel (the microfluidic channel width), as stated by the study.
    pitch_um
        Centre-to-centre pixel spacing, as stated by the study; without it (and without µm positions) the pixels are
        placed in grid units.
    protein
        Antibody-derived tag (ADT) counts on the same pixels.

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    adatas = {"table": _read(Path(counts))}
    if protein is not None:
        adatas["table_protein"] = _read(Path(protein))
        adatas["table_protein"].var_names = [_ADT_BARCODE.sub("", n) for n in adatas["table_protein"].var_names]

    spots = pd.concat([a.obs for a in adatas.values() if isinstance(a.obs, pd.DataFrame)])
    spots = spots[~spots.index.duplicated()]
    if {"um_x", "um_y"} <= set(spots.columns):
        centres = spots[["um_x", "um_y"]].to_numpy(dtype=float)
    else:
        centres = spots[["array_x", "array_y"]].to_numpy(dtype=float) * (pitch_um or 1)
    pitch_unknown = "um_x" not in spots.columns and pitch_um is None
    half = 0.5 if pitch_unknown else pixel_width_um / 2
    squares = gpd.GeoDataFrame(
        {"geometry": [box(x - half, y - half, x + half, y + half) for x, y in centres]}, index=np.arange(len(spots))
    )
    shapes = ShapesModel.parse(squares, transformations={"global": Identity()})

    instance = pd.Series(np.arange(len(spots)), index=spots.index)
    tables = {}
    for key, a in adatas.items():
        a.obs["region"] = pd.Categorical([_SHAPES] * a.n_obs)
        a.obs["instance_id"] = instance.loc[a.obs_names].to_numpy()
        a.obsm["spatial"] = centres[a.obs["instance_id"]]
        a.uns["spatialdata_io"] = {
            "pixel_width_um": pixel_width_um,
            "pitch_um": pitch_um,
            "pitch_unknown": pitch_unknown,
        }
        _set_observation_unit(a, None, pixel_width_um)
        tables[key] = TableModel.parse(a, region=_SHAPES, region_key="region", instance_key="instance_id")
    return _set_reader_metadata(SpatialData(shapes={_SHAPES: shapes}, tables=tables), "dbit_matrix")


def _read(path: Path) -> ad.AnnData:
    """Spots x features with integer grid indices ``array_x``/``array_y`` (and µm ``um_x``/``um_y`` if deposited)."""
    if path.is_dir():
        var = pd.read_csv(path / "features.tsv", sep="\t")
        barcodes = (path / "barcodes.tsv").read_text().split()
        pos = pd.read_csv(path / "positions.csv", index_col=0).loc[barcodes]
        a = ad.AnnData(
            sp.csr_matrix(scipy.io.mmread(path / "matrix.mtx").T),
            obs=pd.DataFrame(
                {"array_x": pos["array_col"].to_numpy(), "array_y": pos["array_row"].to_numpy()}, index=barcodes
            ),
            var=pd.DataFrame({"assay": var["assay"].to_numpy()}, index=var["name"].astype(str).to_numpy()),
        )
        a.var_names_make_unique()  # e.g. microRNA host genes named like an mRNA
        return a
    if path.suffix == ".h5ad":
        a = ad.read_h5ad(path)
        a.uns = {}
        src = a.obs
        assert isinstance(src, pd.DataFrame)
        obs = pd.DataFrame({"array_x": src["array_col"].astype(int), "array_y": src["array_row"].astype(int)})
        if {"um_row", "um_col"} <= set(src.columns):
            obs["um_x"], obs["um_y"] = src["um_col"].astype(float), src["um_row"].astype(float)
        a.obs = obs.join(src.drop(columns=["array_row", "array_col", "um_row", "um_col"], errors="ignore"))
        a.X = sp.csr_matrix(a.X)  # type: ignore[arg-type]
        return a
    df = pd.read_csv(path, sep="\t" if ".tsv" in path.suffixes or ".txt" in path.suffixes else ",", index_col=0)
    if not df.index.astype(str).str.fullmatch(_SPOT.pattern).all():
        df = df.T  # features x spots
    ab = df.index.astype(str).str.extract(_SPOT).to_numpy(dtype=int)
    obs = pd.DataFrame({"array_x": ab[:, 0], "array_y": ab[:, 1]}, index=df.index.astype(str))
    return ad.AnnData(sp.csr_matrix(df.to_numpy()), obs=obs, var=pd.DataFrame(index=df.columns.astype(str)))
