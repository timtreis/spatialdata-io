from __future__ import annotations

import gzip
import io
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import anndata as ad
import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import scipy.sparse as sp
from dask_image.imread import imread
from shapely.geometry import box
from spatialdata import SpatialData
from spatialdata._logging import logger
from spatialdata.models import Image2DModel, ShapesModel, TableModel
from spatialdata.transformations import Identity, Scale

from spatialdata_io.readers._utils._utils import _set_reader_metadata

__all__ = ["dbit_fragments"]

_SHAPES = "epi_spots"
_BLOCK_BYTES = 16 << 20
# collapse the per-block count matrices every this many blocks, so memory follows nnz, not the file size
_COLLAPSE_EVERY = 32
# primary chromosomes (UCSC chrom.sizes); tiles never cover chrM, unplaced or alt contigs
CHROM_SIZES: dict[str, dict[str, int]] = {
    "hg38": {
        "chr1": 248956422,
        "chr2": 242193529,
        "chr3": 198295559,
        "chr4": 190214555,
        "chr5": 181538259,
        "chr6": 170805979,
        "chr7": 159345973,
        "chr8": 145138636,
        "chr9": 138394717,
        "chr10": 133797422,
        "chr11": 135086622,
        "chr12": 133275309,
        "chr13": 114364328,
        "chr14": 107043718,
        "chr15": 101991189,
        "chr16": 90338345,
        "chr17": 83257441,
        "chr18": 80373285,
        "chr19": 58617616,
        "chr20": 64444167,
        "chr21": 46709983,
        "chr22": 50818468,
        "chrX": 156040895,
        "chrY": 57227415,
    },
    "hg19": {
        "chr1": 249250621,
        "chr2": 243199373,
        "chr3": 198022430,
        "chr4": 191154276,
        "chr5": 180915260,
        "chr6": 171115067,
        "chr7": 159138663,
        "chr8": 146364022,
        "chr9": 141213431,
        "chr10": 135534747,
        "chr11": 135006516,
        "chr12": 133851895,
        "chr13": 115169878,
        "chr14": 107349540,
        "chr15": 102531392,
        "chr16": 90354753,
        "chr17": 81195210,
        "chr18": 78077248,
        "chr19": 59128983,
        "chr20": 63025520,
        "chr21": 48129895,
        "chr22": 51304566,
        "chrX": 155270560,
        "chrY": 59373566,
    },
    "mm10": {
        "chr1": 195471971,
        "chr2": 182113224,
        "chr3": 160039680,
        "chr4": 156508116,
        "chr5": 151834684,
        "chr6": 149736546,
        "chr7": 145441459,
        "chr8": 129401213,
        "chr9": 124595110,
        "chr10": 130694993,
        "chr11": 122082543,
        "chr12": 120129022,
        "chr13": 120421639,
        "chr14": 124902244,
        "chr15": 104043685,
        "chr16": 98207768,
        "chr17": 94987271,
        "chr18": 90702639,
        "chr19": 61431566,
        "chrX": 171031299,
        "chrY": 91744698,
    },
    "mm39": {
        "chr1": 195154279,
        "chr2": 181755017,
        "chr3": 159745316,
        "chr4": 156860686,
        "chr5": 151758149,
        "chr6": 149588044,
        "chr7": 144995196,
        "chr8": 130127694,
        "chr9": 124359700,
        "chr10": 130530862,
        "chr11": 121973369,
        "chr12": 120092757,
        "chr13": 120883175,
        "chr14": 125139656,
        "chr15": 104073951,
        "chr16": 98008968,
        "chr17": 95294699,
        "chr18": 90720763,
        "chr19": 61420004,
        "chrX": 169476592,
        "chrY": 91455967,
    },
}


def dbit_fragments(
    spatial: str | Path,
    *,
    genome_build: str,
    fragments: str | Path | None = None,
    peaks: str | Path | None = None,
    peak_matrix: str | Path | None = None,
    tile_size: int = 5000,
    pixel_width_um: float | None = None,
) -> SpatialData:
    """Read a DBiT spatial-ATAC or spatial-CUT&Tag deposit (fragments or a peak matrix, plus the spatial folder).

    ``spatial`` is the extracted barcode tarball: a headerless Visium v1 ``tissue_positions_list.csv``
    (``barcode, in_tissue, row, col, pxl_y, pxl_x``) with ``scalefactors_json.json`` and, optionally,
    ``tissue_hires_image.png`` or ``tissue_lowres_image.png``, anywhere below it. A barcode of the counts, without
    its ``-N`` suffix, is the positions barcode, so a pixel is ``row`` x ``col`` of the DBiT grid.

    The counts come from exactly one of:

        - ``fragments``: a 10x/ArchR ``fragments.tsv(.gz|.bgz)`` (``chrom, start, end, barcode, count``), streamed.
          Each fragment adds its two Tn5 insertion sites (``start`` and ``end - 1``) to the feature holding them
          (ArchR); the duplicate ``count`` is ignored. The features are the ``peaks`` BED/narrowPeak (``.gz``
          allowed; an insertion inside overlapping peaks counts for the one starting last before it) or, without
          ``peaks``, ``tile_size`` bp tiles over the primary chromosomes of ``genome_build``.
        - ``peak_matrix``: a 10x ``raw_peak_bc_matrix.h5``, read as deposited.

    Insertions outside every feature and fragments of barcodes not in the positions are dropped and counted.

    Parameters
    ----------
    spatial
        Folder holding the extracted spatial tarball.
    genome_build
        ``"hg38"``, ``"hg19"``, ``"mm10"`` or ``"mm39"``; never inferred.
    fragments
        Fragments file.
    peaks
        Peaks of ``fragments`` (``chrom, start, end`` in the first three columns).
    peak_matrix
        10x peak-barcode matrix, instead of ``fragments``.
    tile_size
        Tile size in bp when counting ``fragments`` without ``peaks``.
    pixel_width_um
        The stated pixel (channel) width in µm. With it, ``uns["spatialdata_io"]`` gets ``um_per_image_px``
        (``pixel_width_um / spot_diameter_fullres``) and ``centre_spacing_um``; without it, ``pitch_unknown``.

    Returns
    -------
    :class:`spatialdata.SpatialData` with the table ``"table"`` (pixels with at least one fragment or count x
    features ``chr:start-end``, raw counts as CSR int32), one square per pixel in the shapes ``"epi_spots"``
    (side ``spot_diameter_fullres``) and the tissue image ``"tissue_hires"`` under ``Scale(1 / scalef)``, all in
    the full-resolution image pixel frame ``"global"``.
    """
    if (fragments is None) == (peak_matrix is None):
        raise ValueError("Pass exactly one of `fragments` and `peak_matrix`.")
    if genome_build not in CHROM_SIZES:
        raise ValueError(f"Unknown genome_build {genome_build!r}; expected one of {sorted(CHROM_SIZES)}.")
    pos, pos_file = _read_positions(Path(spatial))
    pixel = pd.Series(np.arange(len(pos)), index=pos.index)
    uns: dict[str, Any] = {"reader": "dbit_fragments", "genome_build": genome_build}

    if peak_matrix is not None:
        X, var, barcodes, n_unassigned = _read_peak_matrix(Path(peak_matrix), pixel, genome_build)
        n_fragments = None
        uns |= {"feature_space": "peaks_deposited", "counting": "deposited", "peak_matrix": str(peak_matrix)}
    else:
        if peaks is not None:
            var = _read_peaks(Path(peaks))
            space = "peaks"
        else:
            var = _tiles(CHROM_SIZES[genome_build], tile_size)
            space = f"tiles_{tile_size // 1000}kb" if tile_size % 1000 == 0 else f"tiles_{tile_size}bp"
        X, n_per_pixel, counted = _count_fragments(Path(fragments), pixel, var)  # type: ignore[arg-type]
        n_unassigned = counted["unassigned"]
        n_fragments = n_per_pixel
        uns |= {
            "feature_space": space,
            "counting": "tn5_insertions",
            "fragments": str(fragments),
            "n_insertions_outside_features": counted["outside"],
        }
        barcodes = pos.index[n_per_pixel > 0]
        X = X[np.flatnonzero(n_per_pixel > 0)]
        n_fragments = n_per_pixel[n_per_pixel > 0]

    sf = json.loads((pos_file.parent / "scalefactors_json.json").read_text())
    side = float(sf["spot_diameter_fullres"])
    sel = pos.loc[barcodes]
    grid_step_px = abs(float(np.polyfit(pos["col"], pos["x"], 1)[0])) if pos["col"].nunique() > 1 else None
    um_per_px = pixel_width_um / side if pixel_width_um is not None else None
    uns |= {
        "n_fragments": int(np.sum(n_fragments)) if n_fragments is not None else None,
        "n_unassigned": int(n_unassigned),
        "pixel_width_um": pixel_width_um,
        "um_per_image_px": um_per_px,
        "grid_step_px": grid_step_px,
        "centre_spacing_um": grid_step_px * um_per_px if grid_step_px and um_per_px else None,
        "pitch_unknown": pixel_width_um is None,
    }

    obs = pd.DataFrame(
        {
            "array_row": sel["row"].to_numpy(),
            "array_col": sel["col"].to_numpy(),
            "in_tissue": sel["in_tissue"].to_numpy(),
            "region": pd.Categorical([_SHAPES] * len(sel)),
            "instance_id": sel.index.to_numpy(),
        },
        index=sel.index.to_numpy(),
    )
    if n_fragments is not None:
        obs["n_fragments"] = n_fragments
    adata = ad.AnnData(X=X.astype(np.int32), obs=obs, var=var)
    adata.obsm["spatial"] = sel[["x", "y"]].to_numpy(dtype=float)
    adata.uns["spatialdata_io"] = uns
    table = TableModel.parse(adata, region=_SHAPES, region_key="region", instance_key="instance_id")

    h = side / 2
    squares = [box(x - h, y - h, x + h, y + h) for x, y in np.asarray(adata.obsm["spatial"])]
    shapes = ShapesModel.parse(
        gpd.GeoDataFrame({"geometry": squares}, index=sel.index.to_numpy()), transformations={"global": Identity()}
    )
    images = {}
    for res in ("hires", "lowres"):
        if (image_file := pos_file.parent / f"tissue_{res}_image.png").exists():
            img = imread(str(image_file)).squeeze()
            img = img[..., :3] if img.ndim == 3 else img[..., None]
            k = 1 / float(sf[f"tissue_{res}_scalef"])
            images["tissue_hires"] = Image2DModel.parse(
                img.transpose(2, 0, 1), dims=("c", "y", "x"), transformations={"global": Scale([k, k], axes=("x", "y"))}
            )
            break
    sdata = SpatialData(images=images, shapes={_SHAPES: shapes}, tables={"table": table})
    return _set_reader_metadata(sdata, "dbit_fragments")


def _read_positions(spatial: Path) -> tuple[pd.DataFrame, Path]:
    hits = sorted(spatial.rglob("tissue_positions_list.csv"))
    if not hits:
        raise FileNotFoundError(f"No tissue_positions_list.csv under {spatial}; the barcodes cannot be placed.")
    names = ["barcode", "in_tissue", "row", "col", "y", "x"]
    pos = pd.read_csv(hits[0], header=None, names=names, dtype=str).apply(lambda c: c.str.strip())
    pos = pos.set_index("barcode")
    pos[names[1:]] = pos[names[1:]].apply(pd.to_numeric)
    return pos, hits[0]


def _tiles(sizes: Mapping[str, int], tile_size: int) -> pd.DataFrame:
    starts = {c: np.arange(0, n, tile_size) for c, n in sizes.items()}
    chrom = np.concatenate([np.full(len(s), c, dtype=object) for c, s in starts.items()])
    end = np.concatenate([np.minimum(s + tile_size, sizes[c]) for c, s in starts.items()])
    return _features(chrom, np.concatenate(list(starts.values())), end)


def _read_peaks(path: Path) -> pd.DataFrame:
    with _open(path) as fh:
        lines = b"".join(line for line in fh if not line.startswith((b"#", b"track", b"browser")))
    cols = ["chrom", "start", "end"]
    df = pd.read_csv(io.BytesIO(lines), sep="\t", header=None, usecols=range(3), names=cols, dtype={"chrom": str})
    df = df.drop_duplicates().astype({"start": np.int64, "end": np.int64}).sort_values(cols)
    return _features(df["chrom"].to_numpy(), df["start"].to_numpy(), df["end"].to_numpy())


def _features(chrom: np.ndarray, start: np.ndarray, end: np.ndarray) -> pd.DataFrame:
    index = pd.Index([f"{c}:{s}-{e}" for c, s, e in zip(chrom, start, end, strict=True)])
    return pd.DataFrame({"chrom": pd.Categorical(chrom), "start": start, "end": end}, index=index)


def _open(path: Path) -> Any:
    # gzip.open reads the concatenated members of bgzip files; pyarrow's gzip stream stops after the first
    return gzip.open(path, "rb") if path.suffix in (".gz", ".bgz") else open(path, "rb")


def _count_fragments(
    path: Path, pixel: pd.Series, var: pd.DataFrame
) -> tuple[sp.csr_matrix, np.ndarray, dict[str, int]]:
    """Stream the fragments into a pixels x features CSR of Tn5 insertions, and the fragments per pixel."""
    n_pix, n_feat = len(pixel), len(var)
    # per chromosome: sorted feature starts and ends, and the index of the first feature
    by_chrom: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    for name, idx in var.groupby("chrom", observed=True).indices.items():
        by_chrom[str(name)] = (var["start"].to_numpy()[idx], var["end"].to_numpy()[idx], idx[0])
    with _open(path) as fh:
        n_comments = 0
        while fh.readline().startswith(b"#"):
            n_comments += 1
    names = ["chrom", "start", "end", "barcode"]
    read = pacsv.ReadOptions(column_names=[*names, "count"], skip_rows=n_comments, block_size=_BLOCK_BYTES)
    parse = pacsv.ParseOptions(delimiter="\t")
    convert = pacsv.ConvertOptions(
        include_columns=names,
        column_types={"chrom": pa.string(), "barcode": pa.string(), "start": pa.int64(), "end": pa.int64()},
    )
    blocks: list[sp.csr_matrix] = []
    n_per_pixel = np.zeros(n_pix, dtype=np.int64)
    unassigned = outside = 0
    with _open(path) as fh:
        for batch in pacsv.open_csv(fh, read_options=read, parse_options=parse, convert_options=convert):
            bc = pc.dictionary_encode(batch.column("barcode"))
            lut = pixel.reindex([b.split("-", 1)[0] for b in bc.dictionary.to_pylist()]).fillna(-1).to_numpy(np.int64)
            pix = lut[bc.indices.to_numpy()]
            keep = pix >= 0
            unassigned += int((~keep).sum())
            n_per_pixel += np.bincount(pix[keep], minlength=n_pix)
            ch = pc.dictionary_encode(batch.column("chrom"))
            chrom_names = ch.dictionary.to_pylist()
            codes = ch.indices.to_numpy()
            rows, cols = [], []
            for site in (batch.column("start").to_numpy(), batch.column("end").to_numpy() - 1):
                for code, name in enumerate(chrom_names):
                    m = keep & (codes == code)
                    if name not in by_chrom or not m.any():
                        outside += int(m.sum())
                        continue
                    starts, ends, first = by_chrom[name]
                    p = site[m]
                    i = np.searchsorted(starts, p, side="right") - 1
                    hit = (i >= 0) & (p < ends[np.maximum(i, 0)])
                    outside += int((~hit).sum())
                    rows.append(pix[m][hit])
                    cols.append(first + i[hit])
            if rows:
                r, k = np.concatenate(rows), np.concatenate(cols)
                blocks.append(sp.csr_matrix((np.ones(len(r), np.int32), (r, k)), shape=(n_pix, n_feat)))
            if len(blocks) >= _COLLAPSE_EVERY:
                blocks = [sum(blocks[1:], blocks[0])]
    if unassigned:
        logger.info(f"{path.name}: {unassigned:,} fragments of barcodes not in the positions were dropped.")
    X = sum(blocks[1:], blocks[0]) if blocks else sp.csr_matrix((n_pix, n_feat), dtype=np.int32)
    return X.tocsr(), n_per_pixel, {"unassigned": unassigned, "outside": outside}


def _read_peak_matrix(
    path: Path, pixel: pd.Series, genome_build: str
) -> tuple[sp.csr_matrix, pd.DataFrame, pd.Index, int]:
    """Read a 10x ``*_peak_bc_matrix.h5`` (features x barcodes, CSC) as barcodes in the positions x peaks."""
    with h5py.File(path, "r") as f:
        m = f["matrix"]
        shape = tuple(m["shape"][()])
        X = sp.csc_matrix((m["data"][()], m["indices"][()], m["indptr"][()]), shape=shape).T.tocsr()
        barcodes = pd.Index([b.decode() for b in m["barcodes"][()]])
        ids = [i.decode() for i in m["features"]["id"][()]]
        genomes = {g.decode() for g in m["features"]["genome"][()]} if "genome" in m["features"] else set()
    if genomes and genomes != {genome_build}:
        logger.warning(f"{path.name} names genome {sorted(genomes)}, but genome_build is {genome_build!r}.")
    parts = pd.Series(ids).str.extract(r"^(.+):(\d+)-(\d+)$")
    var = _features(parts[0].to_numpy(), parts[1].astype(np.int64).to_numpy(), parts[2].astype(np.int64).to_numpy())
    var.index = pd.Index(ids)
    key = barcodes.str.split("-", n=1).str[0]
    in_pos = key.isin(pixel.index)
    total = np.asarray(X.sum(axis=1)).ravel()
    keep = in_pos & (total > 0)
    return X[np.flatnonzero(keep)], var, pd.Index(key[keep]), int((~in_pos).sum())
