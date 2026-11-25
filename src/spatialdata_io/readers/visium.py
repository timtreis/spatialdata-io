from __future__ import annotations

import json
import os
import re
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData, read_h5ad
from dask_image.imread import imread
from imageio import imread as imread2
from spatialdata import SpatialData
from spatialdata._logging import logger
from spatialdata.models import Image2DModel, ShapesModel, TableModel
from spatialdata.transformations.transformations import Identity, Scale
from xarray import DataArray

from spatialdata_io._constants._constants import VisiumKeys
from spatialdata_io._docs import inject_docs
from spatialdata_io.readers._utils._features import GENE_EXPRESSION, split_by_feature_type
from spatialdata_io.readers._utils._fullres import FULLRES_PLACEMENT_ATTR, fullres_placement
from spatialdata_io.readers._utils._utils import (
    _add_gene_id_and_genome,
    _read_counts,
    _set_observation_unit,
    _set_reader_metadata,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["visium"]


@inject_docs(vx=VisiumKeys)
def visium(
    path: str | Path,
    dataset_id: str | None = None,
    counts_file: str = VisiumKeys.FILTERED_COUNTS_FILE,
    fullres_image_file: str | Path | None = None,
    tissue_positions_file: str | Path | None = None,
    scalefactors_file: str | Path | None = None,
    var_names_make_unique: bool = True,
    protein_file: str | Path | None = None,
    controls: Literal["keep", "drop"] = "keep",
    barcode_rule: Mapping[str, str] | None = None,
    min_barcode_match: float = 1.0,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
    **kwargs: Any,
) -> SpatialData:
    """Read *10x Genomics* Visium formatted dataset.

    This function reads the following files:

        - ``(<dataset_id>_)`{vx.FILTERED_COUNTS_FILE!r}```: Counts and metadata file.
        - ``{vx.IMAGE_HIRES_FILE!r}``: High resolution image.
        - ``{vx.IMAGE_LOWRES_FILE!r}``: Low resolution image.
        - ``{vx.SCALEFACTORS_FILE!r}``: Scalefactors file.
        - ``{vx.SPOTS_FILE_1!r}`` (SpaceRanger 1) or ``{vx.SPOTS_FILE_2!r}`` (SpaceRanger 2):
            Spots positions file.
        - ``fullres_image_file``: large microscopy image used as input for space ranger.

    The gene expression is the table ``"table"``. A Gene and Protein (CytAssist) counts file also holds Antibody
    Capture features, which become a second table ``"table_protein"`` annotating the same spots; antibody counts in
    a separate file are read with ``protein_file``. Any other feature type becomes ``"table_<type>"``.

    The spot circles have the physical size of the capture spots: 10x states Visium spots are 55 µm in diameter, spaced
    100 µm apart. The pixels per µm follow from the spots' array positions and that 100 µm spacing. Space Ranger's
    ``spot_diameter_fullres`` is not used for the circles: 10x describes it as a visualization estimate of 60-70 µm, not
    the spot size.

    .. seealso::

        - `Space Ranger output <https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/output/overview>`_.

    Parameters
    ----------
    path
        Path to the directory containing the data.
    dataset_id
        Dataset identifier to name the constructed `SpatialData` elements. The reader will try to infer it from the
        counts_file filen name (which defaults to ``{vx.FILTERED_COUNTS_FILE!r}``) file name. If the file name does not
        contain the dataset id, it needs to be provided.
    counts_file
        Name of the counts file, defaults to ``{vx.FILTERED_COUNTS_FILE!r}``; a common alternative is
        ``{vx.RAW_COUNTS_FILE!r}``. Also, use when the file names are not following the standard SpaceRanger
        conventions.
    fullres_image_file
        Path to the full-resolution image (the one passed to Space Ranger). If fewer than 99% of the spots fall
        inside it, it is not that image (e.g. a deposited crop or the CytAssist capture image): a warning is logged
        and it is skipped.
    tissue_positions_file
        Path to the tissue positions file.
    scalefactors_file
        Path to the scalefactors file.
    var_names_make_unique
        If `True`, call `.var_names_make_unique()` on each `AnnData` table.
    protein_file
        Antibody counts deposited apart from the counts file (e.g. SPOTS ``*_ADT.csv.gz``, GSE198353, or a CytAssist
        ``*_Protein.h5ad``): a delimited table of antibodies x barcodes (either orientation) or an ``.h5ad`` with
        the barcodes as ``obs_names``. It becomes ``"table_protein"`` on the spots it shares with the counts file;
        its values are taken as deposited.
    controls
        ``"keep"`` keeps any control features in ``"table"``, ``"drop"`` removes them. Every feature is typed by
        ``var["feature_types"]``, the 10x ``feature_type`` (``Gene Expression`` where the counts file has none);
        Antibody Capture goes to ``"table_protein"`` and any other type to ``"table_<type>"``.
    barcode_rule
        How the counts file's barcodes map to the ``tissue_positions`` barcodes when a re-export spells them
        differently (e.g. ``Injury72h_AAACAAGTATCTCCCA.1`` for ``AAACAAGTATCTCCCA-1``). ``"strip_prefix"`` is a
        regular expression removed from the start of each barcode; ``"suffix_from"`` and ``"suffix_to"`` replace a
        trailing suffix. The rule is never guessed. With a rule, the deposited barcodes stay in
        ``obs["barcode_deposited"]``. The rule, the match rate and the unmatched barcodes are recorded in each table's
        ``uns["visium_barcodes"]``.
    min_barcode_match
        Share of the counts barcodes (after ``barcode_rule``) that must be in ``tissue_positions``, else this raises.
        The default 1.0 raises on any miss. Below 1.0, every counts barcode is still kept: an unmatched one has no
        position (NaN) and is listed in ``uns["visium_barcodes"]["unmatched"]``.
    imread_kwargs
        Keyword arguments passed to :func:`dask_image.imread.imread`.
    image_models_kwargs
        Keyword arguments passed to :class:`spatialdata.models.Image2DModel`.

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    path = Path(path)
    imread_kwargs = dict(imread_kwargs)
    image_models_kwargs = dict(image_models_kwargs)

    # try to infer library_id from the counts file
    library_id = None
    try:
        patt = re.compile(f".*{counts_file}")
        first_file = [i for i in os.listdir(path) if patt.match(i)][0]

        if f"_{counts_file}" in first_file:
            library_id = first_file.replace(f"_{counts_file}", "")
            counts_file = f"{library_id}_{counts_file}"
        elif counts_file == first_file:
            library_id = None
            counts_file = counts_file
        else:
            raise ValueError(
                f"Cannot determine the library_id. Expecting a file with format (<library_id>_)"
                f"{counts_file}. If the files have been renamed you may need to specify their file "
                f"names (not their paths), with some of the following arguments: `counts_file`, `fullres_image_file`, "
                f"`tissue_positions_file`, `scalefactors_file` arguments."
            )
    except IndexError as e:
        if counts_file is None:
            logger.error(
                f"{e}. \nError is due to the fact that the library id could not be found, if the counts file is `.mtx` "
                f"(or else), Please provide a `counts_file`.",
            )
            raise e
    assert counts_file is not None

    if library_id is None and dataset_id is None:
        raise ValueError(
            "Cannot determine the `library_id`. Please provide `dataset_id`; the `dataset_id` value will be used to "
            "name the elements in the `SpatialData` object."
        )

    if dataset_id is not None:
        if dataset_id != library_id and library_id is not None:
            logger.warning(
                f"`dataset_id: {dataset_id}` does not match `library_id: {library_id}`. `dataset_id: {dataset_id}` "
                f"will be used to build SpatialData."
            )
        library_id = dataset_id
    else:
        dataset_id = library_id
    assert dataset_id is not None

    # The second element of the returned tuple is the full library as contained in the metadata of
    # VisiumKeys.FILTERED_COUNTS_FILE. For instance, for the spatialdata-sandbox/visium dataset it is:
    #     spaceranger100_count_30458_ST8059048_mm10-3_0_0_premrna
    # We discard this value and use the one inferred from the filename of VisiumKeys.FILTERED_COUNTS_FILE, or the one
    # provided by the user in dataset_id
    if counts_file.endswith((".h5", ".mtx.gz")):
        kwargs.setdefault("gex_only", False)  # keep Antibody Capture features for `table_protein`
    if counts_file.endswith(".mtx.gz"):
        kwargs.setdefault("make_unique", False)  # names are made unique per table below
    adata, _ = _read_counts(path, counts_file=counts_file, library_id=library_id, **kwargs)

    if (path / "spatial" / VisiumKeys.SPOTS_FILE_1).exists() or (
        tissue_positions_file is not None and str(VisiumKeys.SPOTS_FILE_1) in str(tissue_positions_file)
    ):
        tissue_positions_file = (
            path / "spatial" / VisiumKeys.SPOTS_FILE_1
            if tissue_positions_file is None
            else path / tissue_positions_file
        )
    elif (path / "spatial" / VisiumKeys.SPOTS_FILE_2).exists() or (
        tissue_positions_file is not None and str(VisiumKeys.SPOTS_FILE_2) in str(tissue_positions_file)
    ):
        tissue_positions_file = (
            path / "spatial" / VisiumKeys.SPOTS_FILE_2
            if tissue_positions_file is None
            else path / tissue_positions_file
        )
    else:
        raise ValueError(f"Cannot find `tissue_positions` file in `{path}`.")
    coords = pd.read_csv(tissue_positions_file, header=None, index_col=0)

    if tissue_positions_file.name == VisiumKeys.SPOTS_FILE_1:
        # spaceranger_version < 2.0.0 but header detected: https://github.com/scverse/spatialdata-io/issues/146
        if "in_tissue" in coords.iloc[0].tolist():
            coords = pd.read_csv(tissue_positions_file, header=0, index_col=0)
        else:
            assert "in_tissue" not in coords.columns
            coords.columns = ["in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
    else:
        assert tissue_positions_file.name == VisiumKeys.SPOTS_FILE_2
        coords = pd.read_csv(tissue_positions_file, header=0, index_col=0)

    assert isinstance(adata.obs, pd.DataFrame)
    barcodes = _match_barcodes(
        adata, coords.index, barcode_rule, min_barcode_match, f"{counts_file} vs {tissue_positions_file.name}"
    )
    adata.obs = pd.merge(adata.obs, coords, how="left", left_index=True, right_index=True)
    # `coords` above is the raw `tissue_positions` table; the circles are built from the
    # spot coordinates in the order of `adata`, so keep them in a separate variable
    spot_coords = adata.obs[[VisiumKeys.SPOTS_X, VisiumKeys.SPOTS_Y]].to_numpy()
    adata.obsm["spatial"] = spot_coords
    adata.obs = pd.DataFrame(adata.obs)
    adata.obs.drop(columns=[VisiumKeys.SPOTS_X, VisiumKeys.SPOTS_Y], inplace=True)
    adata.obs["spot_id"] = np.arange(len(adata))
    var_names = adata.var_names.copy()  # antibodies share names with their genes (CD163), so dedupe per table
    adata.var_names_make_unique()

    if not adata.obs_names.is_unique:
        logger.info("Non-unique obs names detected, calling `obs_names_make_unique`.")
        # This is required for napari-spatialdata because of the join operation that would otherwise fail
        adata.obs_names_make_unique()

    if (path / "spatial" / VisiumKeys.SCALEFACTORS_FILE).exists() or (
        scalefactors_file is not None and (path / scalefactors_file).exists()
    ):
        scalefactors_file = (
            path / "spatial" / VisiumKeys.SCALEFACTORS_FILE if scalefactors_file is None else path / scalefactors_file
        )
        scalefactors = json.loads(scalefactors_file.read_bytes())
    else:
        # e.g. GSE263617: the hires/lowres images cannot be placed without their scale factors
        scalefactors = {}
        logger.warning(f"No scale factors file in {path}: the hires/lowres images are skipped.")
    # the circles are the physical spots, placed by the array (spots 100 µm apart), never by spot_diameter_fullres
    placed = ~np.isnan(spot_coords.astype(float)).any(axis=1)  # unmatched barcodes (min_barcode_match < 1) have none
    px_per_um = _px_per_um(adata.obs[["array_col", "array_row"]].to_numpy(float)[placed], spot_coords[placed])

    transform_original = Identity()
    transforms = {
        res: Scale(np.array([scalefactors[key], scalefactors[key]]), axes=("y", "x"))
        for res, key in (("hires", VisiumKeys.SCALEFACTORS_HIRES), ("lowres", VisiumKeys.SCALEFACTORS_LOWRES))
        if key in scalefactors
    }
    shapes = {}
    circles = ShapesModel.parse(
        spot_coords,
        geometry=0,
        radius=_SPOT_DIAMETER_UM / 2.0 * px_per_um,
        index=adata.obs["spot_id"].copy(),
        transformations={dataset_id: transform_original}
        | {f"{dataset_id}_downscaled_{res}": t for res, t in transforms.items()},
    )
    shapes[dataset_id] = circles
    adata.obs["region"] = dataset_id
    if "feature_types" not in adata.var:
        adata.var["feature_types"] = GENE_EXPRESSION
    adata.var["_name"] = var_names.astype(str).to_numpy()
    tables = split_by_feature_type(adata, controls=controls)
    for t in tables.values():
        t.var_names = t.var.pop("_name").tolist()
    if protein_file is not None:
        if "table_protein" in tables:
            raise ValueError(f"{counts_file} already holds Antibody Capture features; drop `protein_file`.")
        tables["table_protein"] = _read_protein(path / protein_file, adata.obs)
    if "table" in tables:
        _add_gene_id_and_genome(tables["table"])
    for key, t in tables.items():
        t.uns["visium_barcodes"] = barcodes
        _set_observation_unit(t, None, 55)
        tables[key] = TableModel.parse(t, region=dataset_id, region_key="region", instance_key="spot_id")
        if var_names_make_unique:
            tables[key].var_names_make_unique()

    images = {}
    placement = None  # the placement check of the given full-resolution image, recorded in sdata.attrs
    if fullres_image_file is not None:
        fullres_image_file = path / Path(fullres_image_file)
        if fullres_image_file.exists():
            image = _read_image(fullres_image_file, imread_kwargs)
            hires_file = path / VisiumKeys.IMAGE_HIRES_FILE
            hires_scalef = scalefactors.get(VisiumKeys.SCALEFACTORS_HIRES)
            in_tissue = adata.obs["in_tissue"].to_numpy() == 1 if "in_tissue" in adata.obs else None
            placement = fullres_placement(
                image.shape[-2:],
                spot_coords,
                in_tissue,
                hires_scalef,
                _read_png(hires_file, imread_kwargs).shape[-2:] if hires_file.exists() else None,
            )
            placement = {"file": fullres_image_file.name, **placement}
            reason = placement["reason"]
            if reason is not None:
                # e.g. a deposited crop or the CytAssist capture image: not the frame the spots were measured in
                logger.warning(
                    f"{fullres_image_file.name} ({image.shape[-1]} x {image.shape[-2]} px) is not the full-resolution "
                    f"image the spot coordinates refer to ({reason}); skipping it."
                )
            else:
                full_image = DataArray(image, dims=("c", "y", "x"))
                images[dataset_id + "_full_image"] = Image2DModel.parse(
                    full_image,
                    scale_factors=[2, 2, 2, 2],
                    transformations={dataset_id: transform_original},
                    rgb=None,
                    **image_models_kwargs,
                )
        else:
            logger.warning(f"File {fullres_image_file} does not exist, skipping...")

    for res, file in (("hires", VisiumKeys.IMAGE_HIRES_FILE), ("lowres", VisiumKeys.IMAGE_LOWRES_FILE)):
        if res in transforms and (path / file).exists():
            images[f"{dataset_id}_{res}_image"] = Image2DModel.parse(
                DataArray(_read_png(path / file, imread_kwargs), dims=("c", "y", "x")),
                transformations={f"{dataset_id}_downscaled_{res}": Identity(), dataset_id: transforms[res].inverse()},
                rgb=None,
            )

    sdata = SpatialData(images=images, shapes=shapes, tables=tables)
    if placement is not None:
        sdata.attrs[FULLRES_PLACEMENT_ATTR] = placement
    return _set_reader_metadata(sdata, "visium")


_BARCODE_RULE_KEYS = frozenset({"strip_prefix", "suffix_from", "suffix_to"})


def _match_barcodes(
    adata: AnnData, positions: pd.Index, rule: Mapping[str, str] | None, min_match: float, what: str
) -> dict[str, Any]:
    """Apply the declared ``rule`` to ``adata.obs_names`` in place; raise below ``min_match`` barcodes in ``positions``.

    Space Ranger's positions list every spot of the array, so a counts barcode outside them is a spelling mismatch
    (or a foreign barcode) and its spot has no coordinates.
    """
    rule = dict(rule or {})
    if unknown := set(rule) - _BARCODE_RULE_KEYS:
        raise ValueError(f"Unknown barcode_rule keys {sorted(unknown)}; expected {sorted(_BARCODE_RULE_KEYS)}.")
    if ("suffix_from" in rule) != ("suffix_to" in rule):
        raise ValueError("barcode_rule needs both `suffix_from` and `suffix_to`.")
    if rule:
        deposited = adata.obs_names.astype(str)
        names = (
            deposited.str.replace(f"^(?:{rule['strip_prefix']})", "", regex=True)
            if "strip_prefix" in rule
            else deposited
        )
        if "suffix_from" in rule:
            sfx = names.str.endswith(rule["suffix_from"])
            names = names.where(~sfx, names.str[: -len(rule["suffix_from"])] + rule["suffix_to"])
        if (names == deposited).all():
            raise ValueError(f"barcode_rule {rule} changes no barcode of {what} (e.g. {deposited[0]!r}).")
        if not names.is_unique:
            raise ValueError(f"barcode_rule {rule} maps distinct barcodes of {what} to the same name.")
        adata.obs["barcode_deposited"] = deposited.to_numpy()
        adata.obs_names = names
    matched = adata.obs_names.isin(positions)
    rate = float(matched.mean()) if len(matched) else 1.0
    missing = adata.obs_names[~matched]
    if rate < min_match:
        raise ValueError(
            f"{len(missing)} of {adata.n_obs} counts barcodes are not in tissue_positions ({what}), e.g. "
            f"{missing[0]!r} (positions: {positions[0]!r}); match rate {rate:.4f} < min_barcode_match {min_match}. "
            "Declare how they map with `barcode_rule`."
        )
    return {
        "rule": rule,
        "min_barcode_match": float(min_match),
        "match_rate": rate,
        "n_counts_barcodes": int(adata.n_obs),
        "n_matched": int(matched.sum()),
        "unmatched": {"n": len(missing), "barcodes": [str(b) for b in missing]},
    }


def _read_protein(file: Path, spots: pd.DataFrame) -> AnnData:
    """Antibodies on the counts file's spots (``spots`` is its ``obs``), oriented by the barcodes."""
    if file.suffix == ".h5ad":
        p = read_h5ad(file)
        df = p.to_df()
    else:
        df = pd.read_csv(file, sep="\t" if ".tsv" in file.suffixes else ",", index_col=0)
        if df.columns.isin(spots.index).sum() > df.index.isin(spots.index).sum():
            df = df.T  # antibodies x barcodes
    common = spots.index[spots.index.isin(df.index)]
    if not len(common):
        raise ValueError(f"No barcode of {file.name} is a spot of the counts file.")
    return AnnData(
        sp.csr_matrix(df.loc[common].to_numpy()),
        obs=spots.loc[common].copy(),
        var=pd.DataFrame(index=df.columns.astype(str)),
    )


def _read_png(image_file: Path, imread_kwargs: dict[str, Any]) -> Any:
    """A hires/lowres image as ``(c, y, x)``; grayscale PNGs (e.g. GSE198353) have no channel axis."""
    im = imread(image_file, **imread_kwargs).squeeze()
    return im[None] if im.ndim == 2 else im.transpose(2, 0, 1)


def _read_image(image_file: Path, imread_kwargs: dict[str, Any]) -> Any:
    if "MAX_IMAGE_PIXELS" in imread_kwargs:
        from PIL import Image as ImagePIL

        ImagePIL.MAX_IMAGE_PIXELS = imread_kwargs.pop("MAX_IMAGE_PIXELS")
    if image_file.suffix.lower() == ".ndpi":  # Hamamatsu (E-MTAB-13024): a TIFF variant, full resolution is series 0
        import tifffile

        im = tifffile.imread(image_file)
    elif image_file.suffix != ".btf":
        im = imread(image_file, **imread_kwargs)
    else:
        # dask_image doesn't recognize .btf automatically
        im = imread2(image_file, **imread_kwargs)
        # Depending on the versions of the pipeline, the axes of the image file from the tiff data is ordered in
        # different ways; here let's implement a simple check on the shape to determine the axes ordering.
        # Note that a more robust check could be implemented; this could be the work of a future PR. Unfortunately,
        # the tif data does not (or does not always) have OME metadata, so even such more general parser could lead
        # to edge cases that could be addressed by a more interoperable file format.
    if len(im.shape) not in [3, 4]:
        raise ValueError(f"Image shape {im.shape} is not supported.")
    if len(im.shape) == 4:
        if im.shape[0] == 1:
            im = im.squeeze(0)
        else:
            raise ValueError(f"Image shape {im.shape} is not supported.")
    # for immunofluerence images there could be an arbitrary number of channels (usually, 2, 3 or 4); we can detect this
    # as the dimension which has the minimum size
    min_size = np.argmin(im.shape)
    if min_size == 0:
        image = im
    elif min_size == 2:
        image = im.transpose(2, 0, 1)
    else:
        raise ValueError(f"Image shape {im.shape} is not supported.")
    return image


# Visium array (10x Space Ranger outputs docs, "Tissue positions"): array_col "uses even numbers ... for even rows, and
# odd numbers ... for odd rows", so spots next to each other in a row are 2 columns apart, and the 10x glossary states the
# spots are "spaced 100 µm apart": one array_col step is 50 µm. The row spacing is not used (on Space Ranger output
# the row step is 1.74 column steps, not the sqrt(3) of an ideal hexagon).
_ARRAY_COL_UM = 50.0
# physical diameter of a Visium v1/v2 capture spot (10x glossary: "55 µm in diameter, spaced 100 µm apart")
_SPOT_DIAMETER_UM = 55.0


def _px_per_um(array_col_row: np.ndarray, spot_xy: np.ndarray) -> float:
    """Full-resolution pixels per µm, from the spots' array positions and pixel positions.

    A least-squares affine fit of the pixel positions on ``(array_col, array_row)`` gives the pixel step of one array
    column, which is 50 µm. Raises if the array is not Visium's, where a spot's column and row have the same parity.
    """
    col, row = array_col_row[:, 0].astype(int), array_col_row[:, 1].astype(int)
    if np.any((col - row) % 2):
        raise ValueError(
            "The spots are not on a Visium v1/v2 array (array_col and array_row of a spot must be both even or both "
            "odd), so the pixel size cannot be derived from it."
        )
    design = np.column_stack([array_col_row, np.ones(len(array_col_row))])
    coef, *_ = np.linalg.lstsq(design, spot_xy, rcond=None)
    px_per_col = float(np.linalg.norm(coef[0]))
    if px_per_col == 0:
        raise ValueError("Cannot derive the pixel size from spots that all share one array column.")
    return px_per_col / _ARRAY_COL_UM
