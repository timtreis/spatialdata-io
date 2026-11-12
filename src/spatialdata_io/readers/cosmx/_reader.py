"""CosMx reader for spatialdata.

Reads NanoString CosMx spatial transcriptomics data into a
:class:`spatialdata.SpatialData` object.  Supports all known CosMx export
formats (AtomX flat files, nested CellStatsDir layouts, multimodal
RNA+Protein runs) and transparently handles stitching, polygon
rasterization, and coordinate normalization.

Public API
----------
cosmx
    Top-level reader function (single or multimodal datasets).
CosMxDataset
    Frozen dataclass describing the on-disk layout of a CosMx dataset.
CosMxDatasetReader
    Stateful reader that orchestrates per-element I/O.
"""

from __future__ import annotations

import json
import math
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import dask.array as da
import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry as sgeom
import tifffile
from shapely.affinity import translate as _translate
from spatialdata import SpatialData, sanitize_table
from spatialdata._logging import logger
from spatialdata.models import (
    Image2DModel,
    Labels2DModel,
    PointsModel,
    ShapesModel,
    TableModel,
)
from spatialdata.transformations import Scale, Sequence, Translation, get_transformation, set_transformation

from spatialdata_io._constants._constants import CosmxKeys
from spatialdata_io._docs import inject_docs
from spatialdata_io.readers._utils._features import COSMX_NAME_RULES, feature_types_by_name, split_by_feature_type
from spatialdata_io.readers._utils._utils import _set_observation_unit, _set_reader_metadata

from ._discovery import (
    _infer_dataset_id,
    _modality_from_ancestors,
    _set_up_cosmx_dataset_for_conversion,
)
from ._io import (
    COSMX_FOV_SIZE_PX,
    COSMX_PIXEL_SIZE,
    NoFovCoordinatesError,
    _default_image_kwargs,
    _find_matching_fov_file,
    _get_cosmx_morphology_coords,
    _maybe_warn_big_file,
    _parquet_cache_for_tx,
    _pixel_size_from_fov_positions,
    _read_expr_mat_polars,
    _read_fov_image,
    _read_fov_locs,
    _read_metadata_polars,
    _read_polygons_csv,
    _read_transcripts_csv,
    _set_fov_extent,
    place_local_in_fov_grid,
)
from ._stitching import (
    _canvas_from_fov_locs_for_polygons,
    _plot_fov_preview,
    _polygons_to_label_raster,
    _read_stitched_cell_labels_from_dir,
    _read_stitched_image,
    stitch_segmentation_label_image,
)
from ._utils import (
    _dask_categoricals_to_string,
    _match_canonical,
    _normalize_image_channels,
    _pandas_categoricals_to_string,
    detect_fovs_with_data,
    find_cell_label_tifs,
)

if TYPE_CHECKING:
    from anndata import AnnData

__all__ = ["cosmx"]

PROTEIN = "Protein Expression"
# protein exprMat columns no vendor source types (seen in sddb_jm45slqy3e): the Channel-* stains, Ms IgG1, Rb IgG
_COSMX_PROTEIN_UNVERIFIED = ((re.compile(r"^Channel-|^(Ms|Rb) IgG"), "unverified"),)

# ---------------------------------------------------------------------------
# Dataset descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class CosMxDataset:
    """Frozen descriptor of a CosMx dataset directory layout.

    Automatically infers ``dataset_id`` from marker-file prefixes on disk.
    For multimodal runs (e.g. S3RNA + S3Protein), the ``modalities`` field
    maps each modality name to its own ``CosMxDataset``.
    """

    path: Path = field(default=Path("."), metadata={"help": "Root directory."})
    dataset_id: str | None = field(default=None)
    exprMat_file: Path | None = field(default=None)
    fov_positions_file: Path | None = field(default=None)
    metadata_file: Path | None = field(default=None)
    tx_file: Path | None = field(default=None)
    polygons_file: Path | None = field(default=None)
    analysis_results_dir: Path | None = field(default=None)
    cell_stats_dir: Path | None = field(default=None)
    run_summary_dir: Path | None = field(default=None)
    cell_composite_dir: Path | None = field(default=None)
    cell_labels_dir: Path | None = field(default=None)
    cell_overlay_dir: Path | None = field(default=None)
    celltype_accessory_data: Path | None = field(default=None)
    compartment_labels_dir: Path | None = field(default=None)
    protein_dir: Path | None = field(default=None)
    morphology_2d_dir: Path | None = field(default=None)
    morphology_3d_dir: Path | None = field(default=None)
    raw_morphology_dir: Path | None = field(default=None, metadata={"help": "Read only for the TIFF pixel size."})
    modalities: dict[str, CosMxDataset] | None = field(default=None)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", self.path.resolve())
        if self.modalities:
            if not isinstance(self.modalities, dict) or not self.modalities:
                raise ValueError("modalities must be a non-empty dict.")
            return
        # At least one data file must be present.
        has_data = any(
            val is not None for name, val in self.__dict__.items() if name not in {"dataset_id", "modalities", "path"}
        )
        if not has_data:
            raise ValueError("At least one data file or directory must be present.")
        inferred = _infer_dataset_id(self.path, self.dataset_id)
        object.__setattr__(self, "dataset_id", inferred)

    # -- pretty-printing --

    def __str__(self) -> str:
        root = self.path
        lines = [self.__class__.__name__, f"  path: {root}", f"  dataset_id: {self.dataset_id or '<not set>'}"]
        for name, val in self.__dict__.items():
            if name in {"path", "dataset_id", "modalities"} or val is None:
                continue
            if isinstance(val, Path):
                if not val.exists():
                    continue
                try:
                    val = f"<path>/{val.resolve().relative_to(root).as_posix()}"
                except ValueError:
                    val = str(val.resolve())
                lines.append(f"  {name}: {val}")
            else:
                lines.append(f"  {name}: {val!r}")
        for mod_name, mod_ds in (self.modalities or {}).items():
            lines.append(f"  {mod_name}:")
            lines.append(textwrap.indent(str(mod_ds), "    "))
        return "\n".join(lines)

    def _repr_pretty_(self, p, cycle) -> None:
        p.text(self.__class__.__name__ + "(…)" if cycle else str(self))


# ---------------------------------------------------------------------------
# Transformation helper
# ---------------------------------------------------------------------------


def _translation_transform(ox: float, oy: float) -> Translation:
    """Translation by ``(ox, oy)`` in the global xy coordinate system."""
    return Translation([ox, oy], axes=("x", "y"))


# ---------------------------------------------------------------------------
# Element-name helpers
# ---------------------------------------------------------------------------


def _element_name(base: str, fovs: set[int] | None, *, max_listed: int = 8) -> str:
    """Build a context-dependent element name from *base* and the FOV set."""
    if not fovs:
        return base
    if len(fovs) == 1:
        return f"F{next(iter(fovs)):05d}_{base}"
    sorted_fovs = sorted(fovs)
    if len(sorted_fovs) <= max_listed:
        return "subset_" + "_".join(f"F{f:05d}" for f in sorted_fovs) + f"_{base}"
    return base


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------


def _global_id_series(table: AnnData) -> pd.Series | None:
    """Return the (Int64-coerced) ``global_cell_id`` column of a table, or ``None``."""
    adata = table.table if hasattr(table, "table") else table
    obs = cast(pd.DataFrame, adata.obs)
    if "global_cell_id" not in obs:
        return None
    col = obs["global_cell_id"]
    return col.astype("Int64") if isinstance(col.dtype, pd.CategoricalDtype) else col


def _parse_cell_table(adata: AnnData, *, region: str | None, region_key: str | None = "region_key") -> AnnData:
    """Parse a cell-annotation table with the standard CosMx region/instance keys."""
    return TableModel.parse(
        adata, region=region, region_key=region_key, instance_key="global_cell_id", overwrite_metadata=True
    )


def _parse_labels(arr: Any, ox: float, oy: float) -> Labels2DModel:
    """Parse a ``(y, x)`` label raster with a global translation by ``(ox, oy)``."""
    return Labels2DModel.parse(arr, dims=("y", "x"), transformations={"global": _translation_transform(ox, oy)})


def _count_cells_outside_labels(
    tables: dict[str, AnnData],
    valid_ids: set[int],
    *,
    source_label: str = "label IDs",
) -> dict[str, int]:
    """Count, per table, the rows whose ``global_cell_id`` is not in *valid_ids*, and warn about them.

    Every row is kept: a cell of the expression matrix is never dropped because it is missing from the labels.
    """
    counts: dict[str, int] = {}
    if not valid_ids:
        return counts
    for name, table in tables.items():
        col = _global_id_series(table)
        if col is None:
            continue
        n = int((~col.isin(valid_ids)).sum())
        counts[name] = n
        if n:
            logger.warning(
                "Table %s: %d of %d cell(s) are not in the %s; they are kept in the table.",
                name,
                n,
                len(col),
                source_label,
            )
    return counts


# ---------------------------------------------------------------------------
# CosMxDatasetReader
# ---------------------------------------------------------------------------


class CosMxDatasetReader:
    """Stateful reader that orchestrates reading of all CosMx elements.

    Manages FOV subsetting, coordinate origin normalization,
    global_cell_id computation, and lazy polygon caching.

    Parameters
    ----------
    dataset
        A :class:`CosMxDataset` describing the files on disk.
    fovs
        Optional set of FOV IDs to read (``None`` = all).
    n_workers
        Number of parallel workers for dask/zarr operations.
    flip_image
        Whether to flip images vertically.
    polygons_as_labels
        If ``True``, rasterize polygons into a label image.
    fov_locs
        Pre-parsed FOV positions (skips re-reading the CSV).
    align_rasters_to_polygons
        If ``True``, anchor images/labels to the polygon coordinate origin.
    keep_polygons_after_rasterize
        If ``True``, keep vector polygons even when rasterizing to labels.
    fov_size
        The run's FOV ``(width, height)`` in px (:func:`_fov_size_px`); every FOV is placed with it.
    fov_size_exact
        ``False`` when *fov_size* is only a lower bound (``"local_px_lower_bound"``): then it sizes the FOV boxes
        only, never a canvas or a y-flip height.
    """

    def __init__(
        self,
        dataset: CosMxDataset,
        *,
        fovs: set[int] | None = None,
        n_workers: int | None = None,
        flip_image: bool | None = None,
        polygons_as_labels: bool = False,
        fov_locs: pd.DataFrame | None = None,
        align_rasters_to_polygons: bool | None = None,
        keep_polygons_after_rasterize: bool = False,
        image_normalization_percentile: float | None = None,
        fov_size: tuple[float, float] | None = None,
        fov_size_exact: bool = True,
    ) -> None:
        self.dataset = dataset
        self.fov_size = fov_size
        self.fov_size_exact = fov_size_exact
        self.fovs = fovs
        self.n_workers = n_workers or 1
        self.flip_image = bool(flip_image) if flip_image is not None else False
        self.polygons_as_labels = polygons_as_labels
        self.keep_polygons_after_rasterize = keep_polygons_after_rasterize
        if image_normalization_percentile is not None:
            if not 0.0 <= image_normalization_percentile <= 100.0:
                raise ValueError(
                    f"image_normalization_percentile must be in [0, 100] or None, got "
                    f"{image_normalization_percentile!r}."
                )
            if image_normalization_percentile < 1.0:
                logger.warning(
                    "image_normalization_percentile=%s is < 1; percentiles are 0-100 — did you mean e.g. 99.9?",
                    image_normalization_percentile,
                )
        self.image_normalization_percentile = image_normalization_percentile
        self._image_norm_logged = False

        # Read or reuse FOV positions
        if fov_locs is not None:
            self.fov_locs = fov_locs
        elif dataset.fov_positions_file is not None:
            self.fov_locs = _read_fov_locs(
                dataset.fov_positions_file,
                fovs=sorted(fovs) if fovs else None,
                fov_size=self._placement_size,
            )
        else:
            raise ValueError("FOV positions file is required to read CosMx data.")

        self._base_origin_x = float(self.fov_locs["xmin"].min())
        self._base_origin_y = float(self.fov_locs["ymin"].min())

        # Polygon cache and origin (set lazily by _get_polygons)
        self._poly_df: pd.DataFrame | None = None
        self._origin_x: float | None = None
        self._origin_y: float | None = None

        if align_rasters_to_polygons is None:
            align_rasters_to_polygons = polygons_as_labels
        self.align_rasters_to_polygons = bool(align_rasters_to_polygons)

        # Frozen after first use to ensure consistent IDs across elements
        self.max_cell_id: int | None = None

    # -- Polygon loading and origin computation --

    def _get_polygons(self) -> pd.DataFrame:
        """Load polygons (cached). Sets ``_origin_x`` / ``_origin_y``."""
        if self._poly_df is not None:
            return self._poly_df
        if self.dataset.polygons_file is None:
            raise FileNotFoundError("Dataset has no polygons_file.")

        poly_df = _read_polygons_csv(
            self.dataset.polygons_file,
            fov_locs=self.fov_locs,
            n_workers=self.n_workers,
            fov_set=self.fovs,
            use_polars=True,
            fov_height=self._fov_height,
        )
        poly_df["global_cell_id"] = self.global_cell_id(poly_df)

        # Compute origin from actual polygon bounds
        xs = [float(g.bounds[0]) for g in poly_df["geometry"] if g is not None and not g.is_empty]
        ys = [float(g.bounds[1]) for g in poly_df["geometry"] if g is not None and not g.is_empty]
        ox = min(xs) if xs else self._base_origin_x
        oy = min(ys) if ys else self._base_origin_y

        # Shift polygons to origin
        if ox != 0.0 or oy != 0.0:
            poly_df["geometry"] = [
                _translate(g, xoff=-ox, yoff=-oy) if g is not None else None for g in poly_df["geometry"]
            ]
            poly_df = poly_df.dropna(subset=["geometry"]).reset_index(drop=True)

        self._origin_x = ox
        self._origin_y = oy
        self._poly_df = poly_df
        return self._poly_df

    @property
    def _placement_size(self) -> tuple[float, float] | None:
        """The FOV size for canvases and y flips: ``None`` when only a lower bound is known."""
        return self.fov_size if self.fov_size_exact else None

    @property
    def _fov_height(self) -> float | None:
        return float(self._placement_size[1]) if self._placement_size is not None else None

    def _ensure_origin(self) -> tuple[float, float]:
        """Return the coordinate origin, loading polygons if needed."""
        if self._origin_x is not None and self._origin_y is not None:
            return self._origin_x, self._origin_y
        if self.dataset.polygons_file is not None and (self.align_rasters_to_polygons or self.polygons_as_labels):
            self._get_polygons()
            return self._origin_x, self._origin_y  # type: ignore[return-value]
        self._origin_x = self._base_origin_x
        self._origin_y = self._base_origin_y
        return self._origin_x, self._origin_y

    # -- Global cell ID --

    def _lock_max_cell_id(self, current_max: int) -> int:
        """Lock or validate ``max_cell_id`` and return the base multiplier.

        On first call, locks ``max_cell_id`` to *current_max*.  On subsequent
        calls, validates that *current_max* does not exceed the locked value.

        Returns ``max_cell_id + 1`` (the base used in the ID formula).
        """
        if self.max_cell_id is None:
            self.max_cell_id = current_max
        elif current_max > self.max_cell_id:
            raise ValueError(
                f"cell_ID up to {current_max} but reader locked to "
                f"max_cell_id={self.max_cell_id}. Load the file with the "
                "largest per-FOV cell_ID first (usually the polygon CSV)."
            )
        return self.max_cell_id + 1

    def global_cell_id(self, df: pd.DataFrame) -> pd.Series:
        """Compute ``global_cell_id = fov * (max_cell_id + 1) * (cell_ID > 0) + cell_ID``.

        Locks ``max_cell_id`` on first call for consistency across elements.
        """
        if "fov" not in df.columns or "cell_ID" not in df.columns:
            raise ValueError("Expected columns 'fov' and 'cell_ID'.")
        fov_ser = pd.to_numeric(df["fov"], errors="coerce").fillna(0).astype(int)
        cell_ser = pd.to_numeric(df["cell_ID"], errors="coerce").fillna(0).astype(int)
        current_max = int(cell_ser.max()) if len(cell_ser) else 0
        base = self._lock_max_cell_id(current_max)
        return fov_ser * base * (cell_ser > 0).astype(int) + cell_ser

    # -- Read images --

    def read_images(
        self,
        *,
        read_proteins: bool,
        image_models_kwargs: dict[str, Any],
        imread_kwargs: dict[str, Any],
        channels: list[str] | None = None,
    ) -> dict[str, Image2DModel]:
        """Read morphology (and optionally protein) images."""
        if self.dataset.morphology_2d_dir is None:
            return {}

        images_dir = self.dataset.morphology_2d_dir
        morphology_coords = _get_cosmx_morphology_coords(images_dir)

        if self.align_rasters_to_polygons and self.dataset.polygons_file is not None:
            self._get_polygons()
        ox, oy = self._ensure_origin()

        # Morphology TIFFs are stored y-inverted relative to the FOV-grid
        # placement, so they need a vertical flip to co-register with
        # transcripts/labels.  This must NOT depend on whether polygons were
        # read — gating it on ``align_rasters_to_polygons`` left px-only images
        # mirrored when read without polygons (issue #42).  ``flip_image``
        # defaults to True (a user may pass ``flip_image=False`` for a dataset
        # stored the other way).
        per_fov_flip = self.flip_image

        protein_dirs: dict[int, Path] = {}
        if read_proteins and self.dataset.analysis_results_dir is not None:
            protein_dirs = {
                int(p.parent.name[3:]): p for p in self.dataset.analysis_results_dir.rglob("**/FOV*/ProteinImages")
            }

        transform = _translation_transform(ox, oy)

        # Single FOV — no stitching
        if self.fovs and len(self.fovs) == 1:
            fov = next(iter(self.fovs))
            try:
                fov_path = _find_matching_fov_file(images_dir, fov)
            except FileNotFoundError:
                logger.warning("No image file found for FOV %d — skipping images.", fov)
                return {}
            image, c_coords = _read_fov_image(
                fov_path,
                protein_dirs.get(fov),
                morphology_coords,
                selected_channels=channels,
                **imread_kwargs,
            )
            if per_fov_flip:
                image = image[:, ::-1, :]
            return {
                _element_name("image", self.fovs): self._normalize_and_parse_image(
                    image, c_coords, transform, image_models_kwargs
                )
            }

        # Multi-FOV — stitch
        fovs_filter = self.fovs if self.fovs else None
        sel_locs = self.fov_locs.loc[sorted(self.fovs)] if self.fovs else self.fov_locs
        prot_subset = {k: v for k, v in protein_dirs.items() if k in self.fovs} if self.fovs else protein_dirs

        # Tighten the image canvas to image-bearing FOVs (issue #37), but only
        # when the base FOV origin governs; when polygons drive the origin keep
        # it so the image stays co-registered with transcripts/labels.
        tighten = not (
            self.dataset.polygons_file is not None and (self.align_rasters_to_polygons or self.polygons_as_labels)
        )
        stitched, c_coords, used_ox, used_oy = _read_stitched_image(
            images_dir,
            sel_locs,
            prot_subset,
            morphology_coords,
            flip_image=per_fov_flip,
            fovs_filter=fovs_filter,
            selected_channels=channels,
            n_workers=self.n_workers,
            tighten_to_seen=tighten,
            fov_size=self._placement_size,
            **imread_kwargs,
        )
        if stitched is None:
            return {}
        img_transform = _translation_transform(used_ox, used_oy) if tighten else transform
        name = "stitched_image" if not self.fovs else _element_name("image", self.fovs)
        return {name: self._normalize_and_parse_image(stitched, c_coords, img_transform, image_models_kwargs)}

    def _normalize_and_parse_image(
        self,
        image: da.Array,
        c_coords: list[str],
        transform: Any,
        image_models_kwargs: dict[str, Any],
    ) -> Image2DModel:
        """Parse the assembled image, optionally applying a per-channel percentile stretch.

        Post-stitch; the reversible per-channel divisors are recorded in the element's
        ``attrs`` for promotion into ``sdata.attrs`` at assembly.
        """
        pct = self.image_normalization_percentile
        image, scales = _normalize_image_channels(image, c_coords, percentile=pct, n_workers=self.n_workers)
        stretched = any(div != 1.0 for div in scales.values())  # all-1.0 == nothing applied
        if stretched and not self._image_norm_logged:
            logger.info(
                "Applying per-channel %.4g-th-percentile image normalization (scale-only, reversible via attrs).",
                pct,
            )
            self._image_norm_logged = True
        parsed = Image2DModel.parse(
            image,
            dims=("c", "y", "x"),
            c_coords=c_coords,
            transformations={"global": transform},
            **image_models_kwargs,
        )
        if stretched:
            parsed.attrs["cosmx_image_normalization"] = {"percentile": pct, "channel_scales": scales}
        return parsed

    # -- Read transcripts --

    def read_transcripts(self) -> dict[str, PointsModel]:
        """Read transcript coordinates into a PointsModel."""
        if self.dataset.tx_file is None:
            return {}
        src = self.dataset.tx_file
        if not src.exists() or not src.is_file():
            raise FileNotFoundError(f"Transcript file not found: {src}")

        is_big = _maybe_warn_big_file(src)

        if is_big:
            return self._read_transcripts_parquet(src)
        return self._read_transcripts_csv(src)

    def _place_transcripts(self, df):
        """Map transcripts into the stitched FOV grid.

        Transcripts and polygons share the same per-FOV local coordinate
        system, so we place transcript points with the exact mapping used
        for polygon vertices (``fov_locs`` origin + ``flip_y``).  This
        co-registers transcripts with polygons/images/labels by construction
        on every dataset, regardless of flip or any global-frame offset.

        Falls back to the raw ``x_global_px`` / ``y_global_px`` columns when
        the per-FOV local columns are not available.
        """
        if {"x_local_px", "y_local_px", "fov"}.issubset(df.columns):
            return place_local_in_fov_grid(df, self.fov_locs, fov_height=self._fov_height)
        logger.warning(
            "[transcripts] No local px columns found — using raw global "
            "coordinates, which may not align with images/polygons."
        )
        return df

    def _finalize_transcripts(self, frame: Any, name: str, coord_map: dict[str, str]) -> dict[str, PointsModel]:
        """Parse a placed transcripts frame into a PointsModel at the global origin."""
        ox, oy = self._ensure_origin()
        return {
            name: PointsModel.parse(
                frame,
                coordinates=coord_map,
                feature_key=CosmxKeys.TARGET_OF_TRANSCRIPT,
                transformations={"global": _translation_transform(ox, oy)},
            )
        }

    def _read_transcripts_parquet(self, src: Path) -> dict[str, PointsModel]:
        """Read large transcripts via Parquet cache + Dask."""
        pq_file = _parquet_cache_for_tx(src, self.dataset.dataset_id, n_workers=self.n_workers)
        logger.info("[transcripts] Reading from Parquet cache...")
        df = dd.read_parquet(pq_file, engine="pyarrow")

        if self.fovs:
            df = df[df["fov"].isin(list(self.fovs))]

        df = self._place_transcripts(df)

        if {"fov", "cell_ID"}.issubset(set(df.columns)):
            raw_max = df["cell_ID"].max().compute()
            if pd.isna(raw_max):
                logger.warning("[transcripts] No transcripts found for the requested FOVs — skipping.")
                return {}
            current_max = int(raw_max)
            base = self._lock_max_cell_id(current_max)
            df = df.assign(global_cell_id=df["fov"] * base * (df["cell_ID"] > 0) + df["cell_ID"])

        df = _dask_categoricals_to_string(df)
        name, coord_map = self._transcript_naming(df)

        if isinstance(df, dd.DataFrame):
            # Apply coord transforms for Dask
            if "x" not in df.columns:
                ox, oy = self._ensure_origin()
                df = df.assign(x=df["x_global_px"] - ox, y=df["y_global_px"] - oy)

        return self._finalize_transcripts(df, name, coord_map)

    def _read_transcripts_csv(self, src: Path) -> dict[str, PointsModel]:
        """Read small transcripts directly from CSV."""
        logger.info("[transcripts] Reading %s into memory...", src.name)
        df = _read_transcripts_csv(src, self.dataset.dataset_id)
        if self.fovs:
            df = df[df["fov"].isin(list(self.fovs))]
        if df.empty:
            logger.warning("[transcripts] No transcripts found for the requested FOVs — skipping.")
            return {}

        df = self._place_transcripts(df)

        df["global_cell_id"] = self.global_cell_id(df)

        name, coord_map = self._transcript_naming(df)
        _pandas_categoricals_to_string(df)

        # Convert to Dask before PointsModel.parse to avoid a partition
        # alignment error in spatialdata when it independently converts the
        # feature column and the coordinate columns to separate Dask objects.
        ddf = dd.from_pandas(df, npartitions=max(1, len(df) // 2_000_000))
        return self._finalize_transcripts(ddf, name, coord_map)

    def _transcript_naming(self, df) -> tuple[str, dict[str, str]]:
        """Determine element name and coordinate mapping for transcripts.

        Coordinates are shifted to the shared 0-based origin used by
        images/labels/shapes so all elements align in the ``global``
        coordinate system.  (For Dask frames the shift is applied by the
        caller, since ``x`` may not yet exist here.)
        """
        ox, oy = self._ensure_origin()

        if not isinstance(df, dd.DataFrame):
            df["x"] = (df["x_global_px"] - ox).astype(float)
            df["y"] = (df["y_global_px"] - oy).astype(float)

        return _element_name("points", self.fovs), {"x": "x", "y": "y"}

    # -- Read shapes (polygons) --

    def read_shapes(self) -> dict[str, ShapesModel]:
        """Read cell polygons as vector shapes."""
        if self.dataset.polygons_file is None:
            return {}
        if self.polygons_as_labels and not self.keep_polygons_after_rasterize:
            return {}

        poly_df = self._get_polygons()
        ox, oy = cast(float, self._origin_x), cast(float, self._origin_y)  # set by _get_polygons()
        gdf = gpd.GeoDataFrame(poly_df.copy(), geometry="geometry").set_index("global_cell_id", drop=False)

        return {
            _element_name("cells_polygons", self.fovs): ShapesModel.parse(
                gdf,
                transformations={"global": _translation_transform(ox, oy)},
            )
        }

    # -- Build FOV box shapes --

    def build_fov_shapes(self) -> dict[str, ShapesModel]:
        """Build one box of the run's FOV size per FOV as shape elements."""
        if self.fov_locs is None or self.fov_locs.empty:
            return {}
        if self.fov_size is None:
            raise ValueError("FOV boxes need the run's FOV size (fov_size).")
        fov_w, fov_h = self.fov_size
        ox, oy = self._ensure_origin()

        locs = self.fov_locs.loc[sorted(self.fovs)].copy() if self.fovs else self.fov_locs.copy()

        geoms, fov_vals, skipped = [], [], []
        for fov, row in locs.iterrows():
            xmin = float(row.get("xmin", math.nan))
            ymin = float(row.get("ymin", math.nan))
            if not math.isfinite(xmin) or not math.isfinite(ymin):
                skipped.append(int(cast("int", fov)))
                continue
            x0, y0 = xmin - ox, ymin - oy
            geoms.append(sgeom.box(x0, y0, x0 + fov_w, y0 + fov_h))
            fov_vals.append(int(cast("int", fov)))

        if skipped:
            logger.warning("Skipped %d FOV(s) with NaN coords: %s", len(skipped), skipped)
        if not geoms:
            return {}

        gdf = gpd.GeoDataFrame({"fov": fov_vals, "geometry": geoms}, geometry="geometry")
        return {
            _element_name("fov_boxes", self.fovs): ShapesModel.parse(
                gdf,
                transformations={"global": _translation_transform(ox, oy)},
            )
        }

    # -- Read labels --

    def read_labels(self) -> tuple[dict[str, Labels2DModel], set[int]]:
        """Read or rasterize cell labels.

        Tries in order:
        1. Polygon rasterization (if ``polygons_as_labels`` and polygons exist)
        2. CellLabels directory (pre-computed TIFFs)
        3. Legacy CellStatsDir layout

        Returns ``(labels_dict, set_of_label_ids_present)``.
        """
        # --- Path 1: polygon → raster ---
        if self.polygons_as_labels and self.dataset.polygons_file is not None:
            return self._labels_from_polygons()

        # --- Path 2: flat CellLabels directory ---
        if self.dataset.cell_labels_dir is not None:
            return self._labels_from_cell_labels_dir()

        # --- Path 3: legacy CellStatsDir ---
        if self.dataset.cell_stats_dir is not None:
            return self._labels_from_cell_stats_dir()

        return {}, set()

    def _labels_from_polygons(self) -> tuple[dict[str, Labels2DModel], set[int]]:
        """Rasterize polygons into a label image."""
        poly_df = self._get_polygons()
        ox, oy = cast(float, self._origin_x), cast(float, self._origin_y)  # set by _get_polygons()

        if poly_df.empty:
            logger.warning("No polygons left after alignment. Skipping labels.")
            return {}, set()

        canvas_min_x, canvas_min_y, canvas_w, canvas_h = _canvas_from_fov_locs_for_polygons(
            self.fov_locs,
            origin_x=ox,
            origin_y=oy,
            fovs=self.fovs,
            fov_size=self._placement_size,
        )

        raster = _polygons_to_label_raster(
            poly_df,
            chunks=(2048, 2048),
            n_jobs=self.n_workers,
            canvas_min_x=canvas_min_x,
            canvas_min_y=canvas_min_y,
            canvas_width=canvas_w,
            canvas_height=canvas_h,
        )
        raster.attrs.pop("transform", None)

        present_ids = {int(x) for x in da.unique(raster.data).compute() if x != 0}
        name = _element_name("polygons_labels", self.fovs)
        lbl = _parse_labels(raster, ox, oy)
        return {name: lbl}, present_ids

    def _labels_from_cell_labels_dir(self) -> tuple[dict[str, Labels2DModel], set[int]]:
        """Stitch pre-computed CellLabels TIFFs."""
        # The per-FOV tile flip is derived from the ``flip_y`` column inside the
        # stitcher (it is the inverse of the transcript-placement flip). We pass
        # only ``self.flip_image`` here as the fallback used when no ``flip_y``
        # column is present (labels read without transcripts). Folding ``flip_y``
        # into ``flip_image`` here would double-apply it and mirror the labels.
        fov_local_to_global = self._build_cell_label_luts()

        # When no cell_info.csv is available, build LUTs from the TIFFs
        # themselves using the same global_cell_id formula as read_tables().
        # Without this, labels keep local cell IDs that collide across FOVs
        # and don't match the global IDs in the expression table.
        if fov_local_to_global is None and self.dataset.cell_labels_dir is not None:
            fov_local_to_global = self._build_cell_label_luts_from_tiffs()

        stitched, present_ids, used_fov_locs = _read_stitched_cell_labels_from_dir(
            cast(Path, self.dataset.cell_labels_dir),  # checked by read_labels()
            self.fov_locs,
            fovs=self.fovs,
            flip_image=self.flip_image,
            n_workers=self.n_workers,
            fov_local_to_global=fov_local_to_global,
            fov_size=self._placement_size,
        )

        if stitched is None:
            return {}, set()

        # Update reader state to match the FOVs we actually stitched
        ox, oy = float(used_fov_locs["xmin"].min()), float(used_fov_locs["ymin"].min())
        self.fov_locs = used_fov_locs
        self._base_origin_x = ox
        self._base_origin_y = oy
        self._origin_x = ox
        self._origin_y = oy

        name = _element_name("cell_labels", self.fovs)
        lbl = _parse_labels(stitched, ox, oy)
        return {name: lbl}, present_ids

    def _labels_from_cell_stats_dir(self) -> tuple[dict[str, Labels2DModel], set[int]]:
        """Legacy label stitching from CellStatsDir."""
        ox, oy = self._ensure_origin()
        cell_stats_dir = cast(Path, self.dataset.cell_stats_dir)  # checked by read_labels()
        stitched, df = stitch_segmentation_label_image(
            path=self.dataset.path,
            fov_position_file=str(self.dataset.fov_positions_file),
            cell_info_file=str(cell_stats_dir / f"{self.dataset.dataset_id}_cell_info.csv"),
            dataset_id=self.dataset.dataset_id,
            flip_image=self.flip_image,
            n_workers=self.n_workers,
            fovs=self.fovs if self.fovs else None,
            fov_size=self._placement_size,
        )
        present_ids = {int(x) for x in df["global_cell_id"].to_numpy() if x != 0}
        lbl = _parse_labels(stitched, ox, oy)
        return {"cell_labels_from_cellstats": lbl}, present_ids

    def _build_cell_label_luts(self) -> dict[int, np.ndarray] | None:
        """Build local→global cell ID LUTs for CellLabels stitching."""
        if self.dataset.cell_stats_dir is None or self.dataset.dataset_id is None:
            return None
        info_path = self.dataset.cell_stats_dir / f"{self.dataset.dataset_id}_cell_info.csv"
        if not info_path.exists():
            return None
        cell_info = pd.read_csv(info_path)
        luts: dict[int, np.ndarray] = {}
        for fov in sorted(cell_info["fov"].unique()):
            local_ids = np.unique(cell_info.loc[cell_info["fov"] == fov, "cellID"].astype(int))
            if local_ids.size == 0:
                continue
            lut = np.zeros(int(local_ids.max()) + 1, dtype=int)
            df_f = cell_info.loc[cell_info["fov"] == fov]
            for _, row in df_f.iterrows():
                lut[int(row["cellID"])] = int(row["global_cell_id"])
            luts[fov] = lut
        return luts

    def _build_cell_label_luts_from_tiffs(self) -> dict[int, np.ndarray] | None:
        """Build local→global cell ID LUTs by scanning CellLabels TIFFs.

        Used when no ``cell_info.csv`` is available. Scans each TIFF for
        its maximum local cell ID, then builds per-FOV look-up tables
        using the same ``global_cell_id`` formula as :meth:`global_cell_id`
        so that label IDs match the table's global_cell_id values.
        """
        cell_labels_dir = self.dataset.cell_labels_dir
        if cell_labels_dir is None:
            return None

        fov_tif_map = find_cell_label_tifs(cell_labels_dir)

        # Pass 1: find the max local cell ID across all relevant FOVs.
        fov_max_ids: dict[int, int] = {}
        for fov, img_path in fov_tif_map.items():
            if self.fovs is not None and fov not in self.fovs:
                continue
            if fov not in self.fov_locs.index:
                continue
            arr = tifffile.imread(img_path)
            fov_max_ids[fov] = int(arr.max())

        if not fov_max_ids:
            return None

        global_max_local = max(fov_max_ids.values())

        # Lock max_cell_id if not yet locked; otherwise use the existing
        # locked value so that the LUTs stay consistent with read_tables().
        if self.max_cell_id is None:
            self.max_cell_id = global_max_local
        # If already locked to a smaller value, warn but keep the locked
        # value — labels with IDs beyond the locked max will map to IDs
        # that have no matching table row (acceptable: those cells simply
        # aren't in the expression matrix).
        if global_max_local > self.max_cell_id:
            logger.warning(
                "CellLabels contain local cell IDs up to %d but "
                "max_cell_id is locked to %d. Some label IDs may not "
                "match table rows.",
                global_max_local,
                self.max_cell_id,
            )

        base = self.max_cell_id + 1

        # Pass 2: build per-FOV LUTs.
        luts: dict[int, np.ndarray] = {}
        for fov, max_local in fov_max_ids.items():
            lut = np.zeros(max_local + 1, dtype=np.int64)
            for local_id in range(1, max_local + 1):
                lut[local_id] = fov * base + local_id
            # cell_ID == 0 stays 0 (background)
            luts[fov] = lut

        return luts

    # -- Read tables (expression + metadata) --

    def read_tables(
        self, regions: list[str], *, modality: str | None = None, controls: Literal["keep", "drop"] = "keep"
    ) -> dict[str, AnnData]:
        """Read expression matrix and metadata into an AnnData table, typed by ``var["feature_types"]``.

        RNA features are typed by name (:data:`COSMX_NAME_RULES`); ``controls="drop"`` removes the controls. The
        Protein modality's targets are ``Protein Expression``, its ``Channel-*`` and IgG columns ``unverified``
        (no vendor source types them); `controls` does not apply to it.
        """
        if self.dataset.exprMat_file is None:
            return {}

        adata, bg_df = _read_expr_mat_polars(
            self.dataset.exprMat_file,
            n_rows=None,
            fovs=self.fovs if self.fovs else None,
        )

        by_slide = cast(pd.DataFrame, adata.obs).attrs.get("instance_id_includes_slide")
        if adata.n_obs == 0:
            label = modality or "expression"
            logger.warning("[tables] No %s data for the requested FOVs — skipping.", label)
            return {}

        if self.dataset.metadata_file is not None:
            meta = _read_metadata_polars(self.dataset.metadata_file, n_rows=None)
            expr_obs = cast(pd.DataFrame, adata.obs)
            extra = expr_obs.columns.difference(meta.columns)  # exprMat-only metadata (e.g. AtoMx `cell`)
            no_meta = ~expr_obs.index.isin(meta.index)
            obs = meta.reindex(expr_obs.index)
            if no_meta.any():
                # a cell of the expression matrix is never dropped: keep it, with its exprMat fov / cell_ID
                logger.warning(
                    "%s: %d of %d exprMat cell(s) have no row in %s; they are kept with empty metadata.",
                    self.dataset.exprMat_file.name,
                    int(no_meta.sum()),
                    len(expr_obs),
                    self.dataset.metadata_file.name,
                )
                for c in expr_obs.columns.intersection(meta.columns):
                    obs[c] = obs[c].where(~no_meta, expr_obs[c]).astype(expr_obs[c].dtype)
            adata.obs = obs.join(expr_obs[extra])

        # Compute global cell IDs
        obs = cast(pd.DataFrame, adata.obs)
        if by_slide:
            # fov * base + cell_ID repeats across slides, so number the cells instead
            obs["global_cell_id"] = np.arange(1, adata.n_obs + 1, dtype=int)
            adata.uns["spatialdata_io"] = {
                "instance_id_includes_slide": True,
                "slide_column": by_slide,
                "reason": "fov_cell ids repeat across the slides of one export; cells are keyed <slide>_<fov>_<cell> "
                "and global_cell_id is the row number",
            }
        elif "fov" in obs.columns and "cell_ID" in obs.columns:
            tmp = obs.reset_index(drop=False)
            tmp["global_cell_id"] = self.global_cell_id(tmp)
            obs["global_cell_id"] = tmp.set_index("fov_cellID").loc[obs.index, "global_cell_id"]
        else:
            obs["global_cell_id"] = np.arange(1, adata.n_obs + 1, dtype=int)

        # Determine region
        region = regions[0] if regions else "cells"
        if (
            region == "cells"
            and self.polygons_as_labels
            and self.dataset.polygons_file is not None
            and not self.keep_polygons_after_rasterize
        ):
            region = _element_name("polygons_labels", self.fovs)

        adata.obs["region_key"] = pd.Series(region, index=adata.obs.index, dtype="category")

        if bg_df is not None:
            # var-aligned (genes x FOVs): as an uns DataFrame its gene columns become zarr keys, and probe names
            # like "Gstp1/2" are not valid keys. Set before sanitize_table so varm follows any var renaming.
            missing = adata.var_names.difference(bg_df.columns)
            if len(missing):
                raise ValueError(f"Background rows lack {len(missing)} gene(s) of the table, e.g. {list(missing[:3])}")
            adata.varm["fov_bg_signal"] = pd.DataFrame(
                bg_df[adata.var_names].to_numpy().T, index=adata.var_names, columns=[f"fov_{f}" for f in bg_df.index]
            )

        if (modality or _modality_from_ancestors(self.dataset.exprMat_file, self.dataset.path)) == "Protein":
            adata.var["feature_types"] = feature_types_by_name(adata.var_names, _COSMX_PROTEIN_UNVERIFIED, PROTEIN)
        else:
            adata.var["feature_types"] = feature_types_by_name(adata.var_names, COSMX_NAME_RULES)
            # by name there are only genes and controls, so the split returns one table; varm follows the subset
            adata = split_by_feature_type(adata, controls=controls)["table"]

        # Zarr-safe, case-insensitively unique obs/var keys (spatialdata core),
        # then categoricals → string (zarr cannot serialize categoricals).
        sanitize_table(adata)
        _pandas_categoricals_to_string(cast(pd.DataFrame, adata.obs))

        table = _parse_cell_table(adata, region=region)
        return {"cell_data": table}


# ---------------------------------------------------------------------------
# Shared read orchestration (single- and multimodal paths)
# ---------------------------------------------------------------------------


def _read_images_and_shapes(
    reader: CosMxDatasetReader,
    *,
    read_images: bool,
    read_proteins: bool,
    read_polygons: bool,
    image_models_kwargs: dict[str, Any],
    imread_kwargs: dict[str, Any],
    channels: list[str] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read morphology/protein images and polygon shapes (gate shared by both paths)."""
    images = (
        reader.read_images(
            read_proteins=read_proteins,
            image_models_kwargs=image_models_kwargs,
            imread_kwargs=imread_kwargs,
            channels=channels,
        )
        if read_images and reader.dataset.morphology_2d_dir is not None
        else {}
    )
    shapes = reader.read_shapes() if read_polygons else {}
    return images, shapes


# ---------------------------------------------------------------------------
# Multi-modal orchestrator
# ---------------------------------------------------------------------------


def _cosmx_multi(
    dataset: CosMxDataset,
    *,
    controls: Literal["keep", "drop"],
    read_images: bool,
    read_labels: bool,
    read_proteins: bool,
    read_transcripts: bool,
    read_polygons: bool,
    read_gexp: bool,
    fovs: list[int | str] | int | str | None,
    channels: list[str] | None,
    n_workers: int,
    flip_image: bool | None,
    image_normalization_percentile: float | None,
    image_models_kwargs: dict[str, Any] | None,
    imread_kwargs: dict[str, Any] | None,
    polygons_as_labels: bool,
    keep_polygons_after_rasterize: bool,
    align_rasters_to_polygons: bool | None,
    add_fovs_as_shapes: bool,
    skip_empty_fovs: bool,
    preview_fovs: bool,
    pixel_size_um: float | None,
    fov_size_px: tuple[int, int] | None = None,
) -> SpatialData | None:
    """Read a multimodal CosMx dataset (e.g. RNA + Protein)."""
    modalities = dataset.modalities
    if not modalities:
        raise ValueError("Expected modalities for multi-modal dataset.")

    fov_set = _normalize_fovs(fovs)
    image_models_kwargs, imread_kwargs = _default_image_kwargs(image_models_kwargs, imread_kwargs)

    # Pick modality for shared spatial elements (prefer one with polygons/labels)
    label_ds = next(
        (ds for ds in modalities.values() if ds.polygons_file is not None or ds.cell_labels_dir is not None),
        next(iter(modalities.values())),
    )
    if label_ds.fov_positions_file is None:
        raise ValueError("FOV positions file required for multimodal runs.")

    fov_size_info: dict[str, Any] = {"fov_size_px": None, "fov_size_source": None}
    fov_size: tuple[float, float] | None = None
    fov_size_exact = True
    if read_images or read_labels or read_polygons or read_transcripts or add_fovs_as_shapes or preview_fovs:
        fov_size_info = _fov_size_px(dataset, fov_size_px)
        fov_size = cast("tuple[float, float]", tuple(fov_size_info["fov_size_px"]))
        fov_size_exact = fov_size_info["fov_size_source"] != FOV_SIZE_LOWER_BOUND
    fov_locs = _read_fov_locs(
        label_ds.fov_positions_file,
        fovs=sorted(fov_set) if fov_set else None,
        fov_size=fov_size if fov_size_exact else None,
    )
    if fov_size is not None:
        _check_fov_size_use(
            fov_size_info,
            [dataset, *modalities.values()],
            fov_locs,
            read_labels=read_labels,
            read_polygons=read_polygons,
            read_transcripts=read_transcripts,
            polygons_as_labels=polygons_as_labels,
        )

    if skip_empty_fovs and (read_images or read_labels):
        fov_locs = _prune_empty_fovs(dataset, fov_locs, fov_set)

    if preview_fovs:
        _plot_fov_preview(fov_locs, fov_set, fov_size=cast("tuple[float, float]", fov_size))
        return None

    n_fovs = len(fov_set) if fov_set else len(fov_locs)
    n_mods = len(modalities)
    logger.info(
        "Reading multimodal CosMx dataset (%d modalities, %d FOV(s)). "
        "Source files are gzip-compressed CSVs that must be fully decompressed "
        "even for a FOV subset — this may take a while on first load.",
        n_mods,
        n_fovs,
    )

    # Shared reader for spatial elements
    logger.debug("[1/5] Setting up reader and loading polygons...")
    reader = CosMxDatasetReader(
        label_ds,
        fovs=fov_set,
        n_workers=n_workers,
        flip_image=flip_image,
        polygons_as_labels=polygons_as_labels,
        fov_locs=fov_locs,
        align_rasters_to_polygons=align_rasters_to_polygons,
        keep_polygons_after_rasterize=keep_polygons_after_rasterize,
        image_normalization_percentile=image_normalization_percentile,
        fov_size=fov_size,
        fov_size_exact=fov_size_exact,
    )

    # Pre-load polygons to freeze max_cell_id
    base_max_cell_id: int | None = None
    try:
        reader._get_polygons()
        base_max_cell_id = reader.max_cell_id
    except Exception as e:  # noqa: BLE001 - best-effort probe; failure handled below
        logger.debug("Could not pre-load polygons to freeze max_cell_id: %s", e)

    def _modality_reader(ds: CosMxDataset) -> CosMxDatasetReader:
        """Secondary reader (transcripts/tables) sharing the frozen max_cell_id."""
        r = CosMxDatasetReader(
            ds,
            fovs=fov_set,
            n_workers=n_workers,
            flip_image=flip_image,
            polygons_as_labels=False,
            fov_locs=fov_locs,
            align_rasters_to_polygons=align_rasters_to_polygons,
            fov_size=fov_size,
            fov_size_exact=fov_size_exact,
        )
        if base_max_cell_id is not None:
            r.max_cell_id = base_max_cell_id
        return r

    logger.debug("[2/5] Reading images and labels...")
    images, shapes = _read_images_and_shapes(
        reader,
        read_images=read_images,
        read_proteins=read_proteins,
        read_polygons=read_polygons,
        image_models_kwargs=image_models_kwargs,
        imread_kwargs=imread_kwargs,
        channels=channels,
    )
    labels, label_ids = reader.read_labels() if read_labels else ({}, set())

    if add_fovs_as_shapes:
        shapes = {**shapes, **reader.build_fov_shapes()}

    # Transcripts (from whichever modality has tx_file)
    logger.debug("[3/5] Reading transcripts...")
    points: dict[str, PointsModel] = {}
    tx_mod = next((name for name, ds in modalities.items() if ds.tx_file is not None), None)
    if read_transcripts and tx_mod is not None:
        tx_points = _modality_reader(modalities[tx_mod]).read_transcripts()
        if tx_points:
            points = {"transcripts": next(iter(tx_points.values()))}

    # Label-anchored co-registration (same as cosmx()): the protein-derived
    # segmentation read above is shared by every modality. Each modality's cells not
    # in the rasterised labels are counted and kept, never dropped.
    tables: dict[str, AnnData] = {}
    region_refs = list(labels.keys()) or list(shapes.keys()) or ["cells"]
    if read_gexp:
        for mod_name, mod_ds in modalities.items():
            logger.debug("[4/5] Reading %s data...", mod_name)
            mod_tables = _modality_reader(mod_ds).read_tables(region_refs, modality=mod_name, controls=controls)
            if label_ids:
                _count_cells_outside_labels(mod_tables, label_ids, source_label="rasterised labels")
            if mod_tables:
                tables[mod_name] = next(iter(mod_tables.values()))

    logger.debug("[5/5] Assembling SpatialData object...")
    return _assemble_sdata(images, points, labels, shapes, tables, label_ds, pixel_size_um, fov_size_info)


# ---------------------------------------------------------------------------
# SpatialData assembly
# ---------------------------------------------------------------------------


# the spatialdata-db Lead's ruling (T141), not a vendor value: with no FOV image, a current-camera run has its
# largest cell-centre local px coordinate in this range on each axis
_CURRENT_CAMERA_LOCAL_EXTENT_PX = (4000.0, COSMX_FOV_SIZE_PX)


def _header_hw(path: Path) -> tuple[int, int]:
    """(height, width) of an image from its header only.

    A TIFF page's ``imagelength`` / ``imagewidth`` are the ImageLength / ImageWidth tags (tifffile 2026.11.8,
    ``tifffile.py:6290-6294``), so a planar ``(S, H, W)`` page still gives ``(H, W)``; other formats are read by PIL
    (a scikit-image dependency), which also reads only the header.
    """
    if path.suffix.lower() in {".tif", ".tiff"}:
        with tifffile.TiffFile(path) as tif:
            page = cast("tifffile.TiffPage", tif.pages[0])
            return int(page.imagelength), int(page.imagewidth)
    from PIL import Image

    with Image.open(path) as im:
        return int(im.size[1]), int(im.size[0])


_IMAGE_SUFFIXES = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}


def _fov_image_files(dataset: CosMxDataset) -> tuple[str, list[Path]] | None:
    """The first FOV image directory with files, and its image files, sorted.

    Looked for in ``Morphology2D``, ``RawMorphologyImages``, ``CellLabels`` and ``CellComposite``, in that order.
    """
    for name, directory in (
        ("Morphology2D", dataset.morphology_2d_dir),
        ("RawMorphologyImages", dataset.raw_morphology_dir),
        ("CellLabels", dataset.cell_labels_dir),
        ("CellComposite", dataset.cell_composite_dir),
    ):
        if directory is None:
            continue
        if name == "CellLabels":
            files = sorted(find_cell_label_tifs(directory).values())
        else:
            suffixes = _IMAGE_SUFFIXES if name == "CellComposite" else {".tif", ".tiff"}
            files = sorted(p for p in directory.rglob("*") if p.suffix.lower() in suffixes)
        if files:
            return name, files
    return None


def _fov_tiff_header(dataset: CosMxDataset) -> tuple[float | None, tuple[int, int] | None, str | None]:
    """``ImPixelSize_nm``, (height, width) and name of the first FOV TIFF, read from its header only.

    A morphology TIFF (``Morphology2D``, else ``RawMorphologyImages``) gives all three; without one, a
    ``CellLabels`` TIFF gives the FOV shape.
    """
    found = _fov_image_files(dataset)
    if found is None or found[0] == "CellComposite":
        return None, None, None
    name, files = found
    nm = None
    if name != "CellLabels":
        with tifffile.TiffFile(files[0]) as tif:
            try:
                nm = json.loads(cast("tifffile.TiffPage", tif.pages[0]).description).get("ImPixelSize_nm")
            except (json.JSONDecodeError, AttributeError):
                nm = None
    return (float(nm) if isinstance(nm, int | float) else None), _header_hw(files[0]), files[0].name


def _fov_image_shape(datasets: list[CosMxDataset]) -> tuple[tuple[int, int], str] | None:
    """``(width, height)`` of the run's FOV images and where they were read, from every FOV header.

    Takes the first directory with images in the order of :func:`_fov_image_files` and reads the header of every
    file in it. FOVs of different shapes raise ``ValueError``: every FOV is placed at one size, so a larger tile
    would be cropped.
    """
    for ds in datasets:
        if (found := _fov_image_files(ds)) is None:
            continue
        name, files = found
        shapes: dict[tuple[int, int], list[str]] = {}
        for f in files:
            shapes.setdefault(_header_hw(f), []).append(f.name)
        if len(shapes) > 1:
            listed = "; ".join(f"{w} x {h} px ({len(n)} file(s), e.g. {n[0]})" for (h, w), n in sorted(shapes.items()))
            raise ValueError(
                f"CosMx FOV images in {name} differ in shape: {listed}. Every FOV is placed at one size, so the larger "
                "tiles would be cropped."
            )
        ((h, w),) = shapes
        return (w, h), f"{name} ({len(files)} file(s), e.g. {files[0].name})"
    return None


# local px columns per file kind: polygon vertices, cell centres, transcripts
_LOCAL_PX_COLUMNS = {
    "polygons_file": ("x_local_px", "y_local_px"),
    "metadata_file": ("CenterX_local_px", "CenterY_local_px"),
    "tx_file": ("x_local_px", "y_local_px"),
}


def _local_px_max(datasets: list[CosMxDataset]) -> tuple[tuple[float, float], list[str]] | None:
    """Largest local px per axis over the run's polygon vertices and cell centres, else its transcripts.

    Transcripts are scanned only when no polygon or metadata file has local px columns (they are the largest file).
    """
    import polars as pl

    for kinds in (("polygons_file", "metadata_file"), ("tx_file",)):
        xs, ys, used = [], [], []
        for ds in datasets:
            for kind in kinds:
                path = getattr(ds, kind)
                if path is None or not path.exists():
                    continue
                lf = pl.scan_csv(path, infer_schema_length=10_000)
                cols = {c.lower(): c for c in lf.collect_schema().names()}
                xc, yc = (cols.get(c.lower()) for c in _LOCAL_PX_COLUMNS[kind])
                if xc is None or yc is None:
                    continue
                x, y = lf.select(pl.col(xc).cast(pl.Float64).max(), pl.col(yc).cast(pl.Float64).max()).collect().row(0)
                if x is None or y is None:
                    continue
                xs.append(x)
                ys.append(y)
                used.append(f"{path.name} {xc}/{yc}")
        if xs:
            return (max(xs), max(ys)), used
    return None


# a FOV size from the local px maxima: only a lower bound, used for the FOV boxes and nothing else
FOV_SIZE_LOWER_BOUND = "local_px_lower_bound"
_TABLES_ONLY = (
    "read_images=False, read_labels=False, read_polygons=False, read_transcripts=False and add_fovs_as_shapes=False "
    "read the tables only"
)


def _fov_size_error(detail: str) -> ValueError:
    return ValueError(f"CosMx FOV size unknown: {detail}. Pass fov_size_px=(width, height) to set it; {_TABLES_ONLY}.")


def _fov_size_px(dataset: CosMxDataset, fov_size_px: tuple[int, int] | None = None) -> dict[str, Any]:
    """The run's FOV ``(width, height)`` in px and its source, resolved once per run, in the order of :func:`cosmx`."""
    if fov_size_px is not None:
        w, h = (int(v) for v in fov_size_px)
        if w <= 0 or h <= 0:
            raise ValueError(f"fov_size_px must be two positive ints (width, height), got {fov_size_px!r}.")
        return {"fov_size_px": [w, h], "fov_size_source": "argument"}
    datasets = [dataset, *(dataset.modalities.values() if dataset.modalities else [])]
    if (image := _fov_image_shape(datasets)) is not None:
        (w, h), where = image
        logger.info("CosMx FOV size %d x %d px from the FOV image headers in %s.", w, h, where)
        return {"fov_size_px": [int(w), int(h)], "fov_size_source": "fov_image"}
    if (extent := _local_px_max(datasets)) is not None:
        (x, y), used = extent
        lo, hi = _CURRENT_CAMERA_LOCAL_EXTENT_PX
        if lo <= x <= hi and lo <= y <= hi:
            # spatialdata-db-schemas 9a37d86 specs/nanostring_cosmx.yaml:12-15 (fov_px_current, 4256 x 4256), applied
            # by the Lead's T141 rule "NEW 4000-4256" (spatialdata-io-sddb tasks/todo.md:586, "T141 part 1")
            size = int(COSMX_FOV_SIZE_PX)
            logger.info(
                "CosMx FOV size %d x %d px: largest local px %g x %g (%s) is in the current-camera range.",
                size,
                size,
                x,
                y,
                "; ".join(used),
            )
            return {"fov_size_px": [size, size], "fov_size_source": "spec_current_camera_extent"}
        w, h = math.ceil(x), math.ceil(y)
        logger.warning(
            "CosMx FOV size only bounded below: largest local px %g x %g (%s), outside the current-camera range "
            "%g-%g. It sizes the FOV boxes only; pass fov_size_px=(width, height) for anything else.",
            x,
            y,
            "; ".join(used),
            lo,
            hi,
        )
        return {"fov_size_px": [int(w), int(h)], "fov_size_source": FOV_SIZE_LOWER_BOUND}
    raise _fov_size_error(
        "no FOV image (Morphology2D, RawMorphologyImages, CellLabels_F*.tif or CellComposite) and no local px columns "
        "(x_local_px / y_local_px in the polygons or transcripts file, CenterX_local_px / CenterY_local_px in the "
        "metadata file)"
    )


def _check_fov_size_use(
    fov_size_info: dict[str, Any],
    datasets: list[CosMxDataset],
    fov_locs: pd.DataFrame | None,
    *,
    read_labels: bool,
    read_polygons: bool,
    read_transcripts: bool,
    polygons_as_labels: bool,
) -> None:
    """Raise when a read needs an exact FOV size (a canvas or a y-flip height) and only a lower bound is known."""
    if fov_size_info["fov_size_source"] != FOV_SIZE_LOWER_BOUND:
        return
    has_poly = any(ds.polygons_file is not None for ds in datasets)
    has_tx = any(ds.tx_file is not None for ds in datasets)
    non_flipped = fov_locs is not None and "flip_y" in fov_locs and bool((~fov_locs["flip_y"].astype(bool)).any())
    needs = []
    if read_labels and polygons_as_labels and has_poly:
        needs.append("the polygon label canvas")
    if non_flipped and has_poly and (read_polygons or (read_labels and polygons_as_labels)):
        needs.append("the polygon y flip of px+mm FOVs")
    if non_flipped and has_tx and read_transcripts:
        needs.append("the transcript y flip of px+mm FOVs")
    if needs:
        w, h = fov_size_info["fov_size_px"]
        raise _fov_size_error(
            f"no FOV image, and the largest local px ({w} x {h}) is only a lower bound, which {' and '.join(needs)} "
            "cannot use"
        )


def _local_px_extent(tables: dict[str, Any]) -> tuple[float, float] | None:
    """Largest ``CenterX_local_px`` and ``CenterY_local_px`` over the tables' cells, else ``None``."""
    xs, ys = [], []
    for table in tables.values():
        obs = (table.table if hasattr(table, "table") else table).obs
        if {"CenterX_local_px", "CenterY_local_px"}.issubset(obs.columns):
            xs.append(pd.to_numeric(obs["CenterX_local_px"], errors="coerce").max())
            ys.append(pd.to_numeric(obs["CenterY_local_px"], errors="coerce").max())
    if not xs or np.isnan(np.nanmax(xs)) or np.isnan(np.nanmax(ys)):
        return None
    return float(np.nanmax(xs)), float(np.nanmax(ys))


def _pixel_size_um(dataset: CosMxDataset, tables: dict[str, Any], pixel_size_um: float | None = None) -> dict[str, Any]:
    """The pixel size in µm and its source, in the order of the :func:`cosmx` docstring."""
    if pixel_size_um is not None:
        return {"pixel_size_um": float(pixel_size_um), "pixel_size_source": "argument"}
    fov_positions_file = dataset.fov_positions_file
    measured = _pixel_size_from_fov_positions(fov_positions_file) if fov_positions_file is not None else None
    if measured is not None:
        if abs(measured / COSMX_PIXEL_SIZE - 1) > 0.01:
            logger.warning(
                f"{fov_positions_file.name} gives {measured:.6g} µm per pixel, more than 1% off the CosMx spec "  # type: ignore[union-attr]
                f"{COSMX_PIXEL_SIZE}; using the file's value for the 'microns' coordinate system."
            )
        return {"pixel_size_um": measured, "pixel_size_source": "fov_positions"}
    nm, shape, tif_name = _fov_tiff_header(dataset)
    if nm is not None and nm > 0:
        return {"pixel_size_um": nm / 1000.0, "pixel_size_source": "tiff"}
    lo, hi = _CURRENT_CAMERA_LOCAL_EXTENT_PX
    if shape is not None:
        current = shape == (int(COSMX_FOV_SIZE_PX), int(COSMX_FOV_SIZE_PX))
        geometry = f"FOV image {shape[1]} x {shape[0]} px ({tif_name})"
    elif (extent := _local_px_extent(tables)) is not None:
        current = all(lo <= e <= hi for e in extent)
        geometry = f"largest cell-centre local px extent {extent[0]:g} x {extent[1]:g}"
    else:
        current, geometry = False, "no FOV image and no CenterX/Y_local_px columns"
    if current:
        return {"pixel_size_um": COSMX_PIXEL_SIZE, "pixel_size_source": "spec"}
    logger.warning(
        f"CosMx pixel size unknown: no px+mm FOV positions, no TIFF ImPixelSize_nm > 0, and the geometry ({geometry}) "
        f"is not the {int(COSMX_FOV_SIZE_PX)} x {int(COSMX_FOV_SIZE_PX)} px FOV that the spec value {COSMX_PIXEL_SIZE} "
        "applies to. Pass pixel_size_um= to set it; no 'microns' coordinate system is added."
    )
    return {"pixel_size_um": None, "pixel_size_source": "unknown"}


def _add_microns(sdata: SpatialData, pixel_size_um: float) -> None:
    """Add the ``"microns"`` coordinate system: each element's ``"global"`` (pixel) transformation, then a scale."""
    scale = Scale([pixel_size_um, pixel_size_um], axes=("x", "y"))
    for _, _, element in sdata.gen_spatial_elements():
        to_global = get_transformation(element, to_coordinate_system="global")
        set_transformation(element, Sequence([to_global, scale]), to_coordinate_system="microns")  # type: ignore[list-item]


def _assemble_sdata(
    images: dict[str, Any],
    points: dict[str, Any],
    labels: dict[str, Any],
    shapes: dict[str, Any],
    tables: dict[str, Any],
    dataset: CosMxDataset,
    pixel_size_um: float | None = None,
    fov_size_info: dict[str, Any] | None = None,
) -> SpatialData:
    """Build and annotate the final SpatialData object."""
    # Move each image's per-channel normalization divisors from the (non-persisted) element
    # attrs into the persisted SpatialData attrs, so the reversible stretch survives a
    # write/read round-trip. Popped from the element so the metadata has a single home.
    attrs: dict[str, Any] = {}
    image_norm = {
        nm: im.attrs.pop("cosmx_image_normalization")
        for nm, im in images.items()
        if "cosmx_image_normalization" in getattr(im, "attrs", {})
    }
    if image_norm:
        attrs["cosmx_image_normalization"] = image_norm

    sdata = SpatialData(
        images=images,
        points=points,
        labels=labels,
        shapes=shapes,
        tables=tables,
        attrs=attrs,
    )
    sdata.attrs["cosmx"] = {
        **_pixel_size_um(dataset, tables, pixel_size_um),
        **(fov_size_info or {"fov_size_px": None, "fov_size_source": None}),
    }
    if sdata.attrs["cosmx"]["pixel_size_um"] is not None:
        _add_microns(sdata, sdata.attrs["cosmx"]["pixel_size_um"])

    # Wire tables to label elements
    if labels and sdata.tables:
        main_label = next(iter(labels))
        for table_name, table in list(sdata.tables.items()):
            raw = table.table if hasattr(table, "table") else table
            if "region_key" not in raw.obs:
                continue
            reg_vals = pd.Series(raw.obs["region_key"]).astype("string").unique()
            if len(reg_vals) == 1 and reg_vals[0] == main_label:
                sdata.set_table_annotates_spatialelement(table_name, region=main_label)
                continue
            raw.uns.pop("spatialdata_attrs", None)
            raw.obs["region_key"] = pd.Series(main_label, index=raw.obs.index, dtype="category")
            sdata.tables[table_name] = _parse_cell_table(raw, region=main_label)
            sdata.set_table_annotates_spatialelement(table_name, region=main_label)

    for table in sdata.tables.values():
        _set_observation_unit(table, "cell", None)
    return _set_reader_metadata(sdata, "cosmx")


# ---------------------------------------------------------------------------
# FOV normalization
# ---------------------------------------------------------------------------


def _normalize_fovs(fovs: list[int | str] | int | str | None) -> set[int] | None:
    """Convert the user-facing ``fovs`` argument to an optional set of ints."""
    if fovs is None:
        return None
    if isinstance(fovs, int | str):
        return {int(fovs)}
    return {int(f) for f in fovs}


def _present_fovs(dataset: CosMxDataset) -> set[int]:
    """FOV ids with per-FOV data on disk, unioned over the dataset + modalities (#37)."""
    sources = [dataset, *(dataset.modalities.values() if dataset.modalities else [])]
    present: set[int] = set()
    for s in sources:
        present |= detect_fovs_with_data(
            morphology_2d_dir=s.morphology_2d_dir,
            cell_labels_dir=s.cell_labels_dir,
            cell_stats_dir=s.cell_stats_dir,
            analysis_results_dir=s.analysis_results_dir,
        )
    return present


@overload
def _prune_empty_fovs(dataset: CosMxDataset, fov_locs: pd.DataFrame, fov_set: set[int] | None) -> pd.DataFrame: ...
@overload
def _prune_empty_fovs(dataset: CosMxDataset, fov_locs: None, fov_set: set[int] | None) -> None: ...
def _prune_empty_fovs(
    dataset: CosMxDataset,
    fov_locs: pd.DataFrame | None,
    fov_set: set[int] | None,
) -> pd.DataFrame | None:
    """Drop phantom FOVs (listed in positions, no data files) from *fov_locs* (#37).

    Pruned before any element is read so origin, canvases and FOV boxes stay
    consistent. Never drops an explicitly requested FOV; keeps every FOV when
    detection finds nothing (transcript-/gexp-only datasets).
    """
    if fov_locs is None or fov_locs.empty:
        return fov_locs
    present = _present_fovs(dataset)
    if not present:
        return fov_locs
    listed = {int(f) for f in fov_locs.index}
    phantom = listed - present
    if fov_set is not None:
        phantom -= fov_set
    if not phantom:
        return fov_locs
    logger.info("skip_empty_fovs: dropping %d FOV(s) with no data files: %s", len(phantom), sorted(phantom))
    return fov_locs.loc[sorted(listed - phantom)]


def _prescan_max_cell_id(
    reader: CosMxDatasetReader,
    dataset: CosMxDataset,
    fov_set: set[int] | None,
    read_transcripts: bool = False,
) -> None:
    """Pre-scan cell_ID column from all available files to lock max_cell_id.

    This removes the load-order dependency: without pre-scanning,
    whichever element calls ``global_cell_id()`` first locks the max,
    and subsequent elements with a higher max will crash. The transcripts
    file is included when it will be read: it can assign transcripts to
    cells beyond any polygon or expression row (GSE304160).
    """
    import polars as pl

    max_ids: list[int] = []
    tx = [dataset.tx_file] if read_transcripts else []
    for path in [dataset.polygons_file, dataset.exprMat_file, dataset.metadata_file, *tx]:
        if path is None or not path.exists():
            continue
        try:
            lf = pl.scan_csv(path, n_rows=None)
            cols = lf.collect_schema().names()
        except Exception as e:  # noqa: BLE001 - best-effort probe; failure handled below
            # A file that exists but cannot be scanned is a real problem for a pre-scan
            # whose whole job is to establish the max_cell_id invariant — surface it
            # rather than silently dropping the file (which would mis-lock the max).
            logger.warning("Could not pre-scan %s for max cell_ID: %s", path, e)
            continue
        # Match the id column through the shared canonical alias set, NOT a hardcoded
        # ("cell_ID", "cellID") pair: expression/metadata use ``cell_ID`` while polygon
        # CSVs use ``cellID``, and other exports use ``cell_id``/``object_id``. Including
        # the polygon column is essential — orphan (segmented-only) cells are often the
        # highest per-FOV IDs and would otherwise be missed, mis-locking max_cell_id.
        id_col = _match_canonical(cols, "cell_ID")
        if id_col is None:
            logger.warning("No cell_ID-like column found in %s (columns: %s) — skipping for max cell_ID.", path, cols)
            continue
        fov_col = _match_canonical(cols, "fov")
        if fov_set is not None and fov_col is not None:
            lf = lf.filter(pl.col(fov_col).is_in(list(fov_set)))
        try:
            val = lf.select(pl.col(id_col).max()).collect().item()
        except Exception as e:  # noqa: BLE001 - best-effort probe; failure handled below
            logger.warning("Could not read max %s from %s: %s", id_col, path, e)
            continue
        if val is None:
            continue
        try:
            max_ids.append(int(val))
        except (TypeError, ValueError):
            logger.warning("Non-integer %s in %s (max %r) — skipping for max cell_ID.", id_col, path, val)

    if max_ids:
        reader.max_cell_id = max(max_ids)
        logger.debug("Pre-scanned max_cell_id=%d from %d file(s).", reader.max_cell_id, len(max_ids))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@inject_docs(cx=CosmxKeys)
def cosmx(
    path: str | Path,
    dataset_id: str | None = None,
    *,
    read_images: bool = True,
    read_labels: bool = True,
    read_proteins: bool = True,
    read_transcripts: bool = True,
    read_polygons: bool = True,
    read_gexp: bool = True,
    fovs: list[int | str] | int | str | None = None,
    channels: list[str] | None = None,
    n_workers: int | None = None,
    flip_image: bool | None = None,
    image_normalization_percentile: float | None = None,
    image_models_kwargs: dict[str, Any] | None = None,
    imread_kwargs: dict[str, Any] | None = None,
    polygons_as_labels: bool = True,
    keep_polygons_after_rasterize: bool = False,
    align_rasters_to_polygons: bool | None = None,
    add_fovs_as_shapes: bool = True,
    skip_empty_fovs: bool = True,
    preview_fovs: bool = False,
    controls: Literal["keep", "drop"] = "keep",
    pixel_size_um: float | None = None,
    fov_size_px: tuple[int, int] | None = None,
) -> SpatialData | None:
    """Read *CosMx Nanostring* data into a :class:`spatialdata.SpatialData` object.

    Supports all known CosMx export formats: flat CSV files, nested
    CellStatsDir layouts, multimodal RNA+Protein runs, and both old-style
    (px-only / mm-only) and new-style (px+mm) FOV positions files.

    Files are recognized by their standard CosMx suffixes — counts
    ``{cx.COUNTS_SUFFIX!r}``, metadata ``{cx.METADATA_SUFFIX!r}``, transcripts
    ``{cx.TRANSCRIPTS_SUFFIX!r}``, and FOV positions ``{cx.FOV_SUFFIX!r}`` —
    typically prefixed with the dataset id.

    Parameters
    ----------
    path
        Path to the root directory containing CosMx files.
    dataset_id
        Optional dataset identifier.  Inferred from file prefixes if not given.
    read_images
        Whether to read morphology images.
    read_labels
        Whether to read or rasterize cell labels.
    read_proteins
        Whether to include protein image channels.
    read_transcripts
        Whether to read transcript coordinates.
    read_polygons
        Whether to read cell boundary polygons as shapes.
    read_gexp
        Whether to read the gene expression matrix.
    fovs
        Specific FOV(s) to read.  ``None`` reads all FOVs.
    channels
        Specific image channel names to include.
    n_workers
        Number of parallel workers for stitching operations.
    flip_image
        Flip morphology images vertically to co-register with transcripts and
        labels.  Defaults to ``True`` when ``None`` (CosMx morphology TIFFs are
        stored y-inverted relative to the FOV-grid placement); pass ``False`` for
        a dataset stored the other way.
    image_normalization_percentile
        Optional per-channel contrast normalization for morphology/protein images, applied
        post-stitching on top of the dtype-max scaling.  ``None`` (default) keeps the plain
        dtype-max ``[0, 1]`` image.  A float in ``[0, 100]`` (e.g. ``99.9``) divides each
        channel by that percentile of its non-zero pixels, recovering channels whose real
        signal sits far below the dtype ceiling and would otherwise render near-black
        (issue #38).  The stretch is scale-only (no clipping), so the brightest pixels may
        exceed ``1.0``; it is reversible via the divisors recorded in
        ``sdata.attrs['cosmx_image_normalization']`` — these are in dtype-max ``[0, 1]`` units
        (they reverse to the dtype-max image, not raw counts).  The percentile is approximate
        for large multi-chunk images, so the exact divisor may vary slightly with chunking.
    image_models_kwargs
        Extra kwargs for :class:`spatialdata.models.Image2DModel`.
    imread_kwargs
        Extra kwargs for :func:`dask_image.imread.imread`.
    polygons_as_labels
        Rasterize polygons into a label image.
    keep_polygons_after_rasterize
        Keep vector polygons even when ``polygons_as_labels=True``.
    align_rasters_to_polygons
        Anchor images and labels to the polygon coordinate origin.
    add_fovs_as_shapes
        Add FOV bounding boxes as shape elements.
    skip_empty_fovs
        Drop FOVs that appear in the positions file but have no data files on
        disk (issue #37).  CosMx positions files often list more FOVs than ship
        data; those phantom FOVs otherwise inflate the image canvas, desync the
        image and label canvases, and add empty FOV boxes.  Defaults to ``True``;
        FOVs are pruned only when detection finds a strict, non-empty subset with
        data (transcript-/gexp-only datasets are left untouched).  Pass ``False``
        to keep every listed FOV.
    preview_fovs
        Show a preview plot of FOV positions and return ``None``.
    controls
        ``"keep"`` keeps the RNA control features in the table, ``"drop"`` removes them. CosMx names no feature
        type, so the features are typed by name in ``var["feature_types"]``: ``NegPrb*`` and ``Negative*`` (also
        ``NegativeAdd``) are ``Negative Control Probe``, ``SystemControl*`` and ``FalseCode*`` ``System Control``,
        the rest ``Gene Expression``. A Protein modality's table is never affected; its targets are
        ``Protein Expression`` and its ``Channel-*`` and IgG isotype columns ``unverified``.
    pixel_size_um
        µm per pixel stated by the caller, e.g. from the deposit's description. Overrides every other source; see the
        pixel size order below.
    fov_size_px
        The FOV ``(width, height)`` in px stated by the caller. Overrides every other source; see the FOV size order
        below.

    Returns
    -------
    :class:`spatialdata.SpatialData` or ``None`` (if ``preview_fovs=True``).

    Tables: a single-modality run gives one table, ``cell_data``; a multimodal run gives one table per
    modality, ``RNA`` and ``Protein``, annotating the same labels.

    Every element is in pixels in ``"global"``, and in µm in ``"microns"`` (the same transformation followed by a
    scale) when the pixel size is known. ``sdata.attrs["cosmx"]`` records it as ``pixel_size_um`` with
    ``pixel_size_source``, resolved in this order:

    1. ``"argument"``: ``pixel_size_um``, when given.
    2. ``"fov_positions"``: the FOV positions file, when it lists each FOV in both ``x_global_px`` and
       ``x_global_mm`` (a warning is logged if it is more than 1% off the spec).
    3. ``"tiff"``: ``ImPixelSize_nm`` in the ImageDescription JSON of the first morphology TIFF
       (``Morphology2D``, else ``RawMorphologyImages``), when it is > 0.
    4. ``"spec"``: the CosMx value 0.120280945 µm, only when the FOV geometry is the 4256 x 4256 px FOV it applies
       to: the first FOV image (morphology, else ``CellLabels``) is 4256 x 4256 px or, with no FOV image, the
       largest ``CenterX_local_px`` and ``CenterY_local_px`` are both within 4000-4256.
    5. ``"unknown"``: otherwise ``pixel_size_um`` is ``None``, a warning names the geometry found, and there is no
       ``"microns"`` coordinate system.

    Every FOV is placed at the run's FOV size, recorded as ``fov_size_px`` ``[width, height]`` with
    ``fov_size_source`` in ``sdata.attrs["cosmx"]`` (both ``None`` when nothing is placed), resolved in this order:

    1. ``"argument"``: ``fov_size_px``, when given.
    2. ``"fov_image"``: the headers of every FOV image in the first of ``Morphology2D``, ``RawMorphologyImages``,
       ``CellLabels`` and ``CellComposite`` that has any; FOVs of different shapes raise ``ValueError``.
    3. ``"spec_current_camera_extent"``: 4256 x 4256 px, when there is no FOV image and the largest local px
       (polygon vertices and cell centres, else transcripts) are within 4000-4256 on both axes.
    4. ``"local_px_lower_bound"``: otherwise those maxima, rounded up. They are only a lower bound and size the FOV
       boxes only; a read that needs a canvas (polygon labels) or a y-flip height (px+mm FOVs) raises
       ``ValueError``.

    With none of these, any read that places FOVs raises ``ValueError``. A table cell missing from the labels is
    kept and logged, never dropped.
    """
    n_workers = n_workers or 1
    path = Path(path).resolve()

    dataset = _set_up_cosmx_dataset_for_conversion(path=path, dataset_id=dataset_id)

    # Morphology TIFFs are stored y-inverted relative to the FOV-grid placement,
    # so they need a vertical flip to co-register with transcripts/labels.  When
    # the user does not specify, default to flipping (issue #42).  This replaces
    # an earlier heuristic that inferred the flip from the transcript coordinate
    # convention (``not flip_y``) — the wrong signal, which left px-only images
    # mirrored when read without polygons.  Resolve here, BEFORE the multimodal
    # dispatch, so both single- and multi-modal paths get the same default.
    if flip_image is None:
        flip_image = True
    flip_image = bool(flip_image)

    # --- Multimodal dispatch ---
    if dataset.modalities:
        return _cosmx_multi(
            dataset=dataset,
            controls=controls,
            read_images=read_images,
            read_labels=read_labels,
            read_proteins=read_proteins,
            read_transcripts=read_transcripts,
            read_polygons=read_polygons,
            read_gexp=read_gexp,
            fovs=fovs,
            channels=channels,
            n_workers=n_workers,
            flip_image=flip_image,
            image_normalization_percentile=image_normalization_percentile,
            image_models_kwargs=image_models_kwargs,
            imread_kwargs=imread_kwargs,
            polygons_as_labels=polygons_as_labels,
            keep_polygons_after_rasterize=keep_polygons_after_rasterize,
            align_rasters_to_polygons=align_rasters_to_polygons,
            add_fovs_as_shapes=add_fovs_as_shapes,
            skip_empty_fovs=skip_empty_fovs,
            preview_fovs=preview_fovs,
            pixel_size_um=pixel_size_um,
            fov_size_px=fov_size_px,
        )

    # --- Single-modality path ---
    logger.info(
        "Reading single-modality CosMx dataset (ID: %s). "
        "Gzip-compressed CSV sources must be fully decompressed even for a "
        "FOV subset — this may take a while on first load.",
        dataset.dataset_id,
    )

    fov_set = _normalize_fovs(fovs)
    image_models_kwargs, imread_kwargs = _default_image_kwargs(image_models_kwargs, imread_kwargs)
    fov_size_info: dict[str, Any] = {"fov_size_px": None, "fov_size_source": None}
    fov_size: tuple[float, float] | None = None
    fov_size_exact = True

    try:
        fov_locs = _read_fov_locs(dataset.fov_positions_file) if dataset.fov_positions_file else None
    except NoFovCoordinatesError as e:
        # Nothing per-FOV (images, labels, polygons, transcripts, FOV boxes) can be placed without FOV origins.
        logger.warning("%s Reading the cell tables only; cell positions stay in the metadata columns.", e)
        fov_locs = pd.DataFrame({"xmin": [np.nan], "ymin": [np.nan]})
        read_images = read_labels = read_polygons = read_transcripts = add_fovs_as_shapes = False
    else:
        places_fovs = read_images or read_labels or read_polygons or read_transcripts or add_fovs_as_shapes
        if fov_locs is not None and (places_fovs or preview_fovs):
            fov_size_info = _fov_size_px(dataset, fov_size_px)
            fov_size = cast("tuple[float, float]", tuple(fov_size_info["fov_size_px"]))
            fov_size_exact = fov_size_info["fov_size_source"] != FOV_SIZE_LOWER_BOUND
            if fov_size_exact:
                _set_fov_extent(fov_locs, fov_size)
            _check_fov_size_use(
                fov_size_info,
                [dataset],
                fov_locs,
                read_labels=read_labels,
                read_polygons=read_polygons,
                read_transcripts=read_transcripts,
                polygons_as_labels=polygons_as_labels,
            )

    if skip_empty_fovs and (read_images or read_labels):
        fov_locs = _prune_empty_fovs(dataset, fov_locs, fov_set)

    if preview_fovs:
        if fov_locs is None or fov_size is None:
            raise ValueError("preview_fovs=True but no FOV positions file found.")
        _plot_fov_preview(fov_locs, fov_set, fov_size=fov_size)
        return None

    if align_rasters_to_polygons is None:
        align_rasters_to_polygons = bool(read_polygons or (read_labels and polygons_as_labels))

    reader = CosMxDatasetReader(
        dataset,
        fovs=fov_set,
        n_workers=n_workers,
        flip_image=flip_image,
        polygons_as_labels=polygons_as_labels,
        fov_locs=fov_locs,
        align_rasters_to_polygons=align_rasters_to_polygons,
        keep_polygons_after_rasterize=keep_polygons_after_rasterize,
        image_normalization_percentile=image_normalization_percentile,
        fov_size=fov_size,
        fov_size_exact=fov_size_exact,
    )

    # Pre-scan cell_ID max across all available files to avoid
    # load-order dependency crashes in global_cell_id().
    _prescan_max_cell_id(reader, dataset, fov_set, read_transcripts=read_transcripts)

    # --- Read elements ---
    images, shapes = _read_images_and_shapes(
        reader,
        read_images=read_images,
        read_proteins=read_proteins,
        read_polygons=read_polygons,
        image_models_kwargs=image_models_kwargs,
        imread_kwargs=imread_kwargs,
        channels=channels,
    )
    points = reader.read_transcripts() if read_transcripts and dataset.tx_file is not None else {}

    # Label-anchored co-registration: the table annotates the segmentation. Table
    # cells not in the rasterised labels are counted and kept, never dropped.
    # _cosmx_multi runs the same check per modality.
    labels, label_ids = reader.read_labels() if read_labels else ({}, set())

    region_refs = list(labels.keys()) or list(shapes.keys()) or ["cells"]
    tables = reader.read_tables(region_refs, controls=controls) if read_gexp else {}
    if label_ids:
        _count_cells_outside_labels(tables, label_ids, source_label="rasterised labels")

    if add_fovs_as_shapes:
        shapes = {**shapes, **reader.build_fov_shapes()}

    return _assemble_sdata(images, points, labels, shapes, tables, dataset, pixel_size_um, fov_size_info)
