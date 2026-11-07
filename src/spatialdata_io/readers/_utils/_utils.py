from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import scanpy as sc
from anndata.io import read_text
from h5py import File
from spatialdata._logging import logger

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from anndata import AnnData
    from ome_types.model import Pixels
    from spatialdata import SpatialData

PathLike = os.PathLike | str  # type:ignore[type-arg]

# an Ensembl gene id, optionally after a reference prefix (GRCh38_, mm10___) and before a version
_ENSEMBL_ID = re.compile(r"(?:([A-Za-z0-9.]+)_+)?(ENS[A-Z]*G\d+)(?:\.\d+)?")
# reference prefix of a gene id or name in a combined or prefixed reference
_REFERENCE_PREFIX = re.compile(r"^(GRCh\d+|GRCm\d+|hg\d+|mm\d+)_")
# 10x marks retired genes this way; such an id isn't a current gene
_RETIRED_PREFIX = "DEPRECATED"
_ENSEMBL_SPECIES = {"ENSG": "human", "ENSMUSG": "mouse"}


def _read_counts(
    path: str | Path,
    counts_file: str,
    library_id: str | None = None,
    **kwargs: Any,
) -> tuple[AnnData, str]:
    path = Path(path)
    if counts_file.endswith(".h5"):
        adata: AnnData = sc.read_10x_h5(path / counts_file, **kwargs)
        with File(path / counts_file, mode="r") as f:
            attrs = dict(f.attrs)
            if library_id is None:
                try:
                    lid = attrs.pop("library_ids")[0]
                    library_id = lid.decode("utf-8") if isinstance(lid, bytes) else str(lid)
                except ValueError:
                    raise KeyError(
                        "Unable to extract library id from attributes. Please specify one explicitly."
                    ) from None

            adata.uns["spatial"] = {library_id: {"metadata": {}}}  # can overwrite
            for key in ["chemistry_description", "software_version"]:
                if key not in attrs:
                    continue
                metadata = attrs[key].decode("utf-8") if isinstance(attrs[key], bytes) else attrs[key]
                adata.uns["spatial"][library_id]["metadata"][key] = metadata

        return adata, library_id
    if library_id is None:
        raise ValueError("Please explicitly specify `library id`.")

    if counts_file.endswith((".csv", ".txt")):
        adata = read_text(path / counts_file, **kwargs)
    elif counts_file.endswith(".mtx.gz"):
        try:
            from scanpy.readwrite import read_10x_mtx
        except ImportError:
            raise ImportError("Please install scanpy to read 10x mtx files, `pip install scanpy`.") from None
        prefix = counts_file.replace("matrix.mtx.gz", "")
        adata = read_10x_mtx(path, prefix=prefix, **kwargs)
    else:
        raise NotImplementedError("TODO")

    adata.uns["spatial"] = {library_id: {"metadata": {}}}  # can overwrite
    return adata, library_id


def _initialize_raster_models_kwargs(
    image_models_kwargs: Mapping[str, Any], labels_models_kwargs: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    image_models_kwargs = dict(image_models_kwargs)
    if "chunks" not in image_models_kwargs:
        image_models_kwargs["chunks"] = (1, 4096, 4096)
    if "scale_factors" not in image_models_kwargs:
        image_models_kwargs["scale_factors"] = [2, 2, 2, 2]

    labels_models_kwargs = dict(labels_models_kwargs)
    if "chunks" not in labels_models_kwargs:
        labels_models_kwargs["chunks"] = (4096, 4096)
    if "scale_factors" not in labels_models_kwargs:
        labels_models_kwargs["scale_factors"] = [2, 2, 2, 2]
    return image_models_kwargs, labels_models_kwargs


def calc_scale_factors(lower_scale_limit: float, min_size: int = 1000, default_scale_factor: int = 2) -> list[int]:
    """Calculate scale factors based on image size to get lowest resolution under min_size pixels."""
    # get lowest dimension, ignoring channels
    scale_factor: int = default_scale_factor
    scale_factors = [scale_factor]
    lower_scale_limit /= scale_factor
    while lower_scale_limit >= min_size:
        # scale_factors are cumulative, so we don't need to do e.g. scale_factor *= 2
        scale_factors.append(scale_factor)
        lower_scale_limit /= scale_factor
    return scale_factors


def parse_channels(path: Path) -> list[str]:
    """Parse channel names from an OME-TIFF file."""
    from ome_types import from_tiff

    images = from_tiff(path).images
    if len(images) > 1:
        logger.warning("Found multiple images in OME-TIFF file. Only the first one will be used.")
    channels = images[0].pixels.channels
    logger.debug(channels)
    names = [c.name for c in channels if c.name is not None]
    return names


def _set_reader_metadata(sdata: SpatialData, reader: str) -> SpatialData:
    """Set spatialdata-io provenance metadata on a SpatialData object."""
    from spatialdata_io import __version__

    sdata.attrs["spatialdata_io_software_version"] = __version__
    sdata.attrs["spatialdata_io_reader"] = reader
    return sdata


def _add_gene_id_and_genome(adata: AnnData) -> AnnData:
    """Add ``gene_id`` and ``genome_ref`` to ``adata.var``; deposited names, ``gene_ids``, ``genome`` and counts are kept.

    ``gene_id`` is the Ensembl id without reference prefix or version, empty if there is none (symbols aren't mapped to
    ids, and retired ``DEPRECATED_`` ids stay empty). ``genome_ref`` is the ``genome`` column where it names a genome,
    else the reference prefix, else the species of the Ensembl id, else ``unknown``. Tables with several genomes (e.g.
    ``GRCh38`` and ``mm10``) stay as they are, and these columns tell the genes apart.
    """
    names = adata.var_names.astype(str)
    ids = adata.var["gene_ids"].astype(str) if "gene_ids" in adata.var else names
    genome = adata.var["genome"].astype(str) if "genome" in adata.var else [""] * adata.n_vars
    gene_id, genome_ref = [], []
    for i, name, g in zip(ids, names, genome, strict=True):
        m = _ENSEMBL_ID.fullmatch(i)
        retired = m is not None and m[1] == _RETIRED_PREFIX
        gene_id.append(m[2] if m and not retired else None)
        prefix = _REFERENCE_PREFIX.match(i) or _REFERENCE_PREFIX.match(name)
        if g and g not in ("unknown", "nan"):
            genome_ref.append(g)
        elif prefix:
            genome_ref.append(prefix[1])
        else:
            genome_ref.append(
                _ENSEMBL_SPECIES.get(m[2].rstrip("0123456789"), "unknown") if m and not retired else "unknown"
            )
    adata.var["gene_id"], adata.var["genome_ref"] = gene_id, genome_ref
    return adata


def _set_observation_unit(table: AnnData, unit: str | None, size_um: float | None) -> AnnData:
    """Record what a table row is in ``table.uns["spatialdata_io"]``.

    ``unit`` is ``"cell"`` or ``None``; with ``None`` it is derived from ``size_um`` (the side of a square bin or the
    diameter of a spot, in µm): below 10 µm ``"subcellular_bin"``, otherwise ``"spot_or_bin"``.
    """
    if unit is None:
        unit = "subcellular_bin" if size_um is not None and size_um < 10 else "spot_or_bin"
    meta = table.uns.setdefault("spatialdata_io", {})
    meta["observation_unit"] = unit
    meta["unit_size_um"] = None if size_um is None else float(size_um)
    return table


def walk_visible(root: Path) -> Iterator[tuple[Path, list[str], list[str]]]:
    """Like :func:`os.walk` of ``root`` in sorted order, without hidden files and folders, following symlinks.

    Hidden are names starting with ``.`` (also macOS AppleDouble ``._*`` files and ``.Trash-*`` folders) and
    ``__MACOSX`` folders. A folder reached a second time through a symlink (e.g. a link to an ancestor) is skipped.
    The directory names can be pruned in place, as with :func:`os.walk`.
    """
    visited = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in visited:
            dirnames.clear()
            continue
        visited.add(real)
        dirnames[:] = sorted(name for name in dirnames if not name.startswith(".") and name != "__MACOSX")
        yield Path(dirpath), dirnames, sorted(name for name in filenames if not name.startswith("."))


def parse_physical_size(path: Path | None = None, ome_pixels: Pixels | None = None) -> float:
    """Parse physical size from OME-TIFF to micrometer."""
    from ome_types import from_tiff
    from ome_types.model import UnitsLength

    if ome_pixels is not None:
        pixels = ome_pixels
    elif path is not None:
        pixels = from_tiff(path).images[0].pixels
    else:
        raise ValueError("Either `path` or `ome_pixels` must be provided.")
    logger.debug(pixels)
    if pixels.physical_size_x_unit != pixels.physical_size_y_unit:
        logger.error("Physical units for x and y dimensions are not the same.")
        raise NotImplementedError
    if pixels.physical_size_x != pixels.physical_size_y:
        logger.error("Physical sizes for x and y dimensions are not the same.")
        raise NotImplementedError
    if pixels.physical_size_x is None:
        logger.error("Physical size for the x dimension is not set.")
        raise ValueError("The OME-TIFF metadata does not define a physical size for the x dimension.")
    # convert to micrometer if needed
    if pixels.physical_size_x_unit == UnitsLength.NANOMETER:
        physical_size = pixels.physical_size_x / 1000
    elif pixels.physical_size_x_unit == UnitsLength.MICROMETER:
        physical_size = pixels.physical_size_x
    else:
        logger.error(f"Physical unit not recognized: '{pixels.physical_size_x_unit}'.")
        raise NotImplementedError
    return float(physical_size)
