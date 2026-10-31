from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import dask.array as da
from spatialdata import SpatialData, sanitize_name
from spatialdata.models import Image2DModel, Labels2DModel
from spatialdata.transformations import BaseTransformation, Identity, Translation

from spatialdata_io.readers._utils._akoya import read_tiff, spatial_chunks
from spatialdata_io.readers._utils._utils import _initialize_raster_models_kwargs, _set_reader_metadata

__all__ = ["mibi"]

_TIFF_SUFFIXES = (".tif", ".tiff")


def mibi(
    path: str | Path,
    *,
    labels: Mapping[str, str | Path] = MappingProxyType({}),
    pixel_size_um: float | None = None,
    fov_positions: Mapping[str, tuple[float, float]] | None = None,
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
    labels_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> SpatialData:
    """Read *MIBI* fields of view (FOVs) in the Angelo lab layout (``toffy``/``ark``).

    ``path`` is the folder of one FOV, holding one single-page ``.tif(f)`` per channel named after the channel
    (e.g. ``CD45.tiff``, ``chan_115.tiff``), as extracted by ``toffy`` and deposited per FOV on the BioImage Archive.
    The channels are stacked lazily, in file-name order, into the image ``<fov>_image``; ``<fov>`` is the folder
    name, made a valid element name by :func:`spatialdata.sanitize_name`. All channel files must share shape and
    dtype.

    ``path`` may instead hold one such folder per FOV (e.g. the tiles of one tissue region): each FOV is read as
    above into its own elements of one :class:`spatialdata.SpatialData`.

    Parameters
    ----------
    path
        Folder of one FOV (e.g. an extracted ``<fov>.zip``).
    labels
        Segmentation masks of the FOV by name, e.g. ``{"whole_cell": ".../<fov>_whole_cell.tiff", "nuclear":
        ".../<fov>_nuclear.tiff"}`` (Mesmer/``deepcell_output``). Each is stored as ``<fov>_<name>_labels`` and must
        have the image's shape. ``{fov}`` in a path is replaced by each FOV's folder name.
    pixel_size_um
        Pixel size in µm, stored in ``sdata.attrs["mibi"]["pixel_size_um"]``. MIBI TIFFs carry none; the depositors
        state it in the methods (e.g. 800 µm over 2048 px, 0.390625 µm).
    fov_positions
        Top-left ``(x, y)`` pixel position of each FOV (by folder name) in a shared frame, e.g. from a stitched cell
        table. The FOVs are then placed with a ``Translation`` in ``"global"``; without it, each FOV of a multi-FOV
        ``path`` gets its own coordinate system named after the FOV.
    image_models_kwargs
        Keyword arguments for :class:`spatialdata.models.Image2DModel`.
    labels_models_kwargs
        Keyword arguments for :class:`spatialdata.models.Labels2DModel`.

    Returns
    -------
    :class:`spatialdata.SpatialData` with the elements in pixel units.
    """
    path = Path(path)
    image_models_kwargs, labels_models_kwargs = _initialize_raster_models_kwargs(
        image_models_kwargs, labels_models_kwargs
    )
    chunks = spatial_chunks(image_models_kwargs["chunks"], (4096, 4096))
    fovs = [path] if _channel_files(path) else sorted(d for d in path.iterdir() if d.is_dir() and _channel_files(d))
    if not fovs:
        raise ValueError(f"No channel .tif(f) files in {path} or its subfolders.")
    if fov_positions is not None and (missing := [d.name for d in fovs if d.name not in fov_positions]):
        raise ValueError(f"FOVs with no position in `fov_positions`: {missing}.")

    images: dict[str, Any] = {}
    label_elements: dict[str, Any] = {}
    for fov in fovs:
        transformations: dict[str, BaseTransformation] = (
            {"global": Translation(list(fov_positions[fov.name]), axes=("x", "y"))}
            if fov_positions is not None
            else {"global" if len(fovs) == 1 else fov.name: Identity()}
        )
        fov_labels = {name: Path(str(p).replace("{fov}", fov.name)) for name, p in labels.items()}
        i, lab = _read_fov(fov, fov_labels, chunks, transformations, image_models_kwargs, labels_models_kwargs)
        images |= i
        label_elements |= lab

    sdata = SpatialData(images=images, labels=label_elements)
    sdata.attrs["mibi"] = {"fov": path.name, "pixel_size_um": pixel_size_um}
    if len(fovs) > 1:
        sdata.attrs["mibi"]["fovs"] = [d.name for d in fovs]
    return _set_reader_metadata(sdata, "mibi")


def _channel_files(path: Path) -> list[Path]:
    return sorted(f for f in path.iterdir() if f.suffix.lower() in _TIFF_SUFFIXES and not f.name.startswith("."))


def _read_fov(
    path: Path,
    labels: Mapping[str, Path],
    chunks: Any,
    transformations: dict[str, BaseTransformation],
    image_models_kwargs: Mapping[str, Any],
    labels_models_kwargs: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The image and label elements of one FOV folder."""
    files = _channel_files(path)
    planes = []
    for file in files:
        plane, _, _ = read_tiff(file, chunks)
        if plane.shape[0] != 1:
            raise ValueError(f"{file} holds {plane.shape[0]} planes; expected one 2-D channel image.")
        planes.append(plane[0])
    if len({(p.shape, p.dtype) for p in planes}) > 1:
        found = {f.name: (p.shape, str(p.dtype)) for f, p in zip(files, planes, strict=True)}
        raise ValueError(f"The channel images of one FOV must share shape and dtype, got {found}.")

    fov = sanitize_name(path.name)
    images = {
        f"{fov}_image": Image2DModel.parse(
            da.stack(planes),
            dims=("c", "y", "x"),
            c_coords=[f.stem for f in files],
            transformations=transformations,
            **image_models_kwargs,
        )
    }
    label_elements = {}
    for name, mask_path in labels.items():
        mask, _, _ = read_tiff(mask_path, chunks)
        if mask.shape[0] != 1 or mask.shape[1:] != planes[0].shape:
            raise ValueError(f"Mask {mask_path} has shape {mask.shape}; expected one plane of {planes[0].shape}.")
        label_elements[f"{fov}_{sanitize_name(name)}_labels"] = Labels2DModel.parse(
            mask[0], dims=("y", "x"), transformations=transformations, **labels_models_kwargs
        )
    return images, label_elements
