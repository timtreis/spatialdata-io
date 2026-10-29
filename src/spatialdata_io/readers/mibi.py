from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import dask.array as da
from spatialdata import SpatialData, sanitize_name
from spatialdata.models import Image2DModel, Labels2DModel
from spatialdata.transformations import Identity

from spatialdata_io.readers._utils._akoya import read_tiff, spatial_chunks
from spatialdata_io.readers._utils._utils import _initialize_raster_models_kwargs, _set_reader_metadata

__all__ = ["mibi"]

_TIFF_SUFFIXES = (".tif", ".tiff")


def mibi(
    path: str | Path,
    *,
    labels: Mapping[str, str | Path] = MappingProxyType({}),
    pixel_size_um: float | None = None,
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
    labels_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> SpatialData:
    """Read one *MIBI* field of view (FOV) in the Angelo lab layout (``toffy``/``ark``).

    ``path`` is the folder of one FOV, holding one single-page ``.tif(f)`` per channel named after the channel
    (e.g. ``CD45.tiff``, ``chan_115.tiff``), as extracted by ``toffy`` and deposited per FOV on the BioImage Archive.
    The channels are stacked lazily, in file-name order, into the image ``<fov>_image``; ``<fov>`` is the folder
    name, made a valid element name by :func:`spatialdata.sanitize_name`. All channel files must share shape and
    dtype.

    Parameters
    ----------
    path
        Folder of one FOV (e.g. an extracted ``<fov>.zip``).
    labels
        Segmentation masks of the FOV by name, e.g. ``{"whole_cell": ".../<fov>_whole_cell.tiff", "nuclear":
        ".../<fov>_nuclear.tiff"}`` (Mesmer/``deepcell_output``). Each is stored as ``<fov>_<name>_labels`` and must
        have the image's shape.
    pixel_size_um
        Pixel size in µm, stored in ``sdata.attrs["mibi"]["pixel_size_um"]``. MIBI TIFFs carry none; the depositors
        state it in the methods (e.g. 800 µm over 2048 px, 0.390625 µm).
    image_models_kwargs
        Keyword arguments for :class:`spatialdata.models.Image2DModel`.
    labels_models_kwargs
        Keyword arguments for :class:`spatialdata.models.Labels2DModel`.

    Returns
    -------
    :class:`spatialdata.SpatialData` with the elements in pixel units in the ``"global"`` coordinate system.
    """
    path = Path(path)
    files = sorted(f for f in path.iterdir() if f.suffix.lower() in _TIFF_SUFFIXES and not f.name.startswith("."))
    if not files:
        raise ValueError(f"No channel .tif(f) files in {path}.")
    image_models_kwargs, labels_models_kwargs = _initialize_raster_models_kwargs(
        image_models_kwargs, labels_models_kwargs
    )
    chunks = spatial_chunks(image_models_kwargs["chunks"], (4096, 4096))

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
    transformations = {"global": Identity()}
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
        mask, _, _ = read_tiff(Path(mask_path), chunks)
        if mask.shape[0] != 1 or mask.shape[1:] != planes[0].shape:
            raise ValueError(f"Mask {mask_path} has shape {mask.shape}; expected one plane of {planes[0].shape}.")
        label_elements[f"{fov}_{sanitize_name(name)}_labels"] = Labels2DModel.parse(
            mask[0], dims=("y", "x"), transformations=transformations, **labels_models_kwargs
        )

    sdata = SpatialData(images=images, labels=label_elements)
    sdata.attrs["mibi"] = {"fov": path.name, "pixel_size_um": pixel_size_um}
    return _set_reader_metadata(sdata, "mibi")
