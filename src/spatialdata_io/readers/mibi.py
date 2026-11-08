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
from spatialdata_io.readers._utils._utils import (
    _initialize_raster_models_kwargs,
    _set_observation_unit,
    _set_reader_metadata,
)

__all__ = ["mibi"]

_TIFF_SUFFIXES = (".tif", ".tiff")


def mibi(
    path: str | Path,
    *,
    labels: Mapping[str, str | Path | Mapping[str, str | Path]] = MappingProxyType({}),
    channel_folder: str = "",
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

    ``path`` may also be one multichannel OME-TIFF of a FOV (HuBMAP's MIBI deposits): the channels are named from
    its OME-XML, and ``<fov>`` is the file name without ``.ome.tif(f)``.

    Parameters
    ----------
    path
        Folder of one FOV (e.g. an extracted ``<fov>.zip``).
    labels
        Segmentation masks of the FOV by name, e.g. ``{"whole_cell": ".../<fov>_whole_cell.tiff", "nuclear":
        ".../<fov>_nuclear.tiff"}`` (Mesmer/``deepcell_output``). Each is stored as ``<fov>_<name>_labels`` and must
        have the image's shape. ``{fov}`` in a path is replaced by each FOV's folder name; a mapping from FOV folder
        name to path instead gives each FOV its own mask file, for names that ``{fov}`` can't express (e.g.
        ``Point2311_labels.tiff`` for the FOV ``Point2311_pt1053_20698``). A mask file that stacks several masks
        (HuBMAP SPRM's ``mask.ome.tiff``) gives the plane whose OME channel is named like the mask, e.g. ``cells``.
    channel_folder
        Subfolder of each FOV folder that holds the channel files, e.g. ``"TIFs"`` for ``<fov>/TIFs/CD45.tif``
        (Risom et al. 2022), where ``<fov>`` itself also holds compartment masks. The FOV is named after ``<fov>``.
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
    if path.is_file():
        fovs = [path]
    else:
        fovs = (
            [path]
            if _channel_files(path / channel_folder)
            else sorted(d for d in path.iterdir() if _channel_files(d / channel_folder))
        )
    if not fovs:
        raise ValueError(f"No channel .tif(f) files in {path / channel_folder} or its subfolders.")
    if unmasked := [(n, d.name) for n, p in labels.items() if isinstance(p, Mapping) for d in fovs if d.name not in p]:
        raise ValueError(f"(mask, FOV) pairs with no path in `labels`: {unmasked}.")
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
        fov_labels = {
            name: Path(p[fov.name] if isinstance(p, Mapping) else str(p).replace("{fov}", fov.name))
            for name, p in labels.items()
        }
        files = [fov] if fov.is_file() else _channel_files(fov / channel_folder)
        i, lab = _read_fov(fov, files, fov_labels, chunks, transformations, image_models_kwargs, labels_models_kwargs)
        images |= i
        label_elements |= lab

    sdata = SpatialData(images=images, labels=label_elements)
    sdata.attrs["mibi"] = {"fov": path.name, "pixel_size_um": pixel_size_um}
    if len(fovs) > 1:
        sdata.attrs["mibi"]["fovs"] = [d.name for d in fovs]
    for table in sdata.tables.values():
        _set_observation_unit(table, "cell", None)
    return _set_reader_metadata(sdata, "mibi")


def _channel_files(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(f for f in path.iterdir() if f.suffix.lower() in _TIFF_SUFFIXES and not f.name.startswith("."))


def _read_fov(
    path: Path,
    files: list[Path],
    labels: Mapping[str, Path],
    chunks: Any,
    transformations: dict[str, BaseTransformation],
    image_models_kwargs: Mapping[str, Any],
    labels_models_kwargs: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The image and label elements of one FOV folder from its channel ``files``, or of one multichannel OME-TIFF."""
    if path.is_file():
        stack, _, _ = read_tiff(path, chunks)
        names = _ome_channel_names(path)
        if names is None or len(names) != stack.shape[0]:
            raise ValueError(f"{path}: {stack.shape[0]} planes but OME channel names {names}; expected one per plane.")
        planes = list(stack)
    else:
        names = [f.stem for f in files]
        planes = []
        for file in files:
            plane, _, _ = read_tiff(file, chunks)
            if plane.shape[0] != 1:
                raise ValueError(f"{file} holds {plane.shape[0]} planes; expected one 2-D channel image.")
            planes.append(plane[0])
    if len({(p.shape, p.dtype) for p in planes}) > 1:
        found = {f.name: (p.shape, str(p.dtype)) for f, p in zip(files, planes, strict=True)}
        raise ValueError(f"The channel images of one FOV must share shape and dtype, got {found}.")

    fov = sanitize_name(_fov_name(path))
    images = {
        f"{fov}_image": Image2DModel.parse(
            da.stack(planes),
            dims=("c", "y", "x"),
            c_coords=names,
            transformations=transformations,
            **image_models_kwargs,
        )
    }
    label_elements = {}
    for name, mask_path in labels.items():
        mask, _, _ = read_tiff(mask_path, chunks)
        if mask.shape[0] > 1:  # a stack of masks (HuBMAP SPRM): take the plane whose OME channel is named ``name``
            mask_names = [n.lower() for n in _ome_channel_names(mask_path) or []]
            if name.lower() not in mask_names:
                raise ValueError(
                    f"Mask {mask_path} stacks {mask.shape[0]} planes; none is named {name!r}: {mask_names}."
                )
            mask = mask[mask_names.index(name.lower()) : mask_names.index(name.lower()) + 1]
        if mask.shape[0] != 1 or mask.shape[1:] != planes[0].shape:
            raise ValueError(f"Mask {mask_path} has shape {mask.shape}; expected one plane of {planes[0].shape}.")
        label_elements[f"{fov}_{sanitize_name(name)}_labels"] = Labels2DModel.parse(
            mask[0], dims=("y", "x"), transformations=transformations, **labels_models_kwargs
        )
    return images, label_elements


def _fov_name(path: Path) -> str:
    """A FOV folder's name, or an OME-TIFF's file name without ``.ome.tif(f)``."""
    if not path.is_file():
        return path.name
    name = path.name
    for suffix in (".ome.tiff", ".ome.tif", ".tiff", ".tif"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _ome_channel_names(path: Path) -> list[str] | None:
    """The ``Name`` of every ``Channel`` of the first OME ``Image``, or None when there is no OME-XML or a name is missing."""
    import xml.etree.ElementTree as ET

    import tifffile

    with tifffile.TiffFile(path) as tf:
        xml = tf.ome_metadata
    if not xml:
        return None
    image = next((e for e in ET.fromstring(xml) if e.tag.endswith("}Image")), None)
    if image is None:
        return None
    names = [c.get("Name") for c in image.iter() if c.tag.endswith("}Channel")]
    return names if names and all(names) else None
