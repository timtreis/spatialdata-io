from __future__ import annotations

import re
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import dask.array as da
import pandas as pd
import tifffile
import zarr
from spatialdata import SpatialData, sanitize_name
from spatialdata._logging import logger
from spatialdata.models import Image2DModel, Labels2DModel
from spatialdata.transformations import Affine, BaseTransformation, Identity, Scale

from spatialdata_io.readers._utils._utils import (
    _fit_scale_factors,
    _initialize_raster_models_kwargs,
    _set_observation_unit,
    _set_reader_metadata,
)

__all__ = ["imc", "imc_mcd"]

# Hyperion/XTi ablate a 1 µm spot per pixel; no IMC export states it
PIXEL_SIZE_UM = 1.0
_CHANNEL_NAME = re.compile(r'<(?:\w+:)?Channel [^>]*Name="([^"]+)"')
_ROI_SUFFIX = re.compile(r"(\.ome)?\.tiff?$|\.zarr(\.zip)?$", re.IGNORECASE)


def imc(
    path: str | Path | Sequence[str | Path],
    *,
    panel: str | Path | None = None,
    labels: Mapping[str, str | Path | Sequence[str | Path]] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
    labels_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> SpatialData:
    """Read *imaging mass cytometry* (IMC) regions of interest (ROIs) exported from MCD files.

    ``path`` is one of:

        - an OME-NGFF ``.zarr`` folder or ``.zarr.zip`` (e.g. IDR/BioImage Archive conversions): the full-resolution
          level, with the ``omero`` channel labels;
        - an OME-TIFF with one single-channel OME-XML document per page (e.g. MCD Viewer exports, which lack
          ``SizeC``), or with one OME-XML document on the first page for all channels (e.g. imctools ``full_stack`` exports): the
          channel names come from the ``<Channel Name>`` entries;
        - a multi-page TIFF or ImageJ hyperstack (e.g. a steinbock ``img/<roi>.tiff``): the channel names are the
          ``name`` column of ``panel`` (steinbock ``panel.csv``) for the rows with ``keep`` 1 (all rows without a
          ``keep`` column), else ``ch0``, ``ch1``, ...

    The image is stored as ``<roi>_image``, ``<roi>`` being the file name without its image extension (``.ome.tiff``,
    ``.zarr.zip``, ...), made a valid element name by :func:`spatialdata.sanitize_name`. IMC ablates 1 µm per pixel,
    so pixels are µm: all elements have ``Identity`` to ``"global"``, and ``sdata.attrs["imc"]`` records
    ``pixel_size_um`` 1.0 with its source.

    ``path`` may also be a sequence of ROI images (e.g. all ROIs of one patient), each read as above into its own
    elements of one :class:`spatialdata.SpatialData`. Exported ROIs carry no stage position, so each ROI then gets
    its own coordinate system named after it.

    Parameters
    ----------
    path
        The ROI image, or a sequence of ROI images.
    panel
        steinbock ``panel.csv`` naming the channels of a TIFF without channel names.
    labels
        Segmentation masks of the ROI by name, e.g. ``{"cells": ".../masks/<roi>.tiff"}``. Each is stored as
        ``<roi>_<name>_labels`` and must have the image's shape. For several ROIs, give one mask per ROI, in the
        order of ``path``.
    image_models_kwargs
        Keyword arguments for :class:`spatialdata.models.Image2DModel`.
    labels_models_kwargs
        Keyword arguments for :class:`spatialdata.models.Labels2DModel`.

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    paths = [Path(path)] if isinstance(path, str | Path) else [Path(p) for p in path]
    roi_labels = {name: [Path(m)] if isinstance(m, str | Path) else [Path(x) for x in m] for name, m in labels.items()}
    if bad := {name: len(m) for name, m in roi_labels.items() if len(m) != len(paths)}:
        raise ValueError(f"Expected one mask per ROI ({len(paths)}), got {bad}.")
    image_models_kwargs, labels_models_kwargs = _initialize_raster_models_kwargs(
        image_models_kwargs, labels_models_kwargs
    )
    images: dict[str, Any] = {}
    label_elements: dict[str, Any] = {}
    for i, roi_path in enumerate(paths):
        cs = "global" if len(paths) == 1 else _roi_name(roi_path)
        im, lab = _read_roi(
            roi_path,
            panel,
            {name: m[i] for name, m in roi_labels.items()},
            {cs: Identity()},
            image_models_kwargs,
            labels_models_kwargs,
        )
        if im.keys() & images.keys():
            raise ValueError(f"Two ROIs are both named {_roi_name(roi_path)}.")
        images |= im
        label_elements |= lab

    sdata = SpatialData(images=images, labels=label_elements)
    sdata.attrs["imc"] = {
        "roi": paths[0].name if len(paths) == 1 else [p.name for p in paths],
        "pixel_size_um": PIXEL_SIZE_UM,
        "pixel_size_source": "instrument spec (Hyperion/XTi ablation)",
    }
    for table in sdata.tables.values():
        _set_observation_unit(table, "cell", None)
    return _set_reader_metadata(sdata, "imc")


def imc_mcd(
    path: str | Path,
    *,
    acquisitions: Sequence[int] | None = None,
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> SpatialData:
    """Read all acquisitions (ROIs) of a raw *IMC* ``.mcd`` file with :mod:`readimc` (extra ``mcd``).

    Each acquisition becomes the image ``<mcd>_<id>_image``, ``<mcd>`` being the file name without its extension,
    with channels named by their antibody label (the metal, e.g. ``Ir191``, where unlabelled). The pixel size is the
    acquisition's ablation distance. Acquisitions with stage coordinates are placed on the slide in µm in
    ``"global"`` (with the stage y axis negated where it points up, so the images keep their orientation); any other
    gets its own coordinate system in µm, named after the image. Acquisitions that cannot be
    read (e.g. aborted ones) are skipped with a warning. ``sdata.attrs["imc"]`` lists each acquisition's id,
    description, pixel size and placement.

    Parameters
    ----------
    path
        The ``.mcd`` file.
    acquisitions
        Acquisition ids to read (e.g. one core of a tissue microarray); all by default.
    image_models_kwargs
        Keyword arguments for :class:`spatialdata.models.Image2DModel`.

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    from readimc import MCDFile

    path = Path(path)
    image_models_kwargs, _ = _initialize_raster_models_kwargs(image_models_kwargs, {})
    stem = sanitize_name(path.stem)
    images, info = {}, []
    with MCDFile(path) as f:
        for slide in f.slides:
            for acq in slide.acquisitions:
                if acquisitions is not None and acq.id not in acquisitions:
                    continue
                try:
                    data = f.read_acquisition(acq)
                except Exception as e:  # noqa: BLE001 - aborted or truncated acquisitions
                    logger.warning(f"{path.name}: skipping acquisition {acq.id} ({acq.description}): {e}")
                    continue
                key = f"{stem}_{acq.id}"
                names = [label or metal for label, metal in zip(acq.channel_labels, acq.channel_names, strict=True)]
                if len(set(names)) < len(names):
                    names = [f"{n}_{m}" for n, m in zip(names, acq.channel_names, strict=True)]
                transform, cs = _stage_transform(acq), "global"
                if transform is None:
                    transform, cs = (
                        Scale([acq.pixel_size_x_um or 1.0, acq.pixel_size_y_um or 1.0], axes=("x", "y")),
                        key,
                    )
                images[f"{key}_image"] = Image2DModel.parse(
                    da.from_array(data),
                    dims=("c", "y", "x"),
                    c_coords=names,
                    transformations={cs: transform},
                    **_fit_scale_factors(image_models_kwargs, data.shape[-2:]),
                )
                info.append(
                    {
                        "id": acq.id,
                        "description": acq.description,
                        "pixel_size_um": acq.pixel_size_x_um,
                        "coordinate_system": cs,
                    }
                )
    if not images:
        raise ValueError(f"No readable acquisition in {path}.")
    sdata = SpatialData(images=images)
    sdata.attrs["imc"] = {
        "mcd": path.name,
        "acquisitions": info,
        "pixel_size_source": "MCD AblationDistanceBetweenShots",
    }
    for table in sdata.tables.values():
        _set_observation_unit(table, "cell", None)
    return _set_reader_metadata(sdata, "imc")


def _stage_transform(acq: Any) -> BaseTransformation | None:
    """Pixels -> slide µm from the acquisition's stage corners (top left, top right, bottom right, bottom left)."""
    corners = acq.roi_coords_um
    if corners is None or not acq.width_px or not acq.height_px:
        return None
    (x0, y0), (x1, y1), _, (x3, y3) = corners
    if y3 < y0:  # the Hyperion stage y axis points up; negate it so the images are not shown upside down
        y0, y1, y3 = -y0, -y1, -y3
    w, h = acq.width_px, acq.height_px
    matrix = [[(x1 - x0) / w, (x3 - x0) / h, x0], [(y1 - y0) / w, (y3 - y0) / h, y0], [0, 0, 1]]
    return Affine(matrix, input_axes=("x", "y"), output_axes=("x", "y"))


def _read_roi(
    path: Path,
    panel: str | Path | None,
    labels: Mapping[str, Path],
    transformations: dict[str, Any],
    image_models_kwargs: Mapping[str, Any],
    labels_models_kwargs: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The image and label elements of one ROI."""
    names: list[str] | None
    if ".zarr" in path.suffixes:
        image, names = _read_ngff(path)
    else:
        image, names = _read_tiff(path)
        if names is None:
            names = _panel_names(Path(panel)) if panel is not None else [f"ch{i}" for i in range(image.shape[0])]
    if len(names) != image.shape[0]:
        raise ValueError(f"{path.name} has {image.shape[0]} channels, but {len(names)} channel names were found.")

    roi = _roi_name(path)
    images = {
        f"{roi}_image": Image2DModel.parse(
            image,
            dims=("c", "y", "x"),
            c_coords=names,
            transformations=transformations,
            **_fit_scale_factors(image_models_kwargs, image.shape[1:]),
        )
    }
    label_elements = {}
    for name, mask_path in labels.items():
        mask = tifffile.imread(mask_path, is_ome=False)
        mask = mask.reshape(mask.shape[-2:]) if mask.ndim > 2 and mask.size == mask.shape[-2] * mask.shape[-1] else mask
        if mask.shape != image.shape[1:]:
            raise ValueError(f"Mask {mask_path} has shape {mask.shape}; expected {image.shape[1:]}.")
        label_elements[f"{roi}_{sanitize_name(name)}_labels"] = Labels2DModel.parse(
            mask,
            dims=("y", "x"),
            transformations=transformations,
            **_fit_scale_factors(labels_models_kwargs, mask.shape),
        )
    return images, label_elements


def _read_tiff(path: Path) -> tuple[da.Array, list[str] | None]:
    """The ``(c, y, x)`` image of a TIFF and its OME channel names (``None`` if there is not one per channel)."""
    # is_ome=False: per-page OME-XML without SizeC breaks tifffile's OME series parsing
    with tifffile.TiffFile(path, is_ome=False) as tf:
        descriptions = [getattr(p, "description", "") or "" for p in tf.pages]
    names = [m.group(1) if (m := _CHANNEL_NAME.search(d)) else None for d in descriptions]
    if not all(names):  # one OME-XML document on the first page for all channels (e.g. imctools stacks)
        names = _CHANNEL_NAME.findall(descriptions[0])
    image = da.from_zarr(zarr.open(tifffile.imread(path, is_ome=False, aszarr=True), mode="r"))
    image = image.reshape((-1, *image.shape[-2:]))
    return image, names if all(names) and len(names) == image.shape[0] else None  # type: ignore[return-value]


def _read_ngff(path: Path) -> tuple[da.Array, list[str]]:
    """The full-resolution ``(c, y, x)`` image of an OME-NGFF ROI and its ``omero`` channel labels."""
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            root = next(n.split("/")[0] for n in zf.namelist() if n.endswith(".zattrs"))
        group = zarr.open_group(zarr.storage.ZipStore(path, mode="r"), path=root, mode="r")
    else:
        group = zarr.open_group(path, mode="r")
    attrs: dict[str, Any] = dict(group.attrs.asdict())
    multiscales = attrs["multiscales"][0]
    axes = [a["name"] if isinstance(a, dict) else a for a in multiscales["axes"]]
    image = da.from_zarr(group[multiscales["datasets"][0]["path"]])
    # drop singleton t/z axes
    index = tuple(0 if a in ("t", "z") else slice(None) for a in axes)
    image = image[index]
    names = [str(c.get("label", f"ch{i}")) for i, c in enumerate(attrs.get("omero", {}).get("channels", []))]
    return image, names or [f"ch{i}" for i in range(image.shape[0])]


def _roi_name(path: Path) -> str:
    """The file name without its image extension (``.ome.tiff``, ``.tiff``, ``.zarr.zip``, ...), as an element name."""
    return sanitize_name(_ROI_SUFFIX.sub("", path.name))


def _panel_names(panel: Path) -> list[str]:
    df = pd.read_csv(panel)
    if "keep" in df.columns:
        df = df[df["keep"].astype(bool)]
    return [str(n) for n in df["name"]]
