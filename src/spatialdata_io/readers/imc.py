from __future__ import annotations

import re
import zipfile
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import dask.array as da
import pandas as pd
import tifffile
import zarr
from spatialdata import SpatialData, sanitize_name
from spatialdata.models import Image2DModel, Labels2DModel
from spatialdata.transformations import Identity

from spatialdata_io.readers._utils._utils import _initialize_raster_models_kwargs, _set_reader_metadata

__all__ = ["imc"]

# Hyperion/XTi ablate a 1 µm spot per pixel; no IMC export states it
PIXEL_SIZE_UM = 1.0
_CHANNEL_NAME = re.compile(r'<Channel [^>]*Name="([^"]+)"')


def imc(
    path: str | Path,
    *,
    panel: str | Path | None = None,
    labels: Mapping[str, str | Path] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
    labels_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> SpatialData:
    """Read one *imaging mass cytometry* (IMC) region of interest (ROI) exported from MCD files.

    ``path`` is one of:

        - an OME-NGFF ``.zarr`` folder or ``.zarr.zip`` (e.g. IDR/BioImage Archive conversions): the full-resolution
          level, with the ``omero`` channel labels;
        - an OME-TIFF with one single-channel OME-XML document per page (e.g. MCD Viewer exports, which lack
          ``SizeC``): the channel names come from each page's ``<Channel Name>``;
        - a multi-page TIFF or ImageJ hyperstack (e.g. a steinbock ``img/<roi>.tiff``): the channel names are the
          ``name`` column of ``panel`` (steinbock ``panel.csv``) for the rows with ``keep`` 1 (all rows without a
          ``keep`` column), else ``ch0``, ``ch1``, ...

    The image is stored as ``<roi>_image``, ``<roi>`` being the file name without its extensions, made a valid
    element name by :func:`spatialdata.sanitize_name`. IMC ablates 1 µm per pixel, so pixels are µm: all elements
    have ``Identity`` to ``"global"``, and ``sdata.attrs["imc"]`` records ``pixel_size_um`` 1.0 with its source.

    Parameters
    ----------
    path
        The ROI image.
    panel
        steinbock ``panel.csv`` naming the channels of a TIFF without channel names.
    labels
        Segmentation masks of the ROI by name, e.g. ``{"cells": ".../masks/<roi>.tiff"}``. Each is stored as
        ``<roi>_<name>_labels`` and must have the image's shape.
    image_models_kwargs
        Keyword arguments for :class:`spatialdata.models.Image2DModel`.
    labels_models_kwargs
        Keyword arguments for :class:`spatialdata.models.Labels2DModel`.

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    path = Path(path)
    names: list[str] | None
    if ".zarr" in path.suffixes:
        image, names = _read_ngff(path)
    else:
        image, names = _read_tiff(path)
        if names is None:
            names = _panel_names(Path(panel)) if panel is not None else [f"ch{i}" for i in range(image.shape[0])]
    if len(names) != image.shape[0]:
        raise ValueError(f"{path.name} has {image.shape[0]} channels, but {len(names)} channel names were found.")

    image_models_kwargs, labels_models_kwargs = _initialize_raster_models_kwargs(
        image_models_kwargs, labels_models_kwargs
    )
    roi = sanitize_name(path.name.split(".")[0])
    transformations = {"global": Identity()}
    images = {
        f"{roi}_image": Image2DModel.parse(
            image, dims=("c", "y", "x"), c_coords=names, transformations=transformations, **image_models_kwargs
        )
    }
    label_elements = {}
    for name, mask_path in labels.items():
        mask = tifffile.imread(mask_path, is_ome=False).squeeze()
        if mask.shape != image.shape[1:]:
            raise ValueError(f"Mask {mask_path} has shape {mask.shape}; expected {image.shape[1:]}.")
        label_elements[f"{roi}_{sanitize_name(name)}_labels"] = Labels2DModel.parse(
            mask, dims=("y", "x"), transformations=transformations, **labels_models_kwargs
        )

    sdata = SpatialData(images=images, labels=label_elements)
    sdata.attrs["imc"] = {
        "roi": path.name,
        "pixel_size_um": PIXEL_SIZE_UM,
        "pixel_size_source": "instrument spec (Hyperion/XTi ablation)",
    }
    return _set_reader_metadata(sdata, "imc")


def _read_tiff(path: Path) -> tuple[da.Array, list[str] | None]:
    """The ``(c, y, x)`` image of a TIFF and its per-page OME channel names (``None`` if not every page has one)."""
    # is_ome=False: per-page OME-XML without SizeC breaks tifffile's OME series parsing
    with tifffile.TiffFile(path, is_ome=False) as tf:
        names = [
            m.group(1) if (m := _CHANNEL_NAME.search(getattr(p, "description", "") or "")) else None for p in tf.pages
        ]
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


def _panel_names(panel: Path) -> list[str]:
    df = pd.read_csv(panel)
    if "keep" in df.columns:
        df = df[df["keep"].astype(bool)]
    return [str(n) for n in df["name"]]
