"""Tests for the MIBI reader on synthetic per-channel FOV folders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile
from spatialdata import read_zarr
from spatialdata.models import get_channel_names

from spatialdata_io import mibi

_FAST = {"scale_factors": [2]}


def _write_fov(folder: Path, names: list[str], shape: tuple[int, int] = (64, 48)) -> np.ndarray:
    folder.mkdir(parents=True)
    rng = np.random.default_rng(0)
    data = rng.random((len(names), *shape), dtype=np.float32)
    for name, plane in zip(names, data, strict=True):
        tifffile.imwrite(folder / f"{name}.tiff", plane, compression="zlib")
    return data


def test_reads_channels_and_masks_and_round_trips(tmp_path: Path) -> None:
    fov = tmp_path / "TMA2_R1C3-x"
    data = _write_fov(fov, ["CD45", "Au", "chan_115"])
    (fov / "._CD3.tiff").write_bytes(b"")  # macOS AppleDouble file in the zip
    mask = np.zeros((64, 48), dtype=np.uint16)
    mask[5:10, 5:10] = 1
    tifffile.imwrite(tmp_path / "TMA2_R1C3-x_whole_cell.tiff", mask[None])

    sdata = mibi(
        fov,
        labels={"whole_cell": tmp_path / "TMA2_R1C3-x_whole_cell.tiff"},
        pixel_size_um=0.390625,
        image_models_kwargs=_FAST,
        labels_models_kwargs=_FAST,
    )

    image = sdata.images["TMA2_R1C3-x_image"]
    assert get_channel_names(image) == ["Au", "CD45", "chan_115"]
    np.testing.assert_array_equal(image["scale0"]["image"].values, data[[1, 0, 2]])
    np.testing.assert_array_equal(sdata.labels["TMA2_R1C3-x_whole_cell_labels"]["scale0"]["image"].values, mask)
    assert sdata.attrs["mibi"] == {"fov": "TMA2_R1C3-x", "pixel_size_um": 0.390625}
    sdata.write(tmp_path / "x.zarr")
    read = read_zarr(tmp_path / "x.zarr")
    assert get_channel_names(read.images["TMA2_R1C3-x_image"]) == ["Au", "CD45", "chan_115"]
    assert read.attrs["spatialdata_io_reader"] == "mibi"


def test_rejects_mismatched_channels_and_masks(tmp_path: Path) -> None:
    fov = tmp_path / "fov"
    _write_fov(fov, ["CD45"])
    tifffile.imwrite(fov / "CD3.tiff", np.zeros((32, 48), dtype=np.float32))
    with pytest.raises(ValueError, match="share shape and dtype"):
        mibi(fov)

    (fov / "CD3.tiff").unlink()
    tifffile.imwrite(tmp_path / "mask.tiff", np.zeros((32, 48), dtype=np.uint16))
    with pytest.raises(ValueError, match="expected one plane"):
        mibi(fov, labels={"whole_cell": tmp_path / "mask.tiff"})
    with pytest.raises(ValueError, match="No channel"):
        (tmp_path / "empty").mkdir()
        mibi(tmp_path / "empty")
