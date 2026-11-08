"""Tests for the MIBI reader on synthetic per-channel FOV folders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile
from spatialdata import read_zarr
from spatialdata.models import get_channel_names
from spatialdata.transformations import get_transformation

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


def test_reads_a_multi_fov_unit_placed_by_position(tmp_path: Path) -> None:
    unit = tmp_path / "slide4-C2"
    a = _write_fov(unit / "slide4-C2_0_0", ["CD45", "HH3"])
    _write_fov(unit / "slide4-C2_0_1", ["CD45", "HH3"])
    for i in (0, 1):
        tifffile.imwrite(tmp_path / f"slide4-C2_0_{i}_cleaned_mask.tiff", np.full((64, 48), i + 1, dtype=np.uint16))

    sdata = mibi(
        unit,
        labels={"whole_cell": tmp_path / "{fov}_cleaned_mask.tiff"},
        fov_positions={"slide4-C2_0_0": (48, 0), "slide4-C2_0_1": (0, 0)},
        image_models_kwargs=_FAST,
        labels_models_kwargs=_FAST,
    )

    assert sorted(sdata.images) == ["slide4-C2_0_0_image", "slide4-C2_0_1_image"]
    np.testing.assert_array_equal(sdata.images["slide4-C2_0_0_image"]["scale0"]["image"].values, a)
    assert sdata.labels["slide4-C2_0_1_whole_cell_labels"]["scale0"]["image"].values.max() == 2
    t = get_transformation(sdata.images["slide4-C2_0_0_image"], "global")
    np.testing.assert_array_equal(t.to_affine_matrix(("x", "y"), ("x", "y"))[:2, 2], [48, 0])
    assert get_transformation(sdata.labels["slide4-C2_0_0_whole_cell_labels"], "global") == t
    assert sdata.attrs["mibi"]["fovs"] == ["slide4-C2_0_0", "slide4-C2_0_1"]

    # without positions every FOV keeps its own frame, so unrelated FOVs never overlap
    sdata = mibi(unit, image_models_kwargs=_FAST)
    assert set(get_transformation(sdata.images["slide4-C2_0_1_image"], get_all=True)) == {"slide4-C2_0_1"}
    with pytest.raises(ValueError, match="no position"):
        mibi(unit, fov_positions={"slide4-C2_0_0": (0, 0)})


def test_reads_channels_from_a_subfolder_with_per_fov_mask_paths(tmp_path: Path) -> None:
    # Risom et al. 2022 (Mendeley d87vg86zd8): raw_tifs_masks/<fov>/TIFs/<channel>.tif beside compartment masks in
    # <fov>/, cell labels in a separate tree named by the "Point" prefix only
    root = tmp_path / "raw_tifs_masks"
    seg = tmp_path / "Segmetation_Outlines_and_Labels_Mendeley"
    seg.mkdir()
    fovs = ["Point2310_pt1052_20696", "Point4127_pt1052_31842"]
    for i, fov in enumerate(fovs):
        _write_fov(root / fov / "TIFs", ["CD45", "Au"])
        (root / fov / "TIFs" / ".DS_Store").write_bytes(b"")
        tifffile.imwrite(root / fov / "epi_mask.tif", np.full((64, 48), 255, dtype=np.uint8))
        tifffile.imwrite(seg / f"{fov.split('_')[0]}_labels.tiff", np.full((64, 48), i + 1, dtype=np.int16))

    cells = {fov: seg / f"{fov.split('_')[0]}_labels.tiff" for fov in fovs}
    sdata = mibi(
        root,
        channel_folder="TIFs",
        labels={"whole_cell": cells, "epithelium": root / "{fov}" / "epi_mask.tif"},
        image_models_kwargs=_FAST,
        labels_models_kwargs=_FAST,
    )

    assert sorted(sdata.images) == [f"{fov}_image" for fov in fovs]
    assert get_channel_names(sdata.images[f"{fovs[0]}_image"]) == ["Au", "CD45"]
    assert sdata.labels[f"{fovs[1]}_whole_cell_labels"]["scale0"]["image"].values.max() == 2
    assert f"{fovs[0]}_epithelium_labels" in sdata.labels
    sdata.write(tmp_path / "x.zarr")

    # one FOV folder on its own, as the hub extracts it for a one-FOV record
    one = mibi(root / fovs[0], channel_folder="TIFs", labels={"whole_cell": cells[fovs[0]]})
    assert list(one.images) == [f"{fovs[0]}_image"]
    with pytest.raises(ValueError, match="no path in `labels`"):
        mibi(root, channel_folder="TIFs", labels={"whole_cell": {fovs[0]: cells[fovs[0]]}})


def test_hubmap_multichannel_ome_tiff_with_stacked_masks(tmp_path: Path) -> None:
    # HuBMAP SPRM: one multichannel OME-TIFF per FOV, channel names in the OME-XML; masks stacked in one OME-TIFF
    rng = np.random.default_rng(0)
    expr = rng.integers(0, 255, size=(3, 40, 30), dtype=np.uint8)
    tifffile.imwrite(
        tmp_path / "3D_image_stack_expr.ome.tiff",
        expr,
        ome=True,
        metadata={"axes": "CYX", "Channel": {"Name": ["CD45", "Ki67", "dsDNA"]}},
    )
    masks = np.stack([np.full((40, 30), i, dtype=np.uint32) for i in (1, 2, 3, 4)])
    names = ["cells", "nuclei", "cell_boundaries", "nucleus_boundaries"]
    tifffile.imwrite(tmp_path / "mask.ome.tiff", masks, ome=True, metadata={"axes": "CYX", "Channel": {"Name": names}})
    sdata = mibi(tmp_path / "3D_image_stack_expr.ome.tiff", labels={"cells": tmp_path / "mask.ome.tiff"})
    image = sdata.images["3D_image_stack_expr_image"]
    assert list(get_channel_names(image)) == ["CD45", "Ki67", "dsDNA"]
    level0 = lambda el: el["scale0"].image if hasattr(el, "keys") else el
    np.testing.assert_array_equal(level0(image).data.compute(), expr)
    np.testing.assert_array_equal(level0(sdata.labels["3D_image_stack_expr_cells_labels"]).data.compute(), masks[0])
    with pytest.raises(ValueError, match="none is named 'whole_cell'"):
        mibi(tmp_path / "3D_image_stack_expr.ome.tiff", labels={"whole_cell": tmp_path / "mask.ome.tiff"})
