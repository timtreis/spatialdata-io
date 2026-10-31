"""Tests for the IMC reader on synthetic ROI exports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile
import zarr
from spatialdata import read_zarr
from spatialdata.models import get_channel_names
from spatialdata.transformations import get_transformation

from spatialdata_io import imc, imc_mcd

_FAST = {"scale_factors": [2]}
_DATA = np.random.default_rng(0).random((3, 40, 30), dtype=np.float32)


def _ome_page(name: str) -> str:
    return (
        '<?xml version="1.0"?><OME><Image><Pixels DimensionOrder="XYZTC" Type="float" SizeX="30" SizeY="40">'
        f'<Channel ID="{name}" Name="{name}" SamplesPerPixel="1" /></Pixels></Image></OME>'
    )


def test_steinbock_tiff_with_panel_and_mask_round_trips(tmp_path: Path) -> None:
    (tmp_path / "img").mkdir()
    tifffile.imwrite(tmp_path / "img" / "Batch1_001.tiff", _DATA, imagej=True)
    (tmp_path / "panel.csv").write_text("channel,name,keep\nXe132,132Xe,0\nPr141,aSMA,1\nNd142,CD3,1\nNd143,MBP,1\n")
    mask = np.zeros((40, 30), dtype=np.uint16)
    mask[5:9, 5:9] = 1
    tifffile.imwrite(tmp_path / "mask.tiff", mask)

    sdata = imc(
        tmp_path / "img" / "Batch1_001.tiff",
        panel=tmp_path / "panel.csv",
        labels={"cells": tmp_path / "mask.tiff"},
        image_models_kwargs=_FAST,
        labels_models_kwargs=_FAST,
    )
    image = sdata.images["Batch1_001_image"]
    assert get_channel_names(image) == ["aSMA", "CD3", "MBP"]
    np.testing.assert_array_equal(image["scale0"]["image"].values, _DATA)
    np.testing.assert_array_equal(sdata.labels["Batch1_001_cells_labels"]["scale0"]["image"].values, mask)
    assert sdata.attrs["imc"]["pixel_size_um"] == 1.0
    assert sdata.coordinate_systems == ["global"]
    sdata.write(tmp_path / "x.zarr")
    assert read_zarr(tmp_path / "x.zarr").attrs["spatialdata_io_reader"] == "imc"


def test_per_page_ome_tiff_names(tmp_path: Path) -> None:
    path = tmp_path / "S00T1_ROI2.ome.tiff"
    with tifffile.TiffWriter(path) as tw:
        for plane, name in zip(_DATA, ["80Ar(ArAr80Di)", "Bcatenin(Y89Di)", "CD3(Er170Di)"], strict=True):
            tw.write(plane, description=_ome_page(name), metadata=None)
    sdata = imc(path, image_models_kwargs=_FAST)
    assert get_channel_names(sdata.images["S00T1_ROI2_image"]) == ["80Ar(ArAr80Di)", "Bcatenin(Y89Di)", "CD3(Er170Di)"]
    np.testing.assert_array_equal(sdata.images["S00T1_ROI2_image"]["scale0"]["image"].values, _DATA)


@pytest.mark.parametrize("zipped", [False, True])
def test_ngff_zarr(tmp_path: Path, zipped: bool) -> None:
    root = tmp_path / "ROI01_multichannel.zarr"
    group = zarr.open_group(root, mode="w", zarr_format=2)
    group.create_array("0", data=_DATA[None, :, None])
    group.attrs.update(
        {
            "multiscales": [{"axes": [{"name": a} for a in "tczyx"], "datasets": [{"path": "0"}]}],
            "omero": {"channels": [{"label": n} for n in ["Pr141_aSMA", "Nd142_CD11c", "Nd143_MHC-II"]]},
        }
    )
    path = root
    if zipped:
        import shutil

        path = Path(shutil.make_archive(str(root), "zip", tmp_path, root.name))
        assert path.name == "ROI01_multichannel.zarr.zip"
    sdata = imc(path, image_models_kwargs=_FAST)
    image = sdata.images["ROI01_multichannel_image"]
    assert get_channel_names(image) == ["Pr141_aSMA", "Nd142_CD11c", "Nd143_MHC-II"]
    np.testing.assert_array_equal(image["scale0"]["image"].values, _DATA)


def test_errors(tmp_path: Path) -> None:
    path = tmp_path / "roi.tiff"
    tifffile.imwrite(path, _DATA)
    assert get_channel_names(imc(path)["roi_image"]) == ["ch0", "ch1", "ch2"]
    (tmp_path / "panel.csv").write_text("name\nA\nB\n")
    with pytest.raises(ValueError, match="2 channel names"):
        imc(path, panel=tmp_path / "panel.csv")
    tifffile.imwrite(tmp_path / "mask.tiff", np.zeros((10, 10), dtype=np.uint16))
    with pytest.raises(ValueError, match="expected"):
        imc(path, labels={"cells": tmp_path / "mask.tiff"})


def test_several_rois_each_in_its_own_frame(tmp_path: Path) -> None:
    rois, masks = [], []
    for i in (7, 8):
        tifffile.imwrite(tmp_path / f"img_slide3_00{i}.tiff", _DATA * i, imagej=True)
        tifffile.imwrite(tmp_path / f"masks_slide3_00{i}.tiff", np.full((40, 30), i, dtype=np.uint16))
        rois.append(tmp_path / f"img_slide3_00{i}.tiff")
        masks.append(tmp_path / f"masks_slide3_00{i}.tiff")
    (tmp_path / "panel.csv").write_text("name\naSMA\nCD3\nMBP\n")

    sdata = imc(rois, panel=tmp_path / "panel.csv", labels={"cells": masks}, image_models_kwargs=_FAST)

    assert sorted(sdata.images) == ["img_slide3_007_image", "img_slide3_008_image"]
    assert sdata.labels["img_slide3_008_cells_labels"]["scale0"]["image"].values.max() == 8
    assert sorted(sdata.coordinate_systems) == ["img_slide3_007", "img_slide3_008"]
    assert sdata.attrs["imc"]["roi"] == ["img_slide3_007.tiff", "img_slide3_008.tiff"]
    with pytest.raises(ValueError, match="one mask per ROI"):
        imc(rois, labels={"cells": masks[:1]})


class _Acq:
    def __init__(self, id: int, corners: object) -> None:
        self.id, self.description, self.roi_coords_um = id, f"ROI_{id}", corners
        self.channel_labels, self.channel_names = ["CD3", "", "CD3"], ["Nd142", "Ir191", "Er168"]
        self.width_px, self.height_px, self.pixel_size_x_um, self.pixel_size_y_um = 30, 40, 1.0, 1.0


class _MCD:
    def __init__(self, path: Path) -> None:
        # ROI 1 sits at x 1000-1030 µm, y 2040-2000 µm (stage y points up); ROI 2 has no stage position; ROI 3 aborted
        acqs = [_Acq(1, ((1000, 2040), (1030, 2040), (1030, 2000), (1000, 2000))), _Acq(2, None), _Acq(3, None)]
        self.slides = [type("Slide", (), {"acquisitions": acqs})()]

    def __enter__(self) -> _MCD:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def read_acquisition(self, acq: _Acq) -> np.ndarray:
        if acq.id == 3:
            raise OSError("aborted")
        return _DATA * acq.id


def test_mcd_acquisitions_placed_on_the_slide(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    readimc = pytest.importorskip("readimc")
    monkeypatch.setattr(readimc, "MCDFile", _MCD)

    sdata = imc_mcd(tmp_path / "slide 1.mcd", image_models_kwargs=_FAST)

    assert sorted(sdata.images) == ["slide_1_1_image", "slide_1_2_image"]
    # duplicate labels get their metal; the unlabelled channel keeps its metal
    assert get_channel_names(sdata.images["slide_1_1_image"]) == ["CD3_Nd142", "Ir191_Ir191", "CD3_Er168"]
    t = get_transformation(sdata.images["slide_1_1_image"], "global").to_affine_matrix(("x", "y"), ("x", "y"))
    # the bottom-right pixel corner lands at stage (1030, 2000) with y negated: below the top edge (-2040)
    np.testing.assert_allclose(t @ [30, 40, 1], [1030, -2000, 1])
    np.testing.assert_allclose(t @ [0, 0, 1], [1000, -2040, 1])
    assert set(get_transformation(sdata.images["slide_1_2_image"], get_all=True)) == {"slide_1_2"}
    assert [a["coordinate_system"] for a in sdata.attrs["imc"]["acquisitions"]] == ["global", "slide_1_2"]
