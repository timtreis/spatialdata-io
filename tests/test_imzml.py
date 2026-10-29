"""Tests for the imzML reader on small synthetic processed-mode files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from spatialdata import read_zarr
from spatialdata.transformations import get_transformation

pytest.importorskip("pyimzml")
from pyimzml.ImzMLWriter import ImzMLWriter

from spatialdata_io import imzml

# (x, y): peaks; pixel (2, 2) is not acquired
_SPECTRA = {
    (1, 1): ([100.0, 200.0, 300.0], [1.0, 5.0, 2.0]),
    (2, 1): ([100.0002, 200.0], [3.0, 4.0]),
    (1, 2): ([300.0, 450.0], [1.0, 0.5]),
}


def _write(path: Path, z: int = 1, extra_settings: str = "", extra_instrument: str = "") -> Path:
    with ImzMLWriter(str(path), mode="processed") as w:
        for (x, y), (mz, it) in _SPECTRA.items():
            w.addSpectrum(np.array(mz, dtype=float), np.array(it, dtype=np.float32), (x, y, z))
    text = path.read_text()
    marker = 'name="max count of pixels y"'
    line_end = text.index("/>", text.index(marker)) + 2
    text = text[:line_end] + extra_settings + text[line_end:]
    text = text.replace('<instrumentConfiguration id="IC1">', f'<instrumentConfiguration id="IC1">{extra_instrument}')
    path.write_text(text)
    return path


_PIXEL_SIZE = (
    '<cvParam cvRef="IMS" accession="IMS:1000046" name="pixel size x" value="50" unitName="micrometer"/>'
    '<cvParam cvRef="IMS" accession="IMS:1000047" name="pixel size y" value="50" unitName="micrometer"/>'
)
_ORBITRAP = '<cvParam cvRef="MS" accession="MS:1000484" name="orbitrap"/>'


def test_top_bins_header_pixel_size_and_round_trip(tmp_path: Path) -> None:
    path = _write(tmp_path / "s.imzML", extra_settings=_PIXEL_SIZE, extra_instrument=_ORBITRAP)
    sdata = imzml(path, n_features=3)
    table = sdata["table"]
    uns = table.uns["spatialdata_io"]
    assert (uns["ppm"], uns["ppm_source"], uns["analyser"]) == (5.0, "analyser", "orbitrap")
    assert uns["feature_space"] == "mz_top3_5ppm" and uns["imzml_mode"] == "processed"
    assert (uns["pixel_size_um"], uns["pixel_size_source"], uns["pitch_unknown"]) == (50.0, "imzml header", False)
    assert uns["n_pixels_empty"] == 0 and uns["mz_range"] == [100.0, 450.0]
    # totals: 200 -> 9, 100 (+100.0002, 2 ppm) -> 4, 300 -> 3, 450 -> 0.5 dropped
    np.testing.assert_allclose(table.var["mz"], [100, 200, 300], rtol=5e-6)
    assert list(table.obs_names) == ["1_1", "2_1", "1_2"]
    np.testing.assert_allclose(table.X.toarray(), [[1, 5, 2], [3, 4, 0], [0, 0, 1]])
    assert table.X.dtype == np.float32
    np.testing.assert_allclose(table.obs["tic"], [8, 7, 1.5])
    square = sdata.shapes["msi_pixels"].geometry.iloc[1]
    assert square.bounds == (1, 0, 2, 1)
    scale = get_transformation(sdata.shapes["msi_pixels"], "global").to_affine_matrix(("x", "y"), ("x", "y"))
    np.testing.assert_allclose(np.diag(scale)[:2], [50, 50])
    sdata.write(tmp_path / "x.zarr")
    assert read_zarr(tmp_path / "x.zarr")["table"].uns["spatialdata_io"]["feature_space"] == "mz_top3_5ppm"


def test_deposited_features_and_kwarg_pixel_size(tmp_path: Path) -> None:
    path = _write(tmp_path / "s.imzML")
    (tmp_path / "features.csv").write_text("mz,name\n200.001,PC\n450.0,X\n999,unseen\n")
    table = imzml(path, features=tmp_path / "features.csv", pixel_size_um=10.0)["table"]
    uns = table.uns["spatialdata_io"]
    assert (uns["feature_space"], uns["ppm"], uns["ppm_source"]) == ("deposited", 10.0, "default")
    assert (uns["pixel_size_um"], uns["pixel_size_source"]) == (10.0, "repository text")
    # 200.0 is 5 ppm from 200.001 (within 10 ppm); 100 and 300 match nothing
    np.testing.assert_allclose(table.X.toarray(), [[5, 0, 0], [4, 0, 0], [0, 0.5, 0]])


def test_pitch_unknown_and_3d_error(tmp_path: Path) -> None:
    sdata = imzml(_write(tmp_path / "s.imzML"), ppm=50)
    uns = sdata["table"].uns["spatialdata_io"]
    assert uns["pitch_unknown"] is True and uns["ppm_source"] == "kwarg"
    assert sdata.coordinate_systems == ["global"]
    with pytest.raises(ValueError, match="z planes"):
        imzml(_write(tmp_path / "z.imzML", z=2))


def test_drifting_ion_is_one_feature(tmp_path: Path) -> None:
    path = tmp_path / "d.imzML"
    with ImzMLWriter(str(path), mode="processed") as w:
        # the same ion 5 ppm apart in two pixels (neighbouring bins), and a second ion
        w.addSpectrum(np.array([798.5366, 900.0]), np.array([4.0, 1.0], dtype=np.float32), (1, 1, 1))
        w.addSpectrum(np.array([798.5406]), np.array([3.0], dtype=np.float32), (2, 1, 1))
    table = imzml(path, ppm=5, n_features=2)["table"]
    assert table.n_vars == 2
    np.testing.assert_allclose(table.X.toarray(), [[4, 1], [3, 0]])
